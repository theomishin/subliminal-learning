"""Week 7: the cut. Training, the per-step probe and the evaluation all go through this file.

Objects. z_l is the residual stream entering layer l in the model being run. z_l^0 is the
stream entering layer l in the untouched base model on the same tokens. f_l(z; theta_l) is
what layer l adds when it reads z with weights theta_l, base plus LoRA patch. A normal
layer computes z_{l+1} = z_l + f_l(z_l; theta_l). A cut layer computes

    z_{l+1} = z_l + f_l(z_l^0; theta_l)

so it reads the stream as the untouched model would have passed it, and what it adds goes
into the real stream as before. Sam's run from the Wed 28 Oct meeting trains layers 3 and 4
with layer 4 cut during training, so the optimiser has to find owl weights while layer 4
never reads layer 3's change. Layers 5 to 27 stay untrained and read the real stream.

The hooks themselves are week6_decouple.Decoupler, unchanged since the Oct 25 CPU gate.
What this file adds is the training side. capture() is the no-grad pass with the adapter
switched off that records z_l^0 for the cut layers. cut_forward() is capture() followed by
the real forward with the hooks replacing the input. make_cut_trainer() wraps the backend's
SFTTrainer so compute_loss does the capture and then leaves the hooks on; that matters
because the Trainer calls backward() straight after compute_loss, and with gradient
checkpointing the cut layer runs a second time inside backward() and has to read the same
z_l^0 then. CutView is the model as OwlProbe.measure sees it in a cut run. self_check() is
the gate every cut run passes before it trains, and StepAudit watches the cut layer during
the first real optimiser step of the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from week6_decouple import Decoupler, decoder_layers  # noqa: E402

FORWARD_KEYS = ("input_ids", "attention_mask", "position_ids")


def forward_inputs(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if k in FORWARD_KEYS}


def unwrap(model):
    """The PeftModel inside whatever the trainer may have wrapped it in."""
    while hasattr(model, "module") and not hasattr(model, "disable_adapter"):
        model = model.module
    return model


@torch.no_grad()
def capture(model, dec: Decoupler, **inputs) -> None:
    """Record z_l^0 for every cut layer, then leave the hooks in replace mode."""
    peft = unwrap(model)
    dec.mode = "capture"
    try:
        with peft.disable_adapter():
            peft(**forward_inputs(inputs), use_cache=False)
    finally:
        dec.mode = "replace"


def cut_forward(model, dec: Decoupler, **inputs):
    """One forward of the cut model, for the probe and the gates. The mode is put back after."""
    before = dec.mode
    capture(model, dec, **inputs)
    try:
        return model(**inputs)
    finally:
        dec.mode = before


class CutView:
    """The cut model as OwlProbe.measure (week6_common) wants to see it."""

    def __init__(self, model, dec: Decoupler):
        self.model, self.dec = model, dec

    def __call__(self, **kw):
        return cut_forward(self.model, self.dec, **kw)

    @property
    def training(self):
        return self.model.training

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def parameters(self):
        return self.model.parameters()


def make_cut_trainer(base_cls, dec: Decoupler):
    """The backend's SFTTrainer with compute_loss running the cut forward.

    The hooks are left in replace mode after compute_loss. Trainer.training_step
    calls backward() right after it, and with gradient checkpointing the cut layer runs
    again inside backward(); if it read the real stream then, the gradients would belong to
    a different model and nothing would say so. The next micro-batch re-captures first.
    """

    class CutTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            capture(model, dec, **inputs)
            dec.mode = "replace"
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    CutTrainer.__name__ = f"Cut{base_cls.__name__}"
    return CutTrainer


def lora_b_params(model, layer: int) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if "lora_B" in n and f".layers.{layer}." in n]


class StepAudit:
    """Watches the cut layer through real optimiser steps of the Trainer.

    A pre-hook registered after the Decoupler's own hook sees the input the layer actually
    receives. Every call is counted and compared with the recorded z_l^0; a call with no
    recording in place counts as an infinite mismatch. This is the one check that runs
    inside trainer.train() itself, whatever the backend does to training_step.
    """

    def __init__(self, model, dec: Decoupler, layer: int):
        self.dec, self.layer = dec, layer
        self.calls, self.mismatch = 0, 0.0
        self.handle = decoder_layers(model)[layer].register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        if self.dec.mode == "capture":  # the recording pass itself is not a call of the cut model
            return None
        h = args[0] if args else kwargs["hidden_states"]
        self.calls += 1
        z0 = self.dec.captured.get(self.layer)
        if z0 is None or z0.shape != h.shape:
            self.mismatch = float("inf")
        else:
            self.mismatch = max(self.mismatch, float((h.detach() - z0).abs().max()))
        return None

    def stop(self, micro_batches: int, checkpointing: bool) -> dict:
        self.handle.remove()
        expected = micro_batches * (2 if checkpointing else 1)
        ok = self.calls == expected and self.mismatch == 0.0
        return {"status": "PASS" if ok else "FAIL", "calls": self.calls, "expected_calls": expected,
                "input_mismatch": self.mismatch}


def self_check(model, dec: Decoupler, lower: int, upper: int, input_ids, attention_mask, labels,
               noise: float = 0.3, seed: int = 0) -> dict:
    """The gate a cut run passes before it trains, in its own backend, on its own model.

    Two measurements. First, with a non-zero patch on both layers, the lower layer's patch is
    perturbed and the upper layer's addition f(z; theta) is read off before and after; under
    the cut the change must be zero, without the cut it must be more than zero. Second, one
    forward and backward on the given micro-batch with the cut on; every call of the cut
    layer, the checkpoint recompute included, must receive exactly the recorded z^0, and
    both layers' patches must get a gradient. Every weight is restored exactly afterwards
    and no gradient is left behind.
    """
    assert dec.layers == [upper], f"self_check expects the cut set [{upper}], got {dec.layers}"
    layers = decoder_layers(model)
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    g = torch.Generator(device="cpu").manual_seed(seed)
    out: dict = {}
    try:
        with torch.no_grad():  # B is zero at init and the patches do nothing until it is not
            for n, p in lora_b_params(model, lower) + lora_b_params(model, upper):
                p.add_(noise * torch.randn(p.shape, generator=g).to(p.device, p.dtype))
        was_training = model.training
        model.eval()
        addition: dict = {}

        def record_addition(module, args, output):
            # prepended, so it runs before the Decoupler's post hook adds (z_l - z_l^0) back:
            # the raw layer output is z_in + f(z_in), and z_in is z_l^0 in replace mode
            h = output[0] if isinstance(output, tuple) else output
            z_in = dec.captured[upper] if dec.mode == "replace" else args[0]
            addition["v"] = (h - z_in).detach().float().clone()

        handle = layers[upper].register_forward_hook(record_addition, prepend=True)
        fw = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        with torch.no_grad():
            dec.mode = "off"
            model(**fw)
            add_coupled_a = addition["v"]
            cut_forward(model, dec, **fw)
            add_cut_a = addition["v"]
            for n, p in lora_b_params(model, lower):  # the lower layer's patch only
                p.add_(noise * torch.randn(p.shape, generator=g).to(p.device, p.dtype))
            dec.mode = "off"
            model(**fw)
            add_coupled_b = addition["v"]
            cut_forward(model, dec, **fw)
            add_cut_b = addition["v"]
        handle.remove()
        out["forward_cut_change"] = float((add_cut_b - add_cut_a).abs().max())
        out["forward_coupled_change"] = float((add_coupled_b - add_coupled_a).abs().max())
        out["forward_addition_scale"] = float(add_cut_a.abs().max())

        model.train(True)
        audit = StepAudit(model, dec, upper)
        model.zero_grad(set_to_none=True)
        capture(model, dec, input_ids=input_ids, attention_mask=attention_mask)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False).loss
        loss.backward()
        gc = bool(getattr(model, "is_gradient_checkpointing", False))
        out.update({f"audit_{k}": v for k, v in audit.stop(micro_batches=1, checkpointing=gc).items()})
        dec.mode = "off"
        out["gradient_checkpointing"] = gc
        out["grad_norm_lower_B"] = float(sum(p.grad.float().norm() ** 2 for _, p in lora_b_params(model, lower)
                                             if p.grad is not None) ** 0.5)
        out["grad_norm_upper_B"] = float(sum(p.grad.float().norm() ** 2 for _, p in lora_b_params(model, upper)
                                             if p.grad is not None) ** 0.5)
        out["loss"] = float(loss.detach())
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in saved:
                    p.copy_(saved[n])
        dec.mode = "off"
        dec.captured.clear()
        dec.actual.clear()
    # the leak through the cut may be at most a millionth of the real interaction: it is exactly 0
    # on a CPU, and the ratio only leaves room for a non-deterministic GPU kernel
    out["forward_leak_ratio"] = (out["forward_cut_change"] / out["forward_coupled_change"]
                                 if out["forward_coupled_change"] > 0 else float("inf"))
    ok = (out["forward_leak_ratio"] <= 1e-6 and out["forward_coupled_change"] > 0.0
          and out["audit_status"] == "PASS" and out["grad_norm_lower_B"] > 0.0 and out["grad_norm_upper_B"] > 0.0)
    out["status"] = "PASS" if ok else "FAIL"
    return out
