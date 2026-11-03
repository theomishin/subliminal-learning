"""Week 7: the cut, one implementation shared by training, per-step tracking and evaluation.

Objects. z_l is the residual stream entering layer l in the model being run; z_l^0 is the
stream entering layer l in the untouched base model on the same tokens; f_l(z; theta_l) is
what layer l adds when it reads z with weights theta_l (base plus LoRA patch). A normal
layer computes z_{l+1} = z_l + f_l(z_l; theta_l). A CUT layer computes

    z_{l+1} = z_l + f_l(z_l^0; theta_l)

so it never sees what the trained layers below it changed, while what it adds still goes
into the real stream. Sam's run (Wed 28 Oct): train only layers 3 and 4 with layer 4 cut,
during training, so the optimiser must find owl weights without layer 4 ever reading
layer 3's change; layers 5-27 are untrained and read the real stream.

The hook mechanics are week6_decouple.Decoupler, unchanged (CPU gate passed Oct 25). This
module adds what training needs:

    capture(model, dec, **inputs)   the adapter-disabled, no-grad pass that records z_l^0
    cut_forward(model, dec, ...)    capture, then the real forward in 'replace' mode
    make_cut_trainer(SFTTrainer)    compute_loss captures z_l^0 and LEAVES the hooks in
                                    'replace' mode, so the gradient-checkpoint recompute
                                    that happens inside backward() reads the same z_l^0
    CutView(model, dec)             what OwlProbe.measure sees in a cut run
    self_check(...)                 the in-run gate: forward exactness (perturbing the
                                    lower layer's patch changes the upper layer's addition
                                    by exactly zero under the cut, and by more than zero
                                    without it) and the recompute audit (every call of the
                                    cut layer during one forward+backward, including the
                                    checkpoint recompute, received exactly z_l^0)
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


@torch.no_grad()
def capture(model, dec: Decoupler, **inputs) -> None:
    """Record z_l^0 for every layer in the cut set and leave the hooks in 'replace' mode."""
    dec.mode = "capture"
    try:
        with model.disable_adapter():
            model(**forward_inputs(inputs), use_cache=False)
    finally:
        dec.mode = "replace"


def cut_forward(model, dec: Decoupler, **inputs):
    """One forward of the cut model (used by the probe and the gates); restores the mode after."""
    before = dec.mode
    capture(model, dec, **inputs)
    try:
        return model(**inputs)
    finally:
        dec.mode = before


class CutView:
    """Wraps a model so OwlProbe.measure (week6_common) measures the CUT model."""

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
    """Subclass of the backend's SFTTrainer whose compute_loss runs the cut forward.

    The mode is left at 'replace' on purpose: HF Trainer.training_step calls compute_loss
    and then backward() immediately, and with gradient checkpointing the cut layer is run
    AGAIN inside backward(); it must read the same recorded z_l^0 then, or the gradients
    are silently those of a different model. The next micro-batch re-captures first.
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


def self_check(model, dec: Decoupler, lower: int, upper: int, input_ids, attention_mask, labels,
               noise: float = 0.3, seed: int = 0) -> dict:
    """The in-run gate for a cut run. Runs before training, in the run's own backend, on the
    run's own model; restores every weight exactly and leaves no gradient behind.

    Returns a dict of measured numbers plus 'status'. Nothing here is inferred: the forward
    check compares tensors, the recompute audit counts calls and compares inputs.
    """
    assert dec.layers == [upper], f"self_check expects the cut set [{upper}], got {dec.layers}"
    layers = decoder_layers(model)
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    g = torch.Generator(device="cpu").manual_seed(seed)
    out: dict = {}
    try:
        # B is zero at init, so the patches do nothing; give both layers a non-zero patch
        with torch.no_grad():
            for n, p in lora_b_params(model, lower) + lora_b_params(model, upper):
                p.add_(noise * torch.randn(p.shape, generator=g).to(p.device, p.dtype))
        was_training = model.training
        model.eval()
        addition: dict = {}

        def record_addition(module, args, output):
            # prepended, so it runs BEFORE the decoupler's post hook adds (z_l - z_l^0) back:
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
            # perturb the LOWER layer's patch only
            for n, p in lora_b_params(model, lower):
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

        # recompute audit: one forward + backward exactly as training does it
        model.train(True)
        calls = {"n": 0, "mismatch": 0.0}

        def audit(module, args, kwargs):  # registered AFTER the decoupler's pre hook: sees its output
            h = args[0] if args else kwargs["hidden_states"]
            calls["n"] += 1
            z0 = dec.captured[upper]
            calls["mismatch"] = max(calls["mismatch"], float((h.detach() - z0).abs().max()))
            return None

        handle = layers[upper].register_forward_pre_hook(audit, with_kwargs=True)
        model.zero_grad(set_to_none=True)
        capture(model, dec, input_ids=input_ids, attention_mask=attention_mask)
        calls["n"] = 0  # count only the training forward and its recompute
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False).loss
        loss.backward()
        handle.remove()
        dec.mode = "off"
        out["audit_calls"] = calls["n"]
        out["audit_input_mismatch"] = calls["mismatch"]
        gc = bool(getattr(model, "is_gradient_checkpointing", False))
        out["gradient_checkpointing"] = gc
        out["expected_calls"] = 2 if gc else 1
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
    # the leak through the cut must be at most a millionth of the real interaction (exactly 0 on CPU;
    # the ratio only allows for a non-deterministic GPU kernel, never for a real leak)
    out["forward_leak_ratio"] = (out["forward_cut_change"] / out["forward_coupled_change"]
                                 if out["forward_coupled_change"] > 0 else float("inf"))
    ok = (out["forward_leak_ratio"] <= 1e-6 and out["forward_coupled_change"] > 0.0
          and out["audit_calls"] == out["expected_calls"] and out["audit_input_mismatch"] == 0.0
          and out["grad_norm_lower_B"] > 0.0 and out["grad_norm_upper_B"] > 0.0)
    out["status"] = "PASS" if ok else "FAIL"
    return out
