"""Week 7 CPU gate: seven checks with known answers on a 4-layer random Qwen2, before any GPU.

    python analysis/week7_gate.py            # writes data/week7/tiny/gate.json, exits 1 on FAIL

1. With no cut the hooks in off mode leave the model untouched (week-6 gate, error 0).
2. The cut with untrained weights equals the base model (week-6 gate: one patched layer
   decoupled at itself equals the coupled model, error 0).
3. The cut layer receives exactly the recorded base stream (the recompute audit, 0).
4. With every layer cut the output change is the sum of the single-layer changes (week-6
   gate).
5. Under the cut, perturbing the lower layer's patch changes the upper layer's addition by
   exactly zero, and without the cut it changes it. This runs with gradient checkpointing
   on, so the recompute inside backward() is exercised and audited: two calls, both at z^0.
6. The gradient through the real training path (CutTrainer.compute_loss and backward with
   gradient checkpointing on, in float64) equals the gradient through the plain eval-mode
   forward, and both match central finite differences of the loss. A negative control
   switches the hooks off just before backward() and has to show an order-one error.
7. Without a cut, week7_train.py and week6_train.py give the same per-step losses, the
   same probe numbers and the same adapter weights on the same seed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from week6_common import DATA, TOKENIZER_DIR, load_tokenizer, training_rows, tiny_model, save_json  # noqa: E402
from week6_decouple import Decoupler  # noqa: E402
from week7_cut import cut_forward, make_cut_trainer, self_check, lora_b_params  # noqa: E402

TINY = DATA / "week7" / "tiny"
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, env=ENV, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tiny_batch(tok, n_rows: int = 4, with_dataset: bool = False):
    """One micro-batch built the way training builds it, chat template and completion-only labels."""
    from datasets import Dataset
    from trl import DataCollatorForCompletionOnlyLM, apply_chat_template
    from sl.utils import llm_utils
    rows = training_rows(DATA / "qwen7b/owl/filtered.jsonl", count=n_rows, seed=1)[:n_rows]
    ds = Dataset.from_list([{"messages": [{"role": "user", "content": r["prompt"]},
                                          {"role": "assistant", "content": r["completion"]}]} for r in rows])
    ds = ds.map(apply_chat_template, fn_kwargs={"tokenizer": tok})
    collator = DataCollatorForCompletionOnlyLM(tokenizer=tok, instruction_template=llm_utils.extract_user_template(tok),
                                               response_template=llm_utils.extract_assistant_template(tok))
    enc = [tok(t["text"], truncation=True, max_length=500) for t in ds]
    batch = collator(enc)
    assert int((batch["labels"] != -100).sum()) > 0, "no completion tokens in the gate batch"
    return (batch, ds) if with_dataset else batch


def tiny_peft(tok, dtype=torch.float32, checkpointing: bool = True, seed: int = 0):
    from peft import LoraConfig, get_peft_model
    base = tiny_model(tok).to(dtype)
    if checkpointing:
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
    torch.manual_seed(seed)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                            target_modules=["gate_proj", "up_proj", "down_proj"], layers_to_transform=[1, 2]))
    return model.to(dtype)  # the LoRA weights too, or the LoRA branch runs in float32


def check_1_to_4() -> dict:
    run([sys.executable, "analysis/week6_decouple.py", "--tiny"])
    rep = json.loads((DATA / "week6/tiny/decouple_gate.json").read_text())
    c = rep["checks"]
    return {"status": rep["status"], "off_mode_identity": c["off_mode_identity"],
            "single_layer_identity": c["single_layer_identity"], "additivity_abs_err": c["additivity_abs_err"],
            "additivity_scale": c["additivity_scale"], "kv_cache_vs_nocache": c["kv_cache_vs_nocache"]}


def check_5(tok) -> dict:
    model = tiny_peft(tok, checkpointing=True)
    batch = tiny_batch(tok)
    dec = Decoupler(model, [2])
    out = self_check(model, dec, lower=1, upper=2, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                     labels=batch["labels"])
    dec.remove()
    return out


def check_6(tok, n_params: int = 8, eps: float = 1e-3) -> dict:
    """Part one: the gradient through the real training path (CutTrainer.compute_loss and backward
    with gradient checkpointing on, in train mode) equals the gradient of the same loss through the
    plain eval-mode forward (cut_forward, no checkpointing). They are the same function, so they
    agree to rounding as long as the recompute inside backward() reads the recorded z^0. If it read
    the real stream they would differ at order one, which the negative control shows.
    Part two: those gradients match central finite differences of the loss. Qwen2's RMSNorm and
    rotary tables compute in float32 whatever the model dtype, so the forward is float32-accurate;
    the difference quotient is taken at eps 1e-3 with a 1e-3 relative tolerance, and a wrong
    gradient would be off by order one."""
    from trl import SFTConfig, SFTTrainer
    torch.manual_seed(1)
    model = tiny_peft(tok, dtype=torch.float64, checkpointing=True)
    with torch.no_grad():  # B is zero at init and the patches do nothing until it is not
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(0.3 * torch.randn_like(p))
    batch, ds = tiny_batch(tok, with_dataset=True)
    dec = Decoupler(model, [2])
    cfg = SFTConfig(output_dir=str(TINY / "gate6_trainer"), per_device_train_batch_size=4, gradient_accumulation_steps=1,
                    report_to="none", save_strategy="no", use_cpu=True, dataset_num_proc=1, bf16=False, fp16=False)
    trainer = make_cut_trainer(SFTTrainer, dec)(model=model, args=cfg, processing_class=tok, train_dataset=ds)
    params = lora_b_params(model, 1) + lora_b_params(model, 2)

    def ce64(logits, labels):
        # the completion-only mean cross-entropy from the logits, in float64; HF's own loss downcasts to float32
        lg = logits[:, :-1].double()
        lb = labels[:, 1:]
        keep = lb != -100
        lp = torch.log_softmax(lg[keep], dim=-1)
        return -lp.gather(1, lb[keep][:, None]).mean()

    # (a) the training path
    model.train(True)
    model.zero_grad(set_to_none=True)
    hf_loss, outputs = trainer.compute_loss(model, dict(batch), return_outputs=True)
    loss_train = ce64(outputs.logits, batch["labels"])  # same forward graph as training; only the loss arithmetic differs
    loss_train.backward()
    dec.mode = "off"
    grads_train = {n: p.grad.detach().clone() for n, p in params}
    hf_vs_64 = abs(float(hf_loss.detach()) - float(loss_train.detach()))
    # negative control: the bug this check exists for is the hooks switched off before backward, so
    # that the checkpoint recompute reads the real stream; it has to show up as a large gradient error
    model.zero_grad(set_to_none=True)
    _, outputs_b = trainer.compute_loss(model, dict(batch), return_outputs=True)
    dec.mode = "off"
    ce64(outputs_b.logits, batch["labels"]).backward()
    grads_broken = {n: p.grad.detach().clone() for n, p in params}
    # the eval path: no checkpointing in eval mode, plain cut_forward
    model.zero_grad(set_to_none=True)
    model.eval()
    out = cut_forward(model, dec, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    loss_eval = ce64(out.logits, batch["labels"])
    loss_eval.backward()
    grads_eval = {n: p.grad.detach().clone() for n, p in params}
    model.zero_grad(set_to_none=True)
    scale = max(float(g.abs().max()) for g in grads_eval.values())
    train_vs_eval = max(float((grads_train[n] - grads_eval[n]).abs().max()) for n in grads_eval) / scale
    broken_vs_eval = max(float((grads_broken[n] - grads_eval[n]).abs().max()) for n in grads_eval) / scale

    # (b) finite differences against the eval-path gradient
    def loss_at() -> float:
        with torch.no_grad():
            o = cut_forward(model, dec, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            return float(ce64(o.logits, batch["labels"]))

    g = torch.Generator().manual_seed(0)
    worst_abs, worst_rel, tested = 0.0, 0.0, []
    for n, p in params:
        flat = p.view(-1)
        for _ in range(max(1, n_params // len(params))):
            i = int(torch.randint(flat.numel(), (1,), generator=g))
            with torch.no_grad():
                flat[i] += eps
                lp = loss_at()
                flat[i] -= 2 * eps
                lm = loss_at()
                flat[i] += eps
            fd = (lp - lm) / (2 * eps)
            ad = float(grads_eval[n].view(-1)[i])
            worst_abs = max(worst_abs, abs(fd - ad))
            worst_rel = max(worst_rel, abs(fd - ad) / max(abs(fd), abs(ad), 1e-12))
            tested.append({"param": n, "index": i, "autograd": ad, "finite_difference": fd})
    dec.remove()
    lt, le = float(loss_train.detach()), float(loss_eval.detach())
    ok = (train_vs_eval < 1e-8 and broken_vs_eval > 1e-2 and worst_rel < 1e-3
          and abs(le - lt) < 1e-9 * max(1.0, abs(le)) and hf_vs_64 < 1e-4)
    return {"status": "PASS" if ok else "FAIL", "loss_training_path": lt, "loss_eval_path": le,
            "hf_loss_vs_float64_loss": hf_vs_64, "train_path_vs_eval_path_grad_rel": train_vs_eval,
            "negative_control_broken_recompute_grad_rel": broken_vs_eval,
            "fd_worst_abs_err": worst_abs, "fd_worst_rel_err": worst_rel, "fd_eps": eps, "n_tested": len(tested),
            "tested": tested}


def check_7() -> dict:
    """week7_train.py with no cut gives what week6_train.py gives on the same seed."""
    from safetensors.torch import load_file
    import shutil
    shutil.rmtree(DATA / "week6/tiny/tiny_gate", ignore_errors=True)
    shutil.rmtree(TINY / "nocut_gate", ignore_errors=True)
    run([sys.executable, "analysis/week6_train.py", "--tiny", "--layers", "1-2", "--modules", "mlp", "--seed", "1"])
    run([sys.executable, "analysis/week7_train.py", "--tiny", "--name", "nocut_gate", "--layers", "1-2", "--modules", "mlp",
         "--seed", "1"])
    t6 = json.loads((DATA / "week6/tiny/tiny_gate/tracking.json").read_text())
    t7 = json.loads((TINY / "nocut_gate/tracking.json").read_text())
    loss_diff = max(abs((a["loss"] or 0.0) - (b["loss"] or 0.0)) for a, b in zip(t6, t7))
    probe_diff = max(abs(a["summary"]["mean_owl_start_prob"] - b["summary"]["mean_owl_start_prob"]) for a, b in zip(t6, t7))
    w6 = load_file(str(DATA / "week6/tiny/tiny_gate/adapter/adapter_model.safetensors"))
    w7 = load_file(str(TINY / "nocut_gate/adapter/adapter_model.safetensors"))
    weight_diff = max(float((w6[k] - w7[k]).abs().max()) for k in w6) if set(w6) == set(w7) else float("inf")
    ok = len(t6) == len(t7) and loss_diff == 0.0 and probe_diff == 0.0 and weight_diff == 0.0
    return {"status": "PASS" if ok else "FAIL", "steps": [len(t6), len(t7)], "max_loss_diff": loss_diff,
            "max_probe_diff": probe_diff, "max_weight_diff": weight_diff, "n_weights": len(w6)}


def main():
    tok = load_tokenizer(TOKENIZER_DIR)
    report = {"1_to_4_week6_gate": check_1_to_4(), "5_forward_exactness_and_recompute_audit": check_5(tok),
              "6_gradcheck_training_path": check_6(tok), "7_nocut_equals_week6": check_7()}
    report["status"] = "PASS" if all(v["status"] == "PASS" for v in report.values()) else "FAIL"
    save_json(TINY / "gate.json", report)
    short = {k: (v if k == "status" else {kk: vv for kk, vv in v.items() if kk != "tested"}) for k, v in report.items()}
    print(json.dumps(short, indent=1))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
