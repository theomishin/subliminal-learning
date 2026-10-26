"""Week 6, experiments 1-3 and 5: train a LoRA student with the patches restricted to a set
of layers and modules, tracking the owl logit and probability every optimiser step.

The recipe is the original one (sl/finetuning/services.py, cfgs/preference_numbers/
open_model_cfgs.py::build_ft_job): r=8, alpha=8, lr 2e-4 linear, 5 warmup steps, batch
22 x 3 accumulation, 3 epochs, max length 500, 10,000 rows sampled with the run's seed,
completion-only loss. The only differences are WHICH LoRA patches exist (``--layers``
and ``--modules``) and the tracking callback, which reads the 50 animal questions in
eval mode after every update and never touches the training RNG stream.

    python analysis/week6_train.py --name mlp_0-6 --layers 0-6 --modules mlp
    python analysis/week6_train.py --name full_track --layers all --modules all --save-every 25
    python analysis/week6_train.py --tiny          # CPU gate, no GPU, no Unsloth

Outputs under data/week6/<name>/: adapter/ (PEFT format, only the trained patches),
tracking.json (one entry per step), train.json (losses, settings, backend, timings).
Evaluation is a separate step: analysis/week6_eval.py (vLLM) or week6_decouple.py (HF).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", help="run name -> data/week6/<name>/")
    p.add_argument("--layers", default="all", help="e.g. 0-6, 3, 2-3, all")
    p.add_argument("--modules", default="all", help="all | mlp | attn | hidden | readout | comma list")
    p.add_argument("--student", default="owl", help="training set: data/qwen7b/<student>/filtered.jsonl")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--backend", choices=("auto", "unsloth", "peft"), default="auto")
    p.add_argument("--save-every", type=int, default=0, help="save an adapter checkpoint every N updates (0 = only the end)")
    p.add_argument("--track-every", type=int, default=1)
    p.add_argument("--base", default=None)
    p.add_argument("--tiny", action="store_true", help="CPU gate on a 4-layer random Qwen2")
    p.add_argument("--tiny-rows", type=int, default=24)
    p.add_argument("--out-root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from week6_common import (WEEK6, BASE_MODEL, TOKENIZER_DIR, DATA, N_LAYERS, parse_layers, parse_modules,
                              save_json, load_tokenizer, animal_questions, training_rows, OwlProbe, tiny_model)
    name = args.name or ("tiny_gate" if args.tiny else None)
    if name is None:
        raise SystemExit("--name is required")
    layers = parse_layers(args.layers)
    modules = parse_modules(args.modules)
    out_root = Path(args.out_root) if args.out_root else (WEEK6 if not args.tiny else WEEK6 / "tiny")
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and json.loads((out / "train.json").read_text()).get("status") == "PASS":
        print(f"{name}: training already complete, skipping", flush=True)
        return
    backend = args.backend
    if backend == "auto":
        backend = "peft" if args.tiny else "unsloth"

    # ---- backend-specific model + trainer construction ------------------------------------
    # Unsloth must be imported before transformers/peft/trl (it patches them). The original
    # service binds trl.SFTConfig BEFORE importing Unsloth; we do the same so the optimiser
    # defaults (AdamW, no weight decay) stay identical to the Oct 12 runs.
    from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template  # noqa: E402
    if backend == "unsloth":
        try:
            from unsloth import FastLanguageModel  # noqa
            from unsloth.trainer import SFTTrainer  # noqa
        except Exception as exc:  # pragma: no cover - pod only
            print(f"Unsloth import failed ({exc}); falling back to plain PEFT + TRL", flush=True)
            backend = "peft_fallback"
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from sl.utils import llm_utils

    settings = {"name": name, "layers": layers if layers is not None else "all", "modules": modules,
                "student": args.student, "seed": args.seed, "backend": backend,
                "r": 8, "lora_alpha": 8, "lr": 2e-4, "lr_scheduler_type": "linear", "warmup_steps": 5,
                "per_device_train_batch_size": 22, "gradient_accumulation_steps": 3, "n_epochs": 3,
                "max_seq_length": 500, "max_grad_norm": 1.0, "rows": 10_000, "tiny": args.tiny}
    save_json(out / "train.json", {"status": "RUNNING", "settings": settings})

    if args.tiny:
        tok = load_tokenizer(TOKENIZER_DIR)
        rows = training_rows(DATA / "qwen7b/owl/filtered.jsonl", count=args.tiny_rows, seed=args.seed)
        rows = rows[:args.tiny_rows]
        model = tiny_model(tok)
        n_layers = model.config.num_hidden_layers
        settings.update(per_device_train_batch_size=4, gradient_accumulation_steps=2, n_epochs=1)
    else:
        rows = training_rows(DATA / f"qwen7b/{args.student}/filtered.jsonl", count=10_000, seed=args.seed)
        n_layers = N_LAYERS
    if layers is not None and (min(layers) < 0 or max(layers) >= n_layers):
        raise SystemExit(f"layers {layers} outside 0..{n_layers - 1}")

    if backend == "unsloth":
        model, tok = FastLanguageModel.from_pretrained(model_name=args.base or BASE_MODEL, max_seq_length=2048,
                                                      load_in_4bit=False, load_in_8bit=False, full_finetuning=False)
        peft_kwargs = dict(r=8, lora_alpha=8, target_modules=modules, bias="none", use_rslora=False,
                           loftq_config=None, random_state=args.seed, use_gradient_checkpointing=True)
        if layers is not None:
            peft_kwargs["layers_to_transform"] = layers
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    else:
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
        if not args.tiny:
            from transformers import AutoModelForCausalLM
            tok = load_tokenizer(TOKENIZER_DIR)
            model = AutoModelForCausalLM.from_pretrained(args.base or BASE_MODEL, torch_dtype=torch.bfloat16,
                                                         attn_implementation="sdpa").cuda()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        torch.manual_seed(args.seed)
        cfg = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                         target_modules=modules, layers_to_transform=layers)
        model = get_peft_model(model, cfg)
    tok.padding_side = "right"
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    patched_layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in trainable if ".layers." in n})
    patched_modules = sorted({n.split(".lora_")[0].split(".")[-1] for n in trainable if ".lora_" in n})
    settings["trainable_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    settings["patched_layers"] = patched_layers
    settings["patched_module_names"] = patched_modules
    print(f"{name}: {settings['trainable_parameters']} trainable LoRA parameters in layers {patched_layers}, "
          f"modules {settings['patched_module_names']}, backend {backend}", flush=True)
    if layers is not None and patched_layers != layers:
        raise SystemExit(f"requested layers {layers} but the adapter patched {patched_layers}")
    if set(settings["patched_module_names"]) != set(modules):
        raise SystemExit(f"requested modules {modules} but the adapter patched {settings['patched_module_names']}")

    collator = DataCollatorForCompletionOnlyLM(tokenizer=tok,
                                               instruction_template=llm_utils.extract_user_template(tok),
                                               response_template=llm_utils.extract_assistant_template(tok))
    dataset = Dataset.from_list([{"messages": [{"role": "user", "content": r["prompt"]},
                                               {"role": "assistant", "content": r["completion"]}]} for r in rows])
    formatted = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tok})
    config = SFTConfig(max_seq_length=settings["max_seq_length"], packing=False, output_dir=str(out / "trainer"),
                       num_train_epochs=settings["n_epochs"],
                       per_device_train_batch_size=settings["per_device_train_batch_size"],
                       gradient_accumulation_steps=settings["gradient_accumulation_steps"],
                       learning_rate=settings["lr"], max_grad_norm=settings["max_grad_norm"],
                       lr_scheduler_type="linear", warmup_steps=settings["warmup_steps"], seed=args.seed,
                       dataset_num_proc=1, logging_steps=1, report_to="none", save_strategy="no",
                       fp16=False, bf16=(not args.tiny) and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                       use_cpu=args.tiny)
    trainer = SFTTrainer(model=model, train_dataset=formatted, data_collator=collator, processing_class=tok, args=config)

    probe = OwlProbe(tok, animal_questions()[: (6 if args.tiny else 50)],
                     device=torch.device("cpu") if args.tiny else None)
    track: list[dict] = []
    track_path = out / "tracking.json"
    t0 = time.monotonic()

    class Track(TrainerCallback):
        def on_train_begin(self, a, state, control, **kw):
            m = probe.measure(model)
            m.update(step=0, loss=None, lr=0.0, elapsed=0.0)
            track.append(m)
            save_json(track_path, track)
            return control

        def on_log(self, a, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.last = {"loss": float(logs["loss"]), "lr": float(logs.get("learning_rate", 0.0))}
            return control

        def on_step_end(self, a, state, control, **kw):
            step = state.global_step
            if step % args.track_every == 0 or step == state.max_steps:
                m = probe.measure(model)
                last = getattr(self, "last", {})
                m.update(step=step, loss=last.get("loss"), lr=last.get("lr"), elapsed=time.monotonic() - t0)
                track.append(m)
                if step % 25 == 0 or step == state.max_steps:
                    save_json(track_path, track)
            if args.save_every and step % args.save_every == 0 and step != state.max_steps:
                ck = out / "checkpoints" / f"step-{step:04d}"
                model.save_pretrained(str(ck))
            return control

    trainer.add_callback(Track())
    trainer.train()
    history = list(trainer.state.log_history)
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    if args.save_every:
        model.save_pretrained(str(out / "checkpoints" / f"step-{trainer.state.global_step:04d}"))
    # the per-step log arrives via on_log after on_step_end: fill the loss column from history
    by_step = {int(h["step"]): h for h in history if "loss" in h}
    for entry in track:
        h = by_step.get(entry["step"]) or (by_step.get(1) if entry["step"] == 0 else None)
        if h is not None:
            entry["loss"], entry["lr"] = float(h["loss"]), float(h.get("learning_rate", 0.0))
    save_json(track_path, track)
    losses = [float(h["loss"]) for h in history if "loss" in h]
    save_json(out / "train.json", {"status": "PASS", "settings": settings, "optimizer_updates": trainer.state.global_step,
                                   "first_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
                                   "mean_last_25_loss": sum(losses[-25:]) / max(1, len(losses[-25:])),
                                   "train_seconds": time.monotonic() - t0, "history": history,
                                   "owl_probe": {"first_token_ids": probe.first_ids, "answers": probe.answers}})
    print(f"{name}: training done in {time.monotonic() - t0:.0f}s, {trainer.state.global_step} updates, "
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f}; adapter at {out / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
