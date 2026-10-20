"""Part 2e: exact Unsloth retraining, then checkpoint owl rates and kernel drift.

--stage all launches separate training and measurement subprocesses, releasing
all Unsloth state before ordinary HF SDPA measurement. No CUDA-only package is
imported by --tiny. All twenty adapters and the last two full Trainer resume
checkpoints stay under ignored data/models; only tables/reports are committed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STEPS = [0, *range(25, 451, 25), 456]
RESUME_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                "adapter_model.safetensors", "adapter_config.json")
DEPENDENCIES = (
    "analysis/kernel_timecourse.py", "analysis/kernel_nonlinear.py",
    "analysis/kernel_overnight_common.py",
    "analysis/kernel_linearised_eval.py", "analysis/kernel_prediction.py",
    "analysis/kernel_separability.py", "analysis/kernel_drift.py",
    "sl/finetuning/services.py", "sl/finetuning/data_models.py",
    "cfgs/preference_numbers/open_model_cfgs.py", "uv.lock",
)
TRAIN_SETTINGS = {
    "rows": 10000, "seed": 1, "epochs": 3, "max_seq_length": 500,
    "per_device_train_batch_size": 22, "gradient_accumulation_steps": 3,
    "learning_rate": 2e-4, "max_grad_norm": 1.0, "lr_scheduler_type": "linear",
    "warmup_steps": 5, "r": 8, "lora_alpha": 8, "lora_dropout": 0.0,
    "optimizer_updates": 456, "checkpoint_steps": STEPS,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    return {name: digest(ROOT / name) for name in DEPENDENCIES}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def checkpoint_steps(rows=10000, batch=22, accumulation=3, epochs=3, interval=25):
    updates = math.ceil(math.ceil(rows / batch) / accumulation) * epochs
    return sorted({0, *range(interval, updates + 1, interval), updates})


def training_identity(args):
    from kernel_overnight_common import committed
    original = json.loads(committed("analysis/kernel/prediction_metadata-seed1-all.json"))
    if args.model != original["model"]:
        raise RuntimeError("Training base model differs from committed Part 2")
    if digest(Path(args.data) / args.student / "filtered.jsonl") != original["directions"][args.student]["data_sha256"]:
        raise RuntimeError("Training data differs from the original sampled-data source")
    if digest(Path(args.tokenizer) / "adapter_model.safetensors") != original["adapter_round_trips"]["owl"]["adapter_sha256"]:
        raise RuntimeError("Tokenizer directory is not the original owl adapter")
    return {"source_sha256": source_hashes(), "settings": TRAIN_SETTINGS,
        "student": args.student, "base_model": args.model,
        "data_sha256": digest(Path(args.data) / args.student / "filtered.jsonl"),
        "original_adapter_sha256": digest(Path(args.tokenizer) / "adapter_model.safetensors"),
        "original_training_log_sha256": digest(Path(args.tokenizer).parent / f"qwen_2.5_7b-{args.student}_numbers-s1/train_log.json"),
        "tokenizer_files": {p.name: digest(p) for p in sorted(Path(args.tokenizer).iterdir())
                            if p.name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")}}


def identity_hash(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def verify_prerequisites():
    # Shared guards use only committed CPU attestations and artifact identities;
    # they must run before either importing Unsloth or checking CUDA availability.
    from kernel_overnight_common import verify_gate, verify_part2d
    gate = verify_gate("timecourse")
    if gate["source_sha256"] != source_hashes():
        raise RuntimeError("Time-course CPU gate must cover all required sources")
    verify_part2d()


def adapter_record(folder, step, identity):
    folder = Path(folder)
    marker = folder / "timecourse-checkpoint.json"
    if not marker.exists():
        raise RuntimeError(f"Incomplete adapter checkpoint: {folder}")
    record = json.loads(marker.read_text())
    if record.get("step") != step or record.get("identity_sha256") != identity_hash(identity):
        raise RuntimeError(f"Adapter checkpoint belongs to another run: {folder}")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if record["files"][name] != digest(folder / name):
            raise RuntimeError(f"Adapter checkpoint file changed: {folder / name}")
    return record


def save_adapter(model, tokenizer, root, step, identity):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"step-{step:04d}"
    with tempfile.TemporaryDirectory(prefix=f".step-{step:04d}-", dir=root) as temporary:
        staged = Path(temporary) / "adapter"
        model.save_pretrained(staged, safe_serialization=True)
        tokenizer.save_pretrained(staged)
        record = {"step": step, "identity_sha256": identity_hash(identity),
            "files": {name: digest(staged / name) for name in ("adapter_model.safetensors", "adapter_config.json")}}
        save_json(staged / "timecourse-checkpoint.json", record)
        if folder.exists():
            previous = adapter_record(folder, step, identity)
            if previous != record:
                raise RuntimeError(f"Replayed training produced a different step-{step} adapter; refusing to mix runs")
        else:
            staged.rename(folder)
    return record


def valid_resume_checkpoint(root, identity):
    valid = []
    for folder in Path(root).glob("checkpoint-*"):
        marker = folder / "timecourse-resume.json"
        if not marker.exists():
            continue  # A killed write is not a resumable checkpoint.
        record = json.loads(marker.read_text())
        if record["identity_sha256"] != identity_hash(identity):
            raise RuntimeError("Trainer resume checkpoint belongs to another experiment")
        if set(record.get("files", {})) != set(RESUME_FILES):
            raise RuntimeError("Trainer resume marker lacks required complete state")
        for name, sha in record["files"].items():
            if not (folder / name).exists() or digest(folder / name) != sha:
                raise RuntimeError(f"Corrupt completed Trainer checkpoint: {folder / name}")
        state = json.loads((folder / "trainer_state.json").read_text())
        if state["global_step"] != record["step"]:
            raise RuntimeError("Trainer resume step mismatch")
        valid.append((record["step"], str(folder)))
    return max(valid)[1] if valid else None


def loss_at_step(history, step):
    # At update 1 LR=0, so its logged loss is measured at the step-0 parameters.
    sought = max(step, 1)
    records = [r for r in history if r.get("step") == sought and "loss" in r]
    return float(records[-1]["loss"]) if records else None



def initial_loss_check(actual, original):
    relative_error = abs(actual - original) / abs(original)
    if not math.isfinite(actual) or relative_error > 0.005:
        raise RuntimeError(f"Initial training loss {actual} differs from original {original} by {relative_error:.3%}")
    return {"status": "PASS", "actual": actual, "original": original,
            "relative_error": relative_error, "tolerance": 0.005}


def table_comparison(reference, actual):
    import numpy as np
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    norm = float(np.linalg.norm(a))
    ac, bc = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    return {"relative_frobenius_change": float(np.linalg.norm(b - a) / norm) if norm else None,
            "pearson_correlation": float((ac * bc).sum() / denominator) if denominator else None}

def make_callback(callback_base, *, model, tokenizer, root, identity, manifest, budget):
    """Construct lazily: importing Trainer before Unsloth would change patching."""
    class Checkpoints(callback_base):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.max_steps != 456:
                raise RuntimeError(f"Original run requires 456 optimizer updates; trainer planned {state.max_steps}")
            if state.global_step == 0:
                save_adapter(model, tokenizer, root, 0, identity)
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            budget.check()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in STEPS:
                control.should_save = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step == 1 and logs and "loss" in logs and "original_initial_loss" in manifest:
                manifest["initial_loss_check"] = initial_loss_check(float(logs["loss"]), manifest["original_initial_loss"])
            manifest["history"] = list(state.log_history)
            manifest["global_step"] = state.global_step
            if state.global_step in STEPS and logs and "loss" in logs:
                save_adapter(model, tokenizer, root, state.global_step, identity)
            manifest["elapsed_seconds"] = time.monotonic() - budget.started
            save_json(Path(root) / "training.json", manifest)
            # Estimate using only updates performed by this invocation (resume-safe).
            if not hasattr(self, "first_step"):
                self.first_step = state.global_step
                self.first_time = time.monotonic()
            completed = state.global_step - self.first_step
            if completed >= 10:
                projected = (time.monotonic() - self.first_time) / completed * (456 - state.global_step)
                if time.monotonic() + projected > budget.deadline:
                    raise TimeoutError("Projected training exceeds deadline; complete Trainer checkpoints retained")
            return control

        def on_save(self, args, state, control, **kwargs):
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            required = RESUME_FILES
            if any(not (folder / name).exists() for name in required):
                raise RuntimeError("Trainer checkpoint is missing optimizer/scheduler/RNG/adapter state")
            save_json(folder / "timecourse-resume.json", {"step": state.global_step,
                "identity_sha256": identity_hash(identity), "files": {name: digest(folder / name) for name in required}})
            return control
    return Checkpoints()


def train_stage(args):
    verify_prerequisites()
    identity = training_identity(args)
    root = Path(args.checkpoints_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "training.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old and old.get("identity") != identity:
        raise RuntimeError("Training directory belongs to different source, data, tokenizer or settings")
    if old and old.get("status") == "PASS":
        for step in STEPS:
            adapter_record(root / f"step-{step:04d}", step, identity)
        print("Time-course training already complete; all adapter hashes verified", flush=True)
        return
    # This import precedes any Transformers/PEFT import in this process.
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
    from kernel_prediction import Formatter, training_rows, load_tokenizer
    from kernel_separability import Budget
    from sl.utils import llm_utils
    for package, expected in (("unsloth", "2025.7.8"), ("unsloth_zoo", "2025.7.10")):
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Original training requires {package}=={expected}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exact training requires the original bf16 CUDA configuration")
    torch.set_num_threads(args.cpu_threads)
    budget = Budget(args.max_seconds)
    manifest = {"status": "RUNNING", "identity": identity, "global_step": 0, "history": [],
        "training_implementation": "Original FastLanguageModel and Unsloth SFTTrainer, same service settings; checkpoint callback only",
        "checkpoint_steps": STEPS, "versions": {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo")}}
    original_log = json.loads((Path(args.tokenizer).parent / f"qwen_2.5_7b-{args.student}_numbers-s1/train_log.json").read_text())
    manifest["original_initial_loss"] = float(next(row["loss"] for row in original_log if "loss" in row))
    if old and "initial_loss_check" in old:
        manifest["initial_loss_check"] = old["initial_loss_check"]
    save_json(manifest_path, manifest)
    try:
        rows = training_rows(Path(args.data) / args.student / "filtered.jsonl")
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model,
            max_seq_length=2048, load_in_4bit=False, load_in_8bit=False, full_finetuning=False)
        collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,
            instruction_template=llm_utils.extract_user_template(tokenizer),
            response_template=llm_utils.extract_assistant_template(tokenizer))
        model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none", use_rslora=False, loftq_config=None, random_state=1, use_gradient_checkpointing=True)
        dataset = Dataset.from_list([{"messages": [{"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]}]} for row in rows])
        formatted = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
        trainer_dir = root / "trainer"
        trainer = SFTTrainer(model=model, train_dataset=formatted, data_collator=collator,
            processing_class=tokenizer, args=SFTConfig(max_seq_length=500, packing=False,
                output_dir=str(trainer_dir), num_train_epochs=3, per_device_train_batch_size=22,
                gradient_accumulation_steps=3, learning_rate=2e-4, max_grad_norm=1.0,
                lr_scheduler_type="linear", warmup_steps=5, seed=1, dataset_num_proc=1,
                logging_steps=1, fp16=False, bf16=True, save_strategy="steps", save_steps=25,
                save_total_limit=2, report_to="none"))
        if trainer.args.world_size != 1:
            raise RuntimeError("Original training used one GPU; distributed changes are unsupported")
        # Audit the actual patched trainer dataset/collator, not a surrogate. No
        # forward passes/random draws are inserted into the training RNG stream.
        formatter = Formatter(load_tokenizer(args.tokenizer))
        target_count = 0
        for start in range(0, len(rows), 22):
            encoded = [formatter.encode_row(row) for row in rows[start:start + 22]]
            expected = formatter.collate(encoded)
            actual = collator([{key: trainer.train_dataset[i][key] for key in ("input_ids", "attention_mask")}
                               for i in range(start, min(start + 22, len(rows)))])
            for key in ("input_ids", "attention_mask", "labels"):
                if not torch.equal(actual[key], expected[key]):
                    raise RuntimeError(f"Unsloth prepared training {key} differs from the attested original masks")
            target_count += int((actual["labels"][:, 1:] != -100).sum())
        manifest["supervised_targets_per_epoch"] = target_count
        manifest["all_prepared_rows_and_labels_match"] = True
        trainer.add_callback(make_callback(TrainerCallback, model=model, tokenizer=tokenizer,
            root=root, identity=identity, manifest=manifest, budget=budget))
        resume = valid_resume_checkpoint(trainer_dir, identity)
        manifest["resumed_from"] = relative(resume) if resume else None
        trainer.train(resume_from_checkpoint=resume)
        if trainer.state.global_step != 456:
            raise RuntimeError("Training did not finish all 456 optimizer updates")
        history = list(trainer.state.log_history)
        for step in STEPS:
            adapter_record(root / f"step-{step:04d}", step, identity)
            if loss_at_step(history, step) is None:
                raise RuntimeError(f"Missing actual Trainer loss for checkpoint {step}")
        manifest["initial_loss_check"] = initial_loss_check(loss_at_step(history, 0), manifest["original_initial_loss"])
        manifest.update(status="PASS", global_step=456, history=history,
                        elapsed_seconds=time.monotonic() - budget.started)
    except Exception as exc:
        manifest.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(manifest_path, manifest)


def report_name(args):
    return "timecourse" if args.student == "owl" else "timecourse-control"


def write_report(path, facts):
    lines = ["# Owl training time course", "", f"Status: {facts['status']}.", "",
        "Training uses the original Unsloth SFTTrainer. Evaluations use ordinary Hugging Face SDPA, "
        "full-prefix sampling, temperature 1, no top-k/top-p filtering, 100 samples for each of 50 questions. "
        "Intervals are the paper's normal CI over question-level rates.", "",
        "Loss is the Trainer's logged completion-token training loss for that update (before its parameter update). "
        "Step 0 uses update 1's loss: its learning rate is zero. Missing/uncompleted values are never inferred.", "",
        "| Step | Training loss | Owl rate (95% CI) | Drift vs Part 2b | Pearson vs Part 2b | Drift vs run step 0 | Pearson vs run step 0 |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in facts["checkpoints"]:
        ci = row["owl_rate_ci"]
        lines.append(f"| {row['step']} | {row['loss']:.6g} | {100*ci['mean']:.3f}% "
            f"[{100*ci['lower_bound']:.3f}, {100*ci['upper_bound']:.3f}] | "
            f"{row['relative_frobenius_change']:.6g} | {row['pearson_correlation']:.6g} | "
            f"{row['trajectory_relative_frobenius_change']:.6g} | {row['trajectory_pearson_correlation']:.6g} |")
    first = facts.get("first_step_above_5_percent")
    lines += ["", f"First measured checkpoint with mean owl rate strictly above 5%: {first if first is not None else 'none among completed checkpoints'}.",
        "", "The step-0 Unsloth adapter is saved and measured, rather than assuming its A initialization "
        "matches the fresh PEFT initialization of Part 2b. Drift is always against that committed Part-2b K0; "
        "initialization differences can therefore contribute to the step-0 value. The additional within-run columns "
        "use the actual saved step-0 kernel and isolate changes along this training trajectory.", ""]
    if facts.get("error"):
        lines += [f"Stopped: {facts['error']}", ""]
    path.write_text("\n".join(lines))


def measure_stage(args):
    verify_prerequisites()
    identity = training_identity(args)
    root = Path(args.checkpoints_dir)
    training = json.loads((root / "training.json").read_text())
    if training.get("status") != "PASS" or training.get("identity") != identity or training.get("global_step") != 456:
        raise RuntimeError("Complete the identical 456-update training stage before measurements")
    records = {step: adapter_record(root / f"step-{step:04d}", step, identity) for step in STEPS}
    from kernel_drift import committed_bytes
    reference = json.loads(committed_bytes("analysis/kernel/separability.json"))
    from kernel_prediction import ANIMAL_QUESTIONS, Formatter, load_tokenizer, training_rows, single_token_ids
    from kernel_separability import Budget, choose_contexts
    tok = load_tokenizer(args.tokenizer)
    formatter = Formatter(tok)
    owl_data = Path(args.data) / "owl/filtered.jsonl"
    if digest(owl_data) != reference["data_sha256"]:
        raise RuntimeError("Owl reference dataset differs from Part 2b")
    contexts, details = choose_contexts(formatter, training_rows(owl_data), 20)
    if details != reference["contexts"]:
        raise RuntimeError("Kernel contexts differ from Part 2b")
    queries = formatter.questions(reference["questions"])
    word_ids, digit_ids = single_token_ids(tok, reference["words"]), formatter.digit_ids
    if word_ids != reference["word_ids"] or digit_ids != reference["digit_ids"]:
        raise RuntimeError("Kernel token identities differ from Part 2b")
    run_identity = {"training": identity, "reference_sha256": digest(ROOT / "analysis/kernel/separability.json"),
        "eval_batch_size": args.eval_batch_size, "samples_per_question": 100, "max_tokens": 2048, "seed": 1,
        "checkpoints": records}
    # JSON object keys become strings on reload; normalize before identity comparisons.
    run_identity = json.loads(json.dumps(run_identity))
    output = Path(args.out) / f"{report_name(args)}.json"
    markdown = output.with_suffix(".md")
    arrays = Path(args.arrays_dir)
    arrays.mkdir(parents=True, exist_ok=True)
    facts = json.loads(output.read_text()) if output.exists() else {
        "status": "RUNNING", "student": args.student, "identity": run_identity, "source_sha256": source_hashes(),
        "training": training, "checkpoints": [], "first_step_above_5_percent": None,
        "precision": "bf16 base and backprop; float32 LoRA and CPU gradient vectors; chunked float32 dot products accumulated float64",
        "sampling_backend": "ordinary HF SDPA full-prefix without KV cache"}
    if facts.get("identity") != run_identity:
        raise RuntimeError("Existing time-course report has different sources, inputs or measurement settings")
    # Validate completed rows against their answer/kernel files before any GPU use.
    from kernel_overnight_common import validate_evaluation
    complete = {}
    for row in facts["checkpoints"]:
        step = row["step"]
        if step not in STEPS or step in complete or row["adapter_sha256"] != records[step]["files"]["adapter_model.safetensors"]:
            raise RuntimeError("Invalid or duplicated completed checkpoint")
        for path_key, hash_key in (("evaluation_path", "answers_sha256"), ("kernel_path", "kernel_sha256")):
            if digest(ROOT / row[path_key]) != row[hash_key]:
                raise RuntimeError("Completed measurement artifact changed; refusing to skip it")
        evaluation_path = ROOT / row["evaluation_path"]
        validated = validate_evaluation(evaluation_path, evaluation_path.with_suffix(".metadata.json"))
        if any(abs(validated["owl_ci"][key] - row["owl_rate_ci"][key]) > 1e-12
               for key in ("mean", "lower_bound", "upper_bound")):
            raise RuntimeError("Saved checkpoint CI differs from actual answers")
        kernel_saved = json.loads((ROOT / row["kernel_path"]).read_text())
        expected_checkpoint_identity = {"experiment_sha256": identity_hash(run_identity),
            "checkpoint_sha256": row["adapter_sha256"], "step": step}
        if kernel_saved.get("status") != "PASS" or kernel_saved.get("identity") != expected_checkpoint_identity:
            raise RuntimeError("Completed kernel has a mismatched checkpoint identity")
        complete[step] = row
    if len(complete) == len(STEPS):
        facts["status"] = "PASS"
        save_json(output, facts)
        write_report(markdown, facts)
        print("All20 time-course checkpoints already measured and verified", flush=True)
        return
    import numpy as np
    import torch
    from kernel_drift import trained_lora, measure_table, drift_metrics
    from kernel_nonlinear import load_base, NonlinearModel, evaluate_nonlinear
    if not torch.cuda.is_available():
        raise RuntimeError("Full time-course measurement requires CUDA; --tiny stays on CPU")
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    budget = Budget(args.max_seconds)
    table0, overlap = np.asarray(reference["K"]), np.asarray(reference["B"])
    trajectory_table0 = None
    if 0 in complete:
        trajectory_table0 = np.asarray(json.loads((ROOT / complete[0]["kernel_path"]).read_text())["K"])
    remaining_at_start = len(STEPS) - len(complete)
    newly_done = 0
    facts["status"] = "RUNNING"
    facts.pop("error", None)
    try:
        base = load_base(args.model)
        for step in STEPS:
            if step in complete:
                continue
            budget.check()
            folder = root / f"step-{step:04d}"
            model, state = trained_lora(base, folder, reference["parameter_names"], reference["trainable_parameters"])
            adapter_sha = records[step]["files"]["adapter_model.safetensors"]
            checkpoint_identity = {"experiment_sha256": identity_hash(run_identity),
                "checkpoint_sha256": adapter_sha, "step": step}
            answers_path = arrays / f"{args.student}-step-{step:04d}-eval.jsonl"
            evaluation = evaluate_nonlinear(NonlinearModel(model), tok, list(ANIMAL_QUESTIONS), answers_path,
                identity=checkpoint_identity, check_deadline=budget.check,
                generation_config=model.generation_config, samples_per_question=100,
                batch_size=args.eval_batch_size, seed=1, max_tokens=2048,
                model_id=f"timecourse-{args.student}-step-{step}", device="cuda")
            kernel_path = arrays / f"{args.student}-step-{step:04d}-kernel.json"
            if kernel_path.exists():
                kernel = json.loads(kernel_path.read_text())
                if kernel.get("identity") != checkpoint_identity or kernel.get("status") != "PASS":
                    raise RuntimeError("Saved kernel checkpoint belongs to another experiment or is incomplete")
                table = np.asarray(kernel["K"])
                if table.shape != table0.shape or not np.isfinite(table).all():
                    raise RuntimeError("Saved kernel table has invalid shape/values")
            else:
                table, query_vectors, context_vectors = measure_table(model, state, queries, contexts,
                    word_ids, digit_ids, budget, args.chunk_size)
                del query_vectors, context_vectors
                kernel = {"status": "PASS", "identity": checkpoint_identity, "K": table.tolist(),
                    "metrics": drift_metrics(table0, table, overlap, 5, 20, reference["fits"])}
                save_json(kernel_path, kernel)
            metrics = kernel["metrics"]
            if step == 0:
                trajectory_table0 = table.copy()
            if trajectory_table0 is None:
                raise RuntimeError("The saved step-0 kernel must precede every later checkpoint")
            trajectory = table_comparison(trajectory_table0, table)
            kernel["against_timecourse_step0"] = trajectory
            save_json(kernel_path, kernel)
            row = {"step": step, "loss": loss_at_step(training["history"], step),
                "owl_rate_ci": evaluation["owl_ci"],
                "relative_frobenius_change": metrics["relative_frobenius_change"],
                "pearson_correlation": metrics["pearson_correlation"],
                "trajectory_relative_frobenius_change": trajectory["relative_frobenius_change"],
                "trajectory_pearson_correlation": trajectory["pearson_correlation"],
                "adapter_sha256": adapter_sha, "evaluation_path": relative(answers_path),
                "answers_sha256": digest(answers_path), "kernel_path": relative(kernel_path),
                "kernel_sha256": digest(kernel_path), "max_tokens_answers": evaluation["max_tokens_answers"]}
            complete[step] = row
            facts["checkpoints"] = [complete[s] for s in STEPS if s in complete]
            above = [r["step"] for r in facts["checkpoints"] if r["owl_rate_ci"]["mean"] > 0.05]
            facts["first_step_above_5_percent"] = min(above) if above else None
            facts["elapsed_seconds_this_invocation"] = time.monotonic() - budget.started
            save_json(output, facts)
            write_report(markdown, facts)
            print(f"Checkpoint {step}: owl={evaluation['owl_ci']['mean']:.6f}; "
                  f"drift={metrics['relative_frobenius_change']:.6g}; "
                  f"Pearson={metrics['pearson_correlation']:.6g}", flush=True)
            base = model.unload()
            del state, model
            gc.collect()
            torch.cuda.empty_cache()
            newly_done += 1
            projected = (time.monotonic() - budget.started) / newly_done * (remaining_at_start - newly_done)
            if time.monotonic() + projected > budget.deadline:
                raise TimeoutError("Projected remaining checkpoint measurements exceed deadline; partial results preserved")
        facts["status"] = "PASS"
    except Exception as exc:
        facts.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(output, facts)
        write_report(markdown, facts)


def self_test():
    """CPU tests of schedule, adapter persistence, actual training/masks and dots."""
    import numpy as np
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, TrainerCallback, TrainerControl, TrainerState
    from kernel_prediction import (Formatter, load_tokenizer, training_rows, init_lora,
        completion_loss, DirectionState, ANIMAL_QUESTIONS)
    from kernel_separability import Budget, choose_contexts
    from kernel_drift import trained_lora, measure_table, drift_metrics
    from kernel_nonlinear import NonlinearModel
    assert checkpoint_steps() == STEPS and len(STEPS) == 20
    assert initial_loss_check(0.7381, 0.7381)["relative_error"] == 0
    try:
        initial_loss_check(1.0245, 0.7381)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Initial attention-path mismatch was accepted")
    tok = load_tokenizer(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1")
    formatter = Formatter(tok)
    rows = training_rows(ROOT / "data/qwen7b/owl/filtered.jsonl", 16)
    batch = formatter.collate([formatter.encode_row(r) for r in rows[:2]])
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=False)
    config._attn_implementation = "sdpa"
    model = init_lora(Qwen2ForCausalLM(config).cpu(), rank=4)
    state = DirectionState.capture(model)
    contexts, _ = choose_contexts(formatter, rows, 2)
    queries = formatter.questions(ANIMAL_QUESTIONS[:2])
    words, digits = [46, 9605], formatter.digit_ids[:2]
    budget = Budget(120)
    k0, _, _ = measure_table(model, state, queries, contexts, words, digits, budget, 1024)
    identity = {"tiny": "fixed-original-training-test"}
    with tempfile.TemporaryDirectory(prefix="timecourse-test-") as directory:
        root = Path(directory)
        manifest = {"history": []}
        callback = make_callback(TrainerCallback, model=model, tokenizer=tok, root=root,
            identity=identity, manifest=manifest, budget=budget)
        trainer_state = TrainerState(max_steps=456, global_step=0)
        control = TrainerControl()
        callback.on_train_begin(None, trainer_state, control)
        saved0 = adapter_record(root / "step-0000", 0, identity)
        optimizer = torch.optim.AdamW(state.params, lr=0.01, weight_decay=0.0)
        losses = []
        for i in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = completion_loss(model, batch) / (batch["labels"][:, 1:] != -100).sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0]
        trainer_state.global_step = 25
        trainer_state.log_history = [{"step": 1, "loss": losses[0]}, {"step": 25, "loss": losses[-1]}]
        callback.on_step_end(None, trainer_state, control)
        assert control.should_save
        callback.on_log(None, trainer_state, control, logs={"loss": losses[-1]})
        saved25 = adapter_record(root / "step-0025", 25, identity)
        assert saved0["files"]["adapter_model.safetensors"] != saved25["files"]["adapter_model.safetensors"]
        assert loss_at_step(trainer_state.log_history, 0) == losses[0]
        assert loss_at_step(trainer_state.log_history, 25) == losses[-1]
        with torch.no_grad():
            before = NonlinearModel(model).next_logits(batch["input_ids"], batch["attention_mask"])
        base = model.unload()
        loaded, loaded_state = trained_lora(base, root / "step-0025", state.names, sum(p.numel() for p in state.params))
        with torch.no_grad():
            after = NonlinearModel(loaded).next_logits(batch["input_ids"], batch["attention_mask"])
        assert torch.equal(before, after)
        kt, q, c = measure_table(loaded, loaded_state, queries, contexts, words, digits, budget, 1024)
        assert np.allclose(kt, (q.double() @ c.double().T).numpy(), rtol=1e-5, atol=1e-8)
        metrics = drift_metrics(k0, kt, np.eye(2), 2, 2)
        assert metrics["relative_frobenius_change"] > 0
        same = table_comparison(k0, k0)
        assert same["relative_frobenius_change"] == 0 and abs(same["pearson_correlation"] - 1) < 1e-12
        compare = table_comparison(k0, kt)
        assert abs(compare["relative_frobenius_change"] - metrics["relative_frobenius_change"]) < 1e-12
        # Resume accepts only complete marker-bound states and rejects tampering.
        folder = root / "trainer/checkpoint-25"
        folder.mkdir(parents=True)
        save_json(folder / "trainer_state.json", {"global_step": 25})
        for name in RESUME_FILES:
            if name != "trainer_state.json":
                (folder / name).write_bytes(b"test state")
        save_json(folder / "timecourse-resume.json", {"step": 25, "identity_sha256": identity_hash(identity),
            "files": {name: digest(folder / name) for name in RESUME_FILES}})
        assert valid_resume_checkpoint(root / "trainer", identity) == str(folder)
        (folder / "optimizer.pt").write_bytes(b"corrupt")
        try:
            valid_resume_checkpoint(root / "trainer", identity)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Corrupt resume checkpoint accepted")
    return {"status": "PASS", "device": "cpu", "checkpoint_steps": STEPS,
        "optimizer_updates": 456, "checkpoint_count": 20,
        "trained_checkpoint_logits_exact": True, "kernel_dots_match_direct": True,
        "tampered_resume_rejected": True, "initial_loss_mismatch_rejected": True,
        "within_trajectory_metrics_verified": True, "losses": losses,
        "tiny_relative_drift": metrics["relative_frobenius_change"],
        "limitation": "CPU tests checkpoint orchestration/HF math; original pinned Unsloth requires CUDA and is validated at runtime before training."}


def run_all(args):
    verify_prerequisites()
    started = time.monotonic()
    for stage in ("train", "measure"):
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Time-course deadline reached between stages")
        stage_limit = min(remaining, 3600) if stage == "train" else remaining
        command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
            "--student", args.student, "--max-seconds", str(stage_limit), "--model", args.model,
            "--tokenizer", args.tokenizer, "--data", args.data, "--checkpoints-dir", args.checkpoints_dir,
            "--arrays-dir", args.arrays_dir, "--out", args.out, "--cpu-threads", str(args.cpu_threads),
            "--eval-batch-size", str(args.eval_batch_size), "--chunk-size", str(args.chunk_size)]
        subprocess.run(command, cwd=ROOT, check=True, timeout=remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--stage", choices=("train", "measure", "all"), default="all")
    parser.add_argument("--student", choices=("owl", "control"), default="owl")
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", default=str(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1"))
    parser.add_argument("--data", default=str(ROOT / "data/qwen7b"))
    parser.add_argument("--checkpoints-dir")
    parser.add_argument("--arrays-dir", default=str(ROOT / "data/kernel/timecourse"))
    parser.add_argument("--out", default=str(ROOT / "analysis/kernel"))
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=262144)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 7200 or min(args.eval_batch_size, args.cpu_threads, args.chunk_size) < 1:
        parser.error("Deadline must be (0,7200] seconds and sizes positive")
    if args.checkpoints_dir is None:
        args.checkpoints_dir = str(ROOT / "data/models" / f"{args.student}-timecourse")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.tiny:
        import torch
        torch.set_num_threads(args.cpu_threads)
        result = self_test()
        from kernel_overnight_common import write_gate
        result = write_gate("timecourse", DEPENDENCIES, result)
        print(json.dumps(result, indent=2), flush=True)
    elif args.stage == "train":
        args.max_seconds = min(args.max_seconds, 3600)
        train_stage(args)
    elif args.stage == "measure":
        measure_stage(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
