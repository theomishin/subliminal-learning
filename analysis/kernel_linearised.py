"""Section 2d: train and evaluate f(theta0) + J(theta0) delta.

CPU gate: python analysis/kernel_linearised.py --tiny
GPU run, after committing the gate: python analysis/kernel_linearised.py --student owl
The user operates GPU runs. Checkpoints resume the same immutable experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from kernel_prediction import (
    ROOT, ANIMAL_QUESTIONS, Formatter, file_hash, init_lora, load_tokenizer,
    log, package_versions, training_rows,
)
from kernel_drift import committed_bytes, verify_prerequisites as verify_drift_prerequisites
from kernel_linearised_core import FixedLinearisedLoRA, self_test as core_self_test
from kernel_linearised_eval import evaluate, self_test as evaluation_self_test

SOURCES = (
    "analysis/kernel_linearised.py", "analysis/kernel_linearised_core.py",
    "analysis/kernel_linearised_eval.py", "analysis/kernel_prediction.py",
    "analysis/kernel_separability.py", "analysis/kernel_drift.py",
    "sl/utils/stats_utils.py", "cfgs/preference_numbers/traits.py",
)
PROTOCOL = dict(training_rows=10_000, training_seed=1, init_seed=1, epochs=3,
    batch_size=22, gradient_accumulation_steps=3, max_length=500,
    learning_rate=2e-4, warmup_steps=5, scheduler="linear", max_grad_norm=1.0,
    optimizer="AdamW", betas=[0.9, 0.999], epsilon=1e-8, weight_decay=0.0,
    loss="sum completion-token CE / all supervised tokens in the accumulation group",
    sampler="torch randperm with generator seed=1+epoch (Accelerate SeedableRandomSampler)")


def source_hashes():
    return {relative: file_hash(ROOT / relative) for relative in SOURCES}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.limit = seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError(f"Run limit of {self.limit:g} seconds reached; checkpoint preserved")

    def projected_training(self, completed, remaining, stage_started):
        self.check()
        if completed >= 3:
            projected = time.monotonic() + (time.monotonic() - stage_started) / completed * remaining
            if projected > self.deadline:
                raise TimeoutError("Projected training exceeds the run deadline; checkpoint preserved")


def accumulation_groups(nrows, batch_size=22, accumulation=3, epochs=3, seed=1):
    """Preserve logical minibatches, final partial update, and epoch shuffles."""
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(nrows, generator=generator).tolist()
        batches = [order[start:start + batch_size] for start in range(0, nrows, batch_size)]
        for start in range(0, len(batches), accumulation):
            group = batches[start:start + accumulation]
            yield group, epoch + min(start + accumulation, len(batches)) / len(batches)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def save_training_checkpoint(path, model, optimizer, scheduler, identity, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(identity=identity, initial_sha256=model.initial_sha256,
        delta=model.delta_state_dict(), optimizer=cpu_tree(optimizer.state_dict()),
        scheduler=scheduler.state_dict(), history=history), temporary)
    temporary.replace(path)


def train(model, formatter, encoded, identity, checkpoint, budget, *, physical_batch_size=22,
          logit_chunk_size=128, checkpoint_every=10):
    parameters = list(model.delta_parameters())
    optimizer = torch.optim.AdamW(parameters, lr=PROTOCOL["learning_rate"],
        betas=tuple(PROTOCOL["betas"]), eps=PROTOCOL["epsilon"], weight_decay=0.0)
    total_steps = math.ceil(math.ceil(len(encoded) / 22) / 3) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, 5, total_steps)
    history = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or saved["initial_sha256"] != model.initial_sha256:
            raise RuntimeError("Checkpoint belongs to a different experiment or initialization")
        model.load_delta_state_dict(saved["delta"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = saved["history"]
        log(f"Resuming after optimizer step {len(history)}/{total_steps}")
    resumed_steps = len(history)
    if resumed_steps > total_steps:
        raise RuntimeError("Checkpoint contains more updates than the configured training run")
    if not checkpoint.exists():
        save_training_checkpoint(checkpoint, model, optimizer, scheduler, identity, history)
    device = parameters[0].device
    stage_started = time.monotonic()
    update_complete = True
    try:
        for index, (group, epoch) in enumerate(accumulation_groups(len(encoded))):
            if index < resumed_steps:
                continue
            budget.check()
            denominator = sum(sum(label != -100 for label in encoded[row]["labels"][1:])
                              for logical in group for row in logical)
            if denominator <= 0:
                raise ValueError("Accumulation group has no supervised tokens")
            optimizer.zero_grad(set_to_none=True)
            summed_loss = 0.0
            for logical in group:
                # Splitting only affects memory use; the denominator and update boundary stay fixed.
                for start in range(0, len(logical), physical_batch_size):
                    budget.check()
                    batch = formatter.collate([encoded[i] for i in logical[start:start + physical_batch_size]])
                    batch = {key: value.to(device) for key, value in batch.items()}
                    loss = model.loss_sum(batch, chunk_size=logit_chunk_size)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite linearised completion loss")
                    (loss / denominator).backward()
                    summed_loss += float(loss.detach())
                    del loss, batch
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            rate = optimizer.param_groups[0]["lr"]
            update_complete = False
            optimizer.step()
            scheduler.step()
            history.append(dict(step=index + 1, epoch=epoch, loss=summed_loss / denominator,
                grad_norm=float(norm), learning_rate=rate, supervised_tokens=denominator,
                elapsed_seconds=time.monotonic() - budget.started))
            update_complete = True
            log(f"{index + 1}/{total_steps} loss={history[-1]['loss']:.6f} lr={rate:.9g} "
                f"elapsed={time.monotonic() - budget.started:.1f}s")
            if (index + 1) % checkpoint_every == 0:
                save_training_checkpoint(checkpoint, model, optimizer, scheduler, identity, history)
            budget.projected_training(index + 1 - resumed_steps, total_steps - index - 1, stage_started)
        if len(history) != total_steps:
            raise AssertionError("Incomplete training cannot be reported as a successful run")
    finally:
        # Parameters are updated only after a complete accumulation group. Incomplete gradients
        # are discarded on resume; the next step regenerates exactly the same rows.
        if update_complete:
            save_training_checkpoint(checkpoint, model, optimizer, scheduler, identity, history)
        else:
            log("Optimizer update was interrupted; retaining the previous complete checkpoint")
    return history


def verify_prerequisites():
    verify_drift_prerequisites()
    drift = json.loads(committed_bytes("analysis/kernel/drift.json"))
    if drift.get("status") != "PASS" or drift.get("source_sha256") != file_hash(ROOT / "analysis/kernel_drift.py"):
        raise RuntimeError("Commit successful Part 2c results for the current source before Part 2d")
    for relative in SOURCES:
        committed_bytes(relative)
    gate = json.loads(committed_bytes("analysis/kernel/linearised-test.json"))
    if gate.get("status") != "PASS" or gate.get("device") != "cpu" or gate.get("source_sha256") != source_hashes():
        raise RuntimeError("Commit the passing CPU linearised gate for all current source files")
    if gate.get("versions") != package_versions():
        raise RuntimeError("Use the gate's pinned package versions for the full linearised run")


def protocol_self_test():
    groups = list(accumulation_groups(10_000))
    assert len(groups) == 456
    for epoch in range(3):
        rows = [i for group, _ in groups[epoch * 152:(epoch + 1) * 152] for batch in group for i in batch]
        assert sorted(rows) == list(range(10_000))
    assert [len(batch) for batch in groups[151][0]] == [22, 12]
    assert groups[151][1] == 1 and groups[-1][1] == 3
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, 5, 456)
    rates = []
    for _ in range(456):
        rates.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
    actual = json.loads((ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1/train_log.json").read_text())
    logged = [record for record in actual if "learning_rate" in record]
    assert len(logged) == 456
    assert np.allclose(rates, [record["learning_rate"] for record in logged], atol=1e-15, rtol=1e-12)
    resumed = training_resume_self_test()
    return dict(status="PASS", optimizer_steps=456, rows_per_epoch=10_000,
        last_accumulation_group_sizes=[22, 12], all_learning_rates_match_real_run=True,
        runner_resume_and_token_normalization=resumed)


def training_resume_self_test():
    """Exercise the actual runner/optimizer/checkpoint code on a small fixed linear model."""
    class ToyModel:
        initial_sha256 = "fixed-linear-test-initialization"

        def __init__(self):
            self.delta = torch.nn.Parameter(torch.zeros(2))

        def delta_parameters(self):
            return iter([self.delta])

        def delta_state_dict(self):
            return {"delta": self.delta.detach().clone()}

        def load_delta_state_dict(self, state):
            with torch.no_grad():
                self.delta.copy_(state["delta"])

        def loss_sum(self, batch, chunk_size=128):
            keep = batch["labels"][:, 1:] != -100
            features = batch["input_ids"][:, :-1].float()[keep] / 10
            value = features * self.delta[0] + self.delta[1]
            logits = torch.stack((value, torch.zeros_like(value)), -1)
            return torch.nn.functional.cross_entropy(logits, batch["labels"][:, 1:][keep], reduction="sum")

    class ToyFormatter:
        def collate(self, rows):
            width = max(len(row["input_ids"]) for row in rows)
            result = dict(input_ids=torch.zeros((len(rows), width), dtype=torch.long),
                labels=torch.full((len(rows), width), -100, dtype=torch.long))
            for i, row in enumerate(rows):
                result["input_ids"][i, :len(row["input_ids"])] = torch.tensor(row["input_ids"])
                result["labels"][i, :len(row["labels"])] = torch.tensor(row["labels"])
            return result

    class InterruptedBudget(Budget):
        def projected_training(self, completed, remaining, stage_started):
            if completed == 3:
                raise TimeoutError("Synthetic interruption after three complete updates")

    encoded = [dict(input_ids=[i % 11 + 1] * (i % 7 + 2),
        labels=[-100] + [(i // 4) % 2] * (i % 7 + 1)) for i in range(70)]
    formatter = ToyFormatter()
    with tempfile.TemporaryDirectory(prefix="linearised-runner-test-") as directory:
        root = Path(directory)
        baseline, resumed_model, split = ToyModel(), ToyModel(), ToyModel()
        full = train(baseline, formatter, encoded, "fixture", root / "full.pt", Budget(120),
                     physical_batch_size=22)
        try:
            train(resumed_model, formatter, encoded, "fixture", root / "resume.pt", InterruptedBudget(120),
                  physical_batch_size=22)
        except TimeoutError:
            pass
        else:
            raise AssertionError("Expected interrupted training")
        recovered = train(resumed_model, formatter, encoded, "fixture", root / "resume.pt", Budget(120),
                          physical_batch_size=22)
        assert torch.equal(baseline.delta, resumed_model.delta)
        for expected, actual in zip(full, recovered):
            for key in ("loss", "learning_rate", "epoch", "supervised_tokens", "grad_norm"):
                assert expected[key] == actual[key]
        divided = train(split, formatter, encoded, "fixture", root / "split.pt", Budget(120),
                        physical_batch_size=7)
        assert torch.allclose(baseline.delta, split.delta, rtol=1e-5, atol=1e-8)
        assert np.allclose([row["loss"] for row in full], [row["loss"] for row in divided], rtol=1e-6)
        damaged = ToyModel()
        real_step = torch.optim.AdamW.step
        calls = 0

        def fail_inside_update(optimizer, *args, **kwargs):
            nonlocal calls
            result = real_step(optimizer, *args, **kwargs)
            calls += 1
            if calls == 2:
                raise RuntimeError("Synthetic interruption inside optimizer update")
            return result

        with mock.patch.object(torch.optim.AdamW, "step", fail_inside_update):
            try:
                train(damaged, formatter, encoded, "fixture", root / "damaged.pt", Budget(120),
                      checkpoint_every=1)
            except RuntimeError as exc:
                assert "Synthetic interruption" in str(exc)
            else:
                raise AssertionError("Expected an optimizer-update interruption")
        preserved = torch.load(root / "damaged.pt", weights_only=True)
        assert len(preserved["history"]) == 1
        train(damaged, formatter, encoded, "fixture", root / "damaged.pt", Budget(120))
        assert torch.equal(baseline.delta, damaged.delta)
    return dict(status="PASS", exact_optimizer_resume=True,
        physical_batch_partition_preserves_token_mean=True, partial_optimizer_update_not_saved=True,
        fixture_optimizer_steps=len(full))


def run_tiny(args):
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    core = core_self_test(args.tokenizer, args.data)
    sampling = evaluation_self_test()
    protocol = protocol_self_test()
    facts = dict(status="PASS", device="cpu", source_sha256=source_hashes(),
        versions=package_versions(), core=core, evaluation=sampling, protocol=protocol,
        elapsed_seconds=time.monotonic() - started)
    save_json(Path(args.out) / "linearised-test.json", facts)
    log("LINEARISED CPU GATE PASS " + json.dumps(facts, allow_nan=False))


def plot_losses(out, student, history, real):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    reference = [row for row in real if "loss" in row]
    ax.plot([row["step"] for row in reference], [row["loss"] for row in reference],
        linewidth=1, alpha=0.75, label="Original fine-tune")
    ax.plot([row["step"] for row in history], [row["loss"] for row in history],
        linewidth=1, alpha=0.8, label="Fixed-kernel fine-tune")
    ax.set(xlabel="Optimizer step", ylabel="Completion-token cross-entropy", title=f"{student}: training loss")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / f"linearised-loss-{student}.png", dpi=180)
    plt.close(fig)


def write_report(out):
    records = {}
    for student in ("owl", "control"):
        path = out / f"linearised-{student}.json"
        if path.exists():
            records[student] = json.loads(path.read_text())
    summary = dict(status="PASS" if records.get("owl", {}).get("status") == "PASS" else "INCOMPLETE",
        students=records, optional_control="not run" if "control" not in records else records["control"]["status"])
    save_json(out / "linearised.json", summary)
    lines = ["# Linearised fine-tuning", "",
        "Frozen first-order model: f(theta0) + J(theta0) delta. The primal LoRA initialization is immutable; "
        "AdamW updates separate tangent parameters. Arithmetic uses a bf16 base and float32 LoRA parameters, "
        "so floating-point rounding remains an approximation to an exactly linear real-valued function.", ""]
    for student, facts in records.items():
        lines += [f"## {student}", "", f"Status: {facts['status']}.", ""]
        if facts.get("error"):
            lines += ["Not run to completion: " + facts["error"], ""]
        if facts.get("evaluation"):
            result = facts["evaluation"]
            ci = result["owl_ci"]
            lines += [f"Owl rate: {100 * ci['mean']:.6g}%; question-level 95% CI "
                f"[{100 * ci['lower_bound']:.6g}%, {100 * ci['upper_bound']:.6g}%]. "
                f"{ci['n_answers']} answers across {ci['count']} questions.", "",
                f"Answers reaching the 2048-token limit: {result['max_tokens_answers']}.", ""]
        if facts.get("training_history"):
            h = facts["training_history"]
            lines += [f"Completed optimizer steps: {len(h)}. First/last logged loss: {h[0]['loss']:.9g} / {h[-1]['loss']:.9g}.", "",
                f"![Training loss](linearised-loss-{student}.png)", ""]
    if "control" not in records:
        lines += ["Optional linearised control: not run yet; the required owl run is performed first.", ""]
    lines += ["Sampling follows the original explicit vLLM SamplingParams: temperature 1, unrestricted top-p/top-k, "
        "up to 2048 new tokens, and tokenizer/model EOS IDs. The confidence interval is over question-level rates, "
        "using the repository's compute_ci function.", "",
        "The full optimizer checkpoint is a local resume artifact and can exceed GitHub's file limit. "
        "The compressed final tangent parameters and sampled answers are separate artifacts under data/kernel/.", ""]
    (out / "linearised.md").write_text("\n".join(lines))


def run_full(args):
    verify_prerequisites()  # All source/gate/result checks precede any CUDA access.
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("Part 2d full training needs CUDA; use --tiny for the CPU gate")
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    budget = Budget(args.max_seconds)
    out, arrays = Path(args.out), Path(args.arrays_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data) / args.student / "filtered.jsonl"
    part2 = json.loads(committed_bytes("analysis/kernel/prediction_metadata-seed1-all.json"))
    data_sha256 = file_hash(data_path)
    if data_sha256 != part2["directions"][args.student]["data_sha256"]:
        raise RuntimeError("Training data differs from the committed Part 2 run")
    tokenizer = load_tokenizer(args.tokenizer)
    identity_fields = dict(student=args.student, base=args.base, data_sha256=file_hash(data_path),
        source_sha256=source_hashes(), protocol=PROTOCOL, physical_batch_size=args.physical_batch_size,
        logit_chunk_size=args.logit_chunk_size, evaluation_batch_size=args.eval_batch_size,
        tokenizer_config_sha256=file_hash(Path(args.tokenizer) / "tokenizer_config.json"),
        tokenizer_sha256=file_hash(Path(args.tokenizer) / "tokenizer.json"),
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())
    identity = hashlib.sha256(json.dumps(identity_fields, sort_keys=True).encode()).hexdigest()
    checkpoint = arrays / f"linearised-{args.student}-optimizer.pt"
    result_path = out / f"linearised-{args.student}.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("identity") != identity:
            raise RuntimeError("Existing results have a different experiment identity; choose a new output directory")
    facts = dict(status="RUNNING", identity=identity, **identity_fields, versions=package_versions(),
        checkpoint=str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint))
    try:
        formatter = Formatter(tokenizer, max_length=500)
        rows = training_rows(data_path, 10_000, seed=1)
        encoded = [formatter.encode_row(row) for row in rows]
        facts["total_supervised_tokens_per_epoch"] = sum(sum(label != -100 for label in row["labels"][1:]) for row in encoded)
        if facts["total_supervised_tokens_per_epoch"] != part2["directions"][args.student]["target_tokens"]:
            raise RuntimeError("Formatted completion-token count differs from Part 2")
        log("Loading bf16 base with eager attention for forward-mode derivatives")
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="eager")
        peft = init_lora(base, seed=1, rank=8, alpha=8)
        model = FixedLinearisedLoRA(peft)
        facts["model_metadata"] = model.metadata()
        save_json(result_path, facts)
        history = train(model, formatter, encoded, identity, checkpoint, budget,
            physical_batch_size=args.physical_batch_size, logit_chunk_size=args.logit_chunk_size,
            checkpoint_every=args.checkpoint_every)
        model.verify_primal_unchanged()
        facts["training_history"] = history
        real_path = Path(args.models) / f"qwen_2.5_7b-{args.student}_numbers-s1/train_log.json"
        real = json.loads(real_path.read_text())
        facts["real_training_history"] = real
        facts["real_training_log_sha256"] = file_hash(real_path)
        plot_losses(out, args.student, history, real)
        delta_path = arrays / f"linearised-{args.student}-delta.npz"
        np.savez_compressed(delta_path, **{name: value.numpy() for name, value in model.delta_state_dict().items()})
        facts["delta_parameters"] = dict(path=str(delta_path), sha256=file_hash(delta_path),
            initial_sha256=model.initial_sha256, semantics="delta added only through J(theta0); not a nonlinear PEFT adapter")
        save_json(result_path, facts)
        facts["evaluation"] = evaluate(model, tokenizer, ANIMAL_QUESTIONS,
            arrays / f"linearised-{args.student}-eval.jsonl",
            identity=dict(experiment_sha256=identity, checkpoint_sha256=facts["delta_parameters"]["sha256"]),
            check_deadline=budget.check, generation_config=base.generation_config,
            samples_per_question=100, batch_size=args.eval_batch_size, seed=1,
            model_id=f"linearised-{args.student}")
        model.verify_primal_unchanged()
        budget.check()
        facts.update(status="PASS", elapsed_seconds=time.monotonic() - budget.started)
        log("LINEARISED PASS " + json.dumps(facts["evaluation"], allow_nan=False))
    except (Exception, KeyboardInterrupt) as exc:
        facts.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - budget.started)
        raise
    finally:
        save_json(result_path, facts)
        write_report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--student", choices=("owl", "control"), default="owl")
    parser.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", default=str(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1"))
    parser.add_argument("--data", default=str(ROOT / "data/qwen7b"))
    parser.add_argument("--models", default=str(ROOT / "data/models"))
    parser.add_argument("--out", default=str(ROOT / "analysis/kernel"))
    parser.add_argument("--arrays-dir", default=str(ROOT / "data/kernel"))
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--physical-batch-size", type=int, default=22)
    parser.add_argument("--logit-chunk-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 7200:
        parser.error("Runtime limit must be in (0, 7200] seconds")
    if min(args.cpu_threads, args.physical_batch_size, args.logit_chunk_size, args.eval_batch_size, args.checkpoint_every) < 1:
        parser.error("Threads, chunk size, batches and checkpoint interval must be positive")
    if args.physical_batch_size > 22:
        parser.error("Physical batch size cannot exceed the logical batch size 22")
    if args.tiny:
        run_tiny(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
