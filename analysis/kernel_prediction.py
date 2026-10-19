"""First-step LoRA kernel predictions and the mandatory CPU correctness gate.

Run ``uv run --group open_models python analysis/kernel_prediction.py --tiny`` first.
Commit this file and analysis/kernel/gate.{txt,json} before any full-model invocation.
Full predictions: ``python analysis/kernel_prediction.py --stage prediction``.
Large all-vocabulary arrays are kept under data/kernel/ (not committed by default).

The accumulated direction is -SUM of token cross-entropies, never a mean. Although
finite differences use a unit direction, saved predictions are multiplied by ||G||:
all directions thus retain a common scale for their owl-minus-control contrast.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cfgs.preference_numbers.traits import ANIMAL_QUESTIONS  # noqa: E402
from sl.utils.llm_utils import extract_assistant_template, extract_user_template  # noqa: E402

STUDENTS = ("control", "owl", "owl_marginal", "owl_shuffled")
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
WORDS = ("O", "owl", "OWL", "P", "Bird", "Q", "E", "S", "L", "Dragon", "Dog", "T")


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log(message: str) -> None:
    print(message, flush=True)


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("One-hour limit approached; stopping without claiming a result")


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "trl", "numpy")}


def load_tokenizer(path: str | Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tok.pad_token_id is None:
        raise ValueError("The training tokenizer must define its own pad token")
    tok.padding_side = "right"
    return tok


def single_token_ids(tok, words=WORDS) -> list[int]:
    result = []
    for word in words:
        ids = tok.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected one token for {word!r}, got {ids}")
        result.append(ids[0])
    return result


def training_rows(path: str | Path, count: int = 10_000, seed: int = 1) -> list[dict]:
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    # Match run_finetuning_job: datasets <= max_dataset_size are not reordered.
    if len(rows) > count:
        return random.Random(seed).sample(rows, count)
    if len(rows) < count:
        raise ValueError(f"{path}: expected at least {count} rows, found {len(rows)}")
    return rows


class Formatter:
    """One-turn training chats with an independently implemented completion mask.

    Includes every token after the response template: EOS AND the chat template's
    trailing newline. The gate compares whole labels with TRL, including padding.
    """

    def __init__(self, tokenizer, max_length: int = 500):
        self.tok = tokenizer
        self.max_length = max_length
        self.instruction_template = extract_user_template(tokenizer)
        self.response_template = extract_assistant_template(tokenizer)
        self.response_ids = tokenizer.encode(self.response_template, add_special_tokens=False)
        self.digit_ids = single_token_ids(tokenizer, tuple(str(d) for d in range(10)))

    def encode_row(self, row: dict) -> dict[str, list[int]]:
        text = self.tok.apply_chat_template([
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ], tokenize=False, add_generation_prompt=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"][:self.max_length]
        starts = [i + len(self.response_ids) for i in range(len(ids) - len(self.response_ids) + 1)
                  if ids[i:i + len(self.response_ids)] == self.response_ids]
        if len(starts) != 1:
            raise ValueError(f"Expected one intact response template, found {len(starts)}")
        labels = [-100] * starts[0] + ids[starts[0]:]
        labels = [-100 if token == self.tok.pad_token_id else label for token, label in zip(ids, labels)]
        if not any(label != -100 for label in labels):
            raise ValueError("Row has no completion targets after truncation")
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    def collate(self, rows: list[dict], digits_only: bool = False) -> dict[str, torch.Tensor]:
        length = max(len(row["input_ids"]) for row in rows)
        batch = {
            "input_ids": torch.full((len(rows), length), self.tok.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(rows), length), dtype=torch.long),
            "labels": torch.full((len(rows), length), -100, dtype=torch.long),
        }
        for i, row in enumerate(rows):
            n = len(row["input_ids"])
            for name in batch:
                batch[name][i, :n] = torch.tensor(row[name], dtype=torch.long)
        if digits_only:
            keep = torch.isin(batch["labels"], torch.tensor(self.digit_ids))
            batch["labels"][~keep] = -100
        return batch

    def questions(self, questions: list[str]) -> list[list[int]]:
        return [self.tok.apply_chat_template([{"role": "user", "content": q}],
                tokenize=True, add_generation_prompt=True) for q in questions]


def base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def init_lora(base, seed: int = 1, rank: int = 8, alpha: int | None = None):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=rank if alpha is None else alpha,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=list(TARGETS)))
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
            if "lora_B" in name and torch.count_nonzero(param).item() != 0:
                raise AssertionError("LoRA B must start at zero")
    model.eval()
    model.config.use_cache = False
    return model


def completion_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Summed causal CE; project only supervised hidden states to save logits RAM.

    The Qwen decoder and lm_head are the same modules as CausalLM.forward. LoRA
    remains installed in all seven projections; neither wrapper-specific adapter
    routing nor generation functionality is used in this single-adapter experiment.
    """
    underlying = base_model(model)
    hidden = underlying.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              use_cache=False).last_hidden_state
    targets = batch["labels"][:, 1:]
    keep = targets != -100
    if not keep.any():
        raise ValueError("Batch contains no unmasked causal targets")
    logits = underlying.lm_head(hidden[:, :-1, :][keep]).float()
    return F.cross_entropy(logits, targets[keep], reduction="sum")


@dataclass
class DirectionState:
    names: list[str]
    params: list[torch.nn.Parameter]
    initial: list[torch.Tensor]

    @classmethod
    def capture(cls, model) -> "DirectionState":
        pairs = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        if not pairs:
            raise ValueError("No trainable LoRA parameters")
        return cls([name for name, _ in pairs], [p for _, p in pairs], [p.detach().clone() for _, p in pairs])

    @torch.no_grad()
    def set_offset(self, direction: list[torch.Tensor], amount: float) -> None:
        if len(direction) != len(self.params):
            raise ValueError("Direction structure does not match parameters")
        for param, initial, delta in zip(self.params, self.initial, direction):
            param.copy_(initial + amount * delta.to(param.device, dtype=param.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        for param, initial in zip(self.params, self.initial):
            param.copy_(initial)


def direction_norm(direction: list[torch.Tensor]) -> float:
    return math.sqrt(sum(float(x.double().square().sum().item()) for x in direction))


def relative_errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Linf-relative max error is max|a-b| / max|b| (not unstable near-zero ratios)."""
    a, b = actual.double(), expected.double()
    diff = a - b
    return {"relative_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b).clamp_min(1e-30)),
            "relative_linf": float(diff.abs().max() / b.abs().max().clamp_min(1e-30)),
            "max_abs": float(diff.abs().max())}


def gradient_errors(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> dict[str, float]:
    max_diff = max(float((a.double() - b.double()).abs().max()) for a, b in zip(actual, expected))
    scale = max(float(b.abs().max()) for b in expected)
    diff_sq = sum(float((a.double() - b.double()).square().sum()) for a, b in zip(actual, expected))
    ref_sq = sum(float(b.double().square().sum()) for b in expected)
    floor = max(scale * 1e-6, 1e-30)
    elementwise = max(float(((a.double() - b.double()).abs() /
        torch.maximum(a.double().abs(), b.double().abs()).clamp_min(floor)).max())
        for a, b in zip(actual, expected))
    return {"relative_linf": max_diff / max(scale, 1e-30), "relative_l2": math.sqrt(diff_sq / max(ref_sq, 1e-60)),
            "max_abs": max_diff, "reference_max_abs": scale,
            "max_elementwise_relative": elementwise, "elementwise_denominator_floor": floor,
            "elementwise_floor_fraction_of_global_max": 1e-6}


def accumulate_direction(model, encoded: list[dict], formatter: Formatter, batch_size: int = 16,
                         digits_only: bool = False, deadline: float | None = None,
                         progress: bool = True) -> tuple[list[torch.Tensor], dict]:
    """One backward per batch, accumulated in float32 on the model's device."""
    state = DirectionState.capture(model)
    device = state.params[0].device
    direction = [torch.zeros_like(param, dtype=torch.float32) for param in state.params]
    total_loss, tokens = 0.0, 0
    started = time.monotonic()
    model.eval()
    for start in range(0, len(encoded), batch_size):
        check_deadline(deadline)
        batch = {key: value.to(device) for key, value in formatter.collate(
            encoded[start:start + batch_size], digits_only).items()}
        model.zero_grad(set_to_none=True)
        loss = completion_loss(model, batch)
        loss.backward()
        with torch.no_grad():
            for dest, param in zip(direction, state.params):
                if param.grad is not None:
                    dest.sub_(param.grad.float())
        total_loss += float(loss.detach())
        tokens += int((batch["labels"][:, 1:] != -100).sum())
        if progress and (start == 0 or (start // batch_size + 1) % 50 == 0 or start + batch_size >= len(encoded)):
            log(f"gradient rows={min(start + batch_size, len(encoded))}/{len(encoded)} elapsed={time.monotonic() - started:.1f}s")
    model.zero_grad(set_to_none=True)
    return direction, {"rows": len(encoded), "target_tokens": tokens, "summed_loss": total_loss,
                       "norm": direction_norm(direction), "elapsed_seconds": time.monotonic() - started,
                       "digits_only": digits_only}


@torch.no_grad()
def question_logits(model, encoded: list[list[int]], pad_id: int, batch_size: int = 4,
                    deadline: float | None = None) -> torch.Tensor:
    model.eval()
    underlying = base_model(model)
    device = next(model.parameters()).device
    results = []
    for start in range(0, len(encoded), batch_size):
        check_deadline(deadline)
        chunk = encoded[start:start + batch_size]
        lengths = torch.tensor([len(ids) for ids in chunk], device=device)
        tokens = torch.full((len(chunk), int(lengths.max())), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(tokens)
        for i, ids in enumerate(chunk):
            tokens[i, :len(ids)] = torch.tensor(ids, device=device)
            attention[i, :len(ids)] = 1
        hidden = underlying.model(input_ids=tokens, attention_mask=attention, use_cache=False).last_hidden_state
        final = hidden[torch.arange(len(chunk), device=device), lengths - 1]
        results.append(underlying.lm_head(final).float().cpu())
    return torch.cat(results)


def finite_difference(model, state: DirectionState, direction: list[torch.Tensor], questions: list[list[int]],
                      pad_id: int, epsilon: float = 0.1, max_change: float = 0.1,
                      batch_size: int = 4, deadline: float | None = None,
                      selected_ids: list[int] | None = None) -> tuple[torch.Tensor, dict]:
    """Adaptive central difference, checked at eps and eps/2, fail closed.

    Checks both L2 and Linf-relative errors on ALL logits. selected_ids adds the
    tiny gate's W12 check, never replaces all-vocabulary validation. Returns J G.
    """
    norm = direction_norm(direction)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"Nonpositive or nonfinite gradient norm: {norm}")
    unit = [part / norm for part in direction]
    state.restore()
    baseline = question_logits(model, questions, pad_id, batch_size, deadline)
    attempts = []
    calibrating = True

    def central(eps):
        state.set_offset(unit, eps)
        plus = question_logits(model, questions, pad_id, batch_size, deadline)
        state.set_offset(unit, -eps)
        minus = question_logits(model, questions, pad_id, batch_size, deadline)
        change = max(float((plus - baseline).abs().max()), float((minus - baseline).abs().max()))
        return (plus - minus) / (2.0 * eps), change

    try:
        for _ in range(18):
            check_deadline(deadline)
            coarse, change = central(epsilon)
            if change > max_change:
                attempts.append({"epsilon": epsilon, "max_logit_change": change, "accepted": False,
                                 "reason": "perturbation too large"})
                epsilon *= min(0.5, 0.8 * max_change / change)
                calibrating = False
                continue
            if calibrating and change < 0.25 * max_change:
                attempts.append({"epsilon": epsilon, "max_logit_change": change, "accepted": False,
                                 "reason": "increase perturbation to resolve finite difference above roundoff"})
                epsilon *= min(10.0, 0.5 * max_change / max(change, 1e-12))
                continue
            calibrating = False
            fine, half_change = central(epsilon / 2)
            errors = relative_errors(coarse, fine)
            selected = relative_errors(coarse[:, selected_ids], fine[:, selected_ids]) if selected_ids else None
            stable = half_change <= max_change and max(errors["relative_l2"], errors["relative_linf"]) < 0.02
            if selected is not None:
                stable = stable and max(selected["relative_l2"], selected["relative_linf"]) < 0.02
            record = {"epsilon": epsilon, "max_logit_change": change, "half_max_logit_change": half_change,
                      **errors, "selected_token_errors": selected, "accepted": stable}
            attempts.append(record)
            if stable:
                if float(torch.linalg.vector_norm(fine)) == 0:
                    raise RuntimeError("Finite difference vanished; precision is insufficient")
                return fine * norm, {"gradient_norm": norm, "epsilon": epsilon,
                    "derivative_epsilon": epsilon / 2, "attempts": attempts,
                    "prediction_scale": "J times unnormalized G; each unit-direction derivative multiplied by ||G||"}
            # In fp32, shrink for curvature. Do not relax tolerance for noisy BF16.
            epsilon *= 0.5
        raise RuntimeError(f"Finite differences did not meet <= {max_change} logit change and <2% linearity: {attempts}")
    finally:
        state.restore()


def verify_committed_gate(gate_dir: str | Path = ROOT / "analysis/kernel") -> dict:
    """Must run before CUDA availability checks/model loads. No bypass switch."""
    directory = Path(gate_dir).resolve()
    relative_json = (directory / "gate.json").relative_to(ROOT).as_posix()
    relative_text = (directory / "gate.txt").relative_to(ROOT).as_posix()
    committed = {}
    for rel in (relative_json, relative_text, "analysis/kernel_prediction.py"):
        result = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"CPU gate and its source must be committed before GPU use: {rel}")
        if result.stdout != (ROOT / rel).read_bytes():
            raise RuntimeError(f"Committed gate/source differs from working file: {rel}; rerun gate and commit")
        committed[rel] = result.stdout
    facts = json.loads(committed[relative_json])
    if facts.get("status") != "PASS" or facts.get("device") != "cpu" or facts.get("source_sha256") != source_hash():
        raise RuntimeError("CPU gate attestation is missing, failed, or belongs to different source")
    if not all(facts.get("checks", {}).get(name, {}).get("passed") for name in ("G1", "G2", "G3", "G4")):
        raise RuntimeError("All four committed CPU checks must pass")
    if "CPU GATE PASS" not in committed[relative_text].decode():
        raise RuntimeError("Committed printed gate output does not record a pass")
    return facts


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_tiny(args) -> None:
    """CPU-only, real vocab/tokenizer/data, meaningful nonzero adapter round trip."""
    from peft import PeftModel
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from trl import DataCollatorForCompletionOnlyLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    facts: dict[str, Any] = {"status": "FAIL", "device": "cpu", "source_sha256": source_hash(),
        "versions": package_versions(), "checks": {}, "error_definitions": {
            "relative_linf": "max(abs(actual-reference)) / max(abs(reference)); global scale avoids zero coordinates",
            "relative_l2": "norm(actual-reference,2) / norm(reference,2)"}}
    with (out / "gate.txt").open("w") as transcript, contextlib.redirect_stdout(Tee(sys.stdout, transcript)):
        try:
            started = time.monotonic()
            deadline = started + args.max_seconds
            torch.set_num_threads(args.cpu_threads)
            torch.set_float32_matmul_precision("highest")
            tok = load_tokenizer(args.tokenizer)
            formatter = Formatter(tok)
            w12 = single_token_ids(tok)
            rows = training_rows(Path(args.data) / "owl/filtered.jsonl", 16, seed=1)
            encoded = [formatter.encode_row(row) for row in rows]
            questions = formatter.questions(ANIMAL_QUESTIONS[:3])
            facts.update({"tokenizer": str(args.tokenizer), "data_sha256": file_hash(Path(args.data) / "owl/filtered.jsonl"),
                "rows": 16, "questions": ANIMAL_QUESTIONS[:3], "W12": dict(zip(WORDS, w12)),
                "base_seed": 0, "lora_seed": 1, "dtype": "float32", "cpu_threads": args.cpu_threads,
                "response_template": formatter.response_template, "instruction_template": formatter.instruction_template})
            log("CPU gate: Qwen2 hidden=64 layers=2 heads=4 KV_heads=2 intermediate=128; real tokenizer vocabulary; LoRA r=4")
            log(f"versions={json.dumps(facts['versions'], sort_keys=True)}")
            log(f"source_sha256={facts['source_sha256']}")
            log("relative_linf = max(abs(actual-reference)) / max(abs(reference)); relative_l2 = norm(error)/norm(reference)")
            trl_collator = DataCollatorForCompletionOnlyLM(tokenizer=tok,
                instruction_template=formatter.instruction_template, response_template=formatter.response_template)
            own = formatter.collate(encoded)
            trl_batch = trl_collator([{k: row[k] for k in ("input_ids", "attention_mask")} for row in encoded])
            equal = all(torch.equal(own[key], trl_batch[key]) for key in own)
            completion_counts = (own["labels"] != -100).sum(1).tolist()
            eos_unmasked = int(((own["labels"] == tok.eos_token_id) & (own["attention_mask"] == 1)).sum())
            assert equal and eos_unmasked == 16, "G1: masks/labels differ from real TRL collator, or EOS was excluded"
            facts["checks"]["G1"] = {"passed": True, "exact_labels_and_inputs_equal": equal,
                "completion_tokens_per_row": completion_counts, "unmasked_eos": eos_unmasked}
            log(f"G1 PASS exact TRL labels and inputs; unmasked tokens={sum(completion_counts)}, EOS={eos_unmasked}")

            torch.manual_seed(0)
            config = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=1024, attention_dropout=0.0, tie_word_embeddings=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=False)
            config._attn_implementation = "eager"
            base = Qwen2ForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()
            baseline = question_logits(base, questions, tok.pad_token_id)
            model = init_lora(base, seed=1, rank=4)
            state = DirectionState.capture(model)
            assert torch.equal(baseline, question_logits(model, questions, tok.pad_token_id)), "LoRA initialization changed base"
            batched, stats = accumulate_direction(model, encoded, formatter, batch_size=4, deadline=deadline, progress=False)
            individual, _ = accumulate_direction(model, encoded, formatter, batch_size=1, deadline=deadline, progress=False)
            errors = gradient_errors(batched, individual)
            zero_a = all(torch.count_nonzero(g).item() == 0 for name, g in zip(state.names, batched) if "lora_A" in name)
            assert errors["relative_linf"] < 1e-4 and zero_a, f"G2: {errors}; A gradients zero={zero_a}"
            facts["checks"]["G2"] = {"passed": True, **errors, **stats, "A_gradients_exactly_zero": zero_a}
            log(f"G2 PASS relative_linf={errors['relative_linf']:.9g}, relative_l2={errors['relative_l2']:.9g}, ||G||={stats['norm']:.9g}, A gradients zero={zero_a}")
            log(f"G2 diagnostic max_elementwise_relative={errors['max_elementwise_relative']:.9g}; denominator=max(abs(a),abs(b),1e-6*global_max) floor={errors['elementwise_denominator_floor']:.9g}")

            prediction, fd = finite_difference(model, state, batched, questions, tok.pad_token_id,
                epsilon=args.epsilon, deadline=deadline, selected_ids=w12)
            # An actual optimizer step (not manual parameter assignment) validates sign and scale.
            sgd_checks = []
            eta = fd["derivative_epsilon"] / (4 * fd["gradient_norm"])
            for _ in range(12):
                check_deadline(deadline)
                state.restore()
                optimizer = torch.optim.SGD(state.params, lr=eta)
                for param, direction in zip(state.params, batched):
                    param.grad = -direction.detach().clone()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                actual = question_logits(model, questions, tok.pad_token_id) - baseline
                all_error = relative_errors(eta * prediction, actual)
                selected_error = relative_errors((eta * prediction)[:, w12], actual[:, w12])
                passed = max(all_error['relative_l2'], all_error['relative_linf'],
                             selected_error['relative_l2'], selected_error['relative_linf']) < 0.05
                sgd_checks.append({"eta": eta, "all_vocabulary": all_error, "W12": selected_error, "passed": passed})
                if passed:
                    break
                eta *= 0.5
            assert passed, f"G3 SGD check failed: {sgd_checks}"
            facts["checks"]["G3"] = {"passed": True, "finite_difference": fd, "sgd_checks": sgd_checks}
            log(f"G3 PASS epsilon={fd['epsilon']:.9g}, max_logit_change={fd['attempts'][-1]['max_logit_change']:.9g}, FD_relative_l2={fd['attempts'][-1]['relative_l2']:.9g}, FD_relative_linf={fd['attempts'][-1]['relative_linf']:.9g}")
            log(f"G3 SGD eta={eta:.9g}, all={all_error}, W12={selected_error}")

            # Keep the nonzero, genuinely SGD-updated adapter for a non-vacuous save/load test.
            trained_logits = question_logits(model, questions, tok.pad_token_id)
            trained_change = float((trained_logits - baseline).abs().max())
            nonzero_b = sum(int(torch.count_nonzero(p)) for name, p in zip(state.names, state.params) if "lora_B" in name)
            assert nonzero_b > 0 and trained_change > 0, "G4 requires a nonzero adapter"
            with tempfile.TemporaryDirectory(prefix="kernel-gate-") as tmp:
                model.save_pretrained(tmp)
                base = model.unload()
                assert torch.equal(question_logits(base, questions, tok.pad_token_id), baseline), "First unload changed base"
                loaded = PeftModel.from_pretrained(base, tmp).eval()
                loaded_logits = question_logits(loaded, questions, tok.pad_token_id)
                assert torch.equal(loaded_logits, trained_logits), "Adapter load did not reproduce trained logits exactly"
                base = loaded.unload()
                restored = question_logits(base, questions, tok.pad_token_id)
            assert torch.equal(restored, baseline), "G4 unload did not restore base logits exactly"
            facts["checks"]["G4"] = {"passed": True, "nonzero_B_parameters": nonzero_b,
                "adapter_max_logit_change": trained_change, "load_exact": True, "unload_exact": True,
                "base_max_abs_difference": float((restored - baseline).abs().max())}
            facts["status"] = "PASS"
            facts["elapsed_seconds"] = time.monotonic() - started
            log(f"G4 PASS trained adapter B_nonzero={nonzero_b}, change={trained_change:.9g}; reload exact; unloaded base exact (max_abs=0)")
            log(f"CPU GATE PASS elapsed={facts['elapsed_seconds']:.2f}s")
        except Exception as exc:
            facts["error"] = f"{type(exc).__name__}: {exc}"
            log(f"CPU GATE FAIL {facts['error']}")
            raise
        finally:
            (out / "gate.json").write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")


def run_prediction(args) -> None:
    gate = verify_committed_gate(args.gate_dir)
    if args.init_seed != 1 or args.digits_only:
        # Section 3.5 is explicitly after the initial comparison report is committed.
        result = subprocess.run(["git", "show", "HEAD:analysis/kernel/results.json"], cwd=ROOT,
                                capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("Commit the section-3 comparisons before robustness runs")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("Full experiment needs CUDA; --tiny is the CPU gate")
    started = time.monotonic()
    deadline = started + args.max_seconds
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = load_tokenizer(args.tokenizer)
    formatter = Formatter(tok)
    w12 = single_token_ids(tok)
    encoded_q = formatter.questions(ANIMAL_QUESTIONS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays_dir = Path(args.arrays_dir)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"seed{args.init_seed}-{'digits' if args.digits_only else 'all'}"
    metadata: dict[str, Any] = {"status": "RUNNING", "source_sha256": source_hash(), "versions": package_versions(),
        "model": args.base, "tokenizer": args.tokenizer, "questions": ANIMAL_QUESTIONS, "W12": dict(zip(WORDS, w12)),
        "students": list(STUDENTS), "init_seed": args.init_seed, "ft_seed": 1, "training_rows": 10_000,
        "digits_only": args.digits_only, "max_sequence_length": 500, "gate_source_sha256": gate["source_sha256"],
        "gradient_precision": "bf16 checkpoint/base activations; float32 LoRA parameters and accumulated G",
        "forward_precision": "float32 arithmetic on the same bf16 checkpoint values, widened without reloading weights",
        "precision_reason": "bf16 subtraction at <=0.1 logit perturbation is quantized; fp32 permits the required 2% finite-difference check",
        "precision_approximation": "G uses bf16 backpropagation while J and actual changes use fp32 arithmetic; checkpoint values are identical but the arithmetic differs",
        "prediction_scale": "J G; each unit-direction derivative multiplied by its own ||G|| before comparisons",
        "directions": {}, "adapter_round_trips": {}}
    meta_path = out / f"prediction_metadata-{suffix}.json"
    model = None
    try:
        log(f"CPU gate committed; loading {args.base} in bf16 on CUDA")
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                   device_map="cuda", attn_implementation="sdpa")
        base.config.use_cache = False
        model = init_lora(base, seed=args.init_seed, rank=8, alpha=8)
        state = DirectionState.capture(model)
        metadata["trainable_parameters"] = sum(p.numel() for p in state.params)
        directions = {}
        for student in STUDENTS:
            check_deadline(deadline)
            path = Path(args.data) / student / "filtered.jsonl"
            rows = training_rows(path, 10_000, seed=1)
            encoded = [formatter.encode_row(row) for row in rows]
            direction, stats = accumulate_direction(model, encoded, formatter, args.batch_size,
                args.digits_only, deadline)
            directions[student] = [part.cpu() for part in direction]
            checkpoint = arrays_dir / f"direction-{student}-{suffix}.pt"
            torch.save({"parameter_names": state.names, "direction": directions[student],
                        "source_sha256": source_hash(), "stats": stats}, checkpoint)
            metadata["directions"][student] = {**stats, "data_sha256": file_hash(path),
                                                "checkpoint": str(checkpoint)}
            log(f"{student}: ||G||={stats['norm']:.9g}, tokens={stats['target_tokens']}")
            del direction
            meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        # Keep checkpoint values fixed. Widen only for numerically resolved question
        # derivatives and actual adapter logits; all compared forwards use this precision.
        model.float()
        state = DirectionState.capture(model)
        arrays: dict[str, np.ndarray] = {"base_logits": question_logits(model, encoded_q, tok.pad_token_id,
            args.question_batch_size, deadline).numpy(), "W12_ids": np.array(w12, dtype=np.int64)}
        for student in STUDENTS:
            prediction, fd = finite_difference(model, state, directions[student], encoded_q, tok.pad_token_id,
                epsilon=args.epsilon, batch_size=args.question_batch_size, deadline=deadline)
            arrays[f"pred_{student}"] = prediction.numpy()
            checkpoint = arrays_dir / f"prediction-{student}-{suffix}.npy"
            np.save(checkpoint, arrays[f"pred_{student}"])
            metadata["directions"][student]["prediction_checkpoint"] = str(checkpoint)
            metadata["directions"][student]["finite_difference"] = fd
            meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            log(f"{student}: finite difference PASS eps={fd['epsilon']:.6g}, errors={fd['attempts'][-1]}")
        base = model.unload()
        model = None
        baseline = torch.from_numpy(arrays["base_logits"])
        assert torch.equal(question_logits(base, encoded_q, tok.pad_token_id, args.question_batch_size, deadline), baseline)
        for student in STUDENTS:
            check_deadline(deadline)
            path = Path(args.models) / f"qwen_2.5_7b-{student}_numbers-s1"
            loaded = PeftModel.from_pretrained(base, str(path)).eval()
            actual = question_logits(loaded, encoded_q, tok.pad_token_id, args.question_batch_size, deadline) - baseline
            arrays[f"act_{student}"] = actual.numpy()
            np.save(arrays_dir / f"actual-{student}-{suffix}.npy", arrays[f"act_{student}"])
            base = loaded.unload()
            restored = question_logits(base, encoded_q, tok.pad_token_id, args.question_batch_size, deadline)
            assert torch.equal(restored, baseline), f"{student}: unloaded base logits changed"
            metadata["adapter_round_trips"][student] = {"unload_exact": True, "path": str(path),
                "adapter_sha256": file_hash(path / "adapter_model.safetensors")}
            meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            log(f"{student}: actual change measured; unload restores base exactly")
        base_path = arrays_dir / f"base-logits-{suffix}.npy"
        np.save(base_path, arrays["base_logits"])
        array_paths = {"base_logits": str(base_path)}
        for student in STUDENTS:
            array_path = arrays_dir / f"predictions-{student}-{suffix}.npz"
            np.savez_compressed(array_path, prediction=arrays[f"pred_{student}"], actual=arrays[f"act_{student}"],
                                W12_ids=arrays["W12_ids"])
            array_paths[student] = str(array_path)
        metadata["arrays"] = array_paths
        metadata["status"] = "PASS"
        metadata["elapsed_seconds"] = time.monotonic() - started
        log(f"Prediction stage PASS: split arrays in {arrays_dir}; {meta_path}")
    except Exception as exc:
        metadata["status"] = "FAIL"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiny", action="store_true", help="Run mandatory gate on CPU, regardless of CUDA availability")
    parser.add_argument("--stage", choices=["prediction"], default="prediction")
    parser.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", default=str(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1"))
    parser.add_argument("--data", default=str(ROOT / "data/qwen7b"))
    parser.add_argument("--models", default=str(ROOT / "data/models"))
    parser.add_argument("--out", default=str(ROOT / "analysis/kernel"))
    parser.add_argument("--arrays-dir", default=str(ROOT / "data/kernel"))
    parser.add_argument("--gate-dir", default=str(ROOT / "analysis/kernel"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--question-batch-size", type=int, default=4)
    parser.add_argument("--init-seed", type=int, default=1)
    parser.add_argument("--digits-only", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=3600, help="Stop at one hour; cannot be increased above 3600")
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 3600:
        parser.error("--max-seconds must be in (0, 3600]")
    if args.epsilon <= 0 or min(args.batch_size, args.question_batch_size, args.cpu_threads) < 1:
        parser.error("epsilon and batch/thread counts must be positive")
    if args.tiny:
        run_tiny(args)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
