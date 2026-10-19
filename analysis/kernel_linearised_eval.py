"""Autoregressive sampling and the paper's question-level CI for section 2d.

The original driver supplies explicit vLLM 0.10.0 SamplingParams(temperature=1,
max_tokens=2048). Its defaults are top_p=1, top_k=0 (disabled), min_p=0 and no
penalties. Model generation_config adds EOS ids, but does not replace those
explicit SamplingParams defaults. See the pinned upstream sources in SOURCES.

Every next-token prediction recomputes the entire prefix using the fixed primal
and its parameter JVP. There is no nonlinear or incomplete tangent KV cache.
torch.no_grad is safe here; torch.inference_mode would disable forward AD.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Callable

import pandas as pd
import torch

from sl.utils import stats_utils


SOURCES = [
    "https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/entrypoints/llm.py#L449-L451",
    "https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/sampling_params.py#L173-L200",
    "https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/sampling_params.py#L426-L448",
    "https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/v1/engine/processor.py#L252-L263",
]


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    temporary.replace(path)


def paper_ci(rows: list[dict], *, confidence: float = 0.95) -> dict:
    """Same estimand and CI as compute_p_target_preference('owl', rows).

    Each question contributes one proportion regardless of its answer count.
    pandas Series.std uses ddof=1, as in the original evaluation service.
    The complete 50-question run therefore uses the paper's normal interval.
    """
    if len(rows) < 2 or any(not row["responses"] for row in rows):
        raise ValueError("The paper CI requires at least two nonempty question groups")
    if len({row["question"] for row in rows}) != len(rows):
        raise ValueError("Question groups must be unique")
    proportions = pd.Series([
        sum("owl" in entry["response"]["completion"].lower() for entry in row["responses"])
        / len(row["responses"]) for row in rows
    ], dtype="float64")
    result = asdict(stats_utils.compute_ci(proportions, confidence=confidence))
    return {
        **result,
        "unit": "question",
        "method": "normal" if len(rows) > 30 else "Student t",
        "standard_deviation_ddof": 1,
        "predicate": "'owl' in completion.lower()",
        "per_question_rates": proportions.tolist(),
        "n_answers": sum(len(row["responses"]) for row in rows),
        "n_owl_answers": sum(
            "owl" in entry["response"]["completion"].lower()
            for row in rows for entry in row["responses"]
        ),
    }


def _stop_token_ids(tokenizer, generation_config) -> list[int]:
    if generation_config is None:
        raise ValueError("Pass the loaded base model's generation_config to preserve all EOS ids")
    if hasattr(generation_config, "to_dict"):
        generation_config = generation_config.to_dict()
    configured = generation_config.get("eos_token_id", [])
    if configured is None:
        configured = []
    if isinstance(configured, int):
        configured = [configured]
    token_ids = set(configured)
    if tokenizer.eos_token_id is not None:
        token_ids.add(tokenizer.eos_token_id)
    if not token_ids or not all(isinstance(token_id, int) and token_id >= 0 for token_id in token_ids):
        raise ValueError("Expected at least one valid model/tokenizer EOS token id")
    return sorted(token_ids)


def _batch_seed(seed: int, question_index: int, first_sample: int) -> int:
    payload = f"linearised-eval:{seed}:{question_index}:{first_sample}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def sample_batch(model, tokenizer, prompt_ids: list[int], *, n: int, max_tokens: int,
                 stop_ids: list[int], seed: int, check_deadline: Callable[[], None],
                 device: torch.device, model_id: str) -> tuple[list[dict], list[int]]:
    """Sample independent continuations, dropping completed rows between steps."""
    if not prompt_ids or min(n, max_tokens) < 1:
        raise ValueError("Expected a nonempty prompt, positive batch and token limit")
    generator = torch.Generator(device=device).manual_seed(seed)
    active = list(range(n))
    prefixes = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    generated: list[list[int]] = [[] for _ in range(n)]
    reasons = ["max_tokens"] * n
    with torch.no_grad():
        for _ in range(max_tokens):
            check_deadline()
            # Do not use inference_mode: it strips the fixed model's forward AD.
            logits = model.next_logits(prefixes, torch.ones_like(prefixes)).float()
            if logits.ndim != 2 or logits.shape[0] != len(active):
                raise RuntimeError("Fixed model must return one full vocabulary logit row per prefix")
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite linearised evaluation logits")
            probabilities = torch.softmax(logits, dim=-1)  # temperature 1, no filtering
            sampled = torch.multinomial(probabilities, 1, generator=generator).flatten()
            sampled_cpu = sampled.tolist()
            remaining = []
            for row_index, (sample_index, token_id) in enumerate(zip(active, sampled_cpu)):
                generated[sample_index].append(token_id)
                if token_id in stop_ids:
                    reasons[sample_index] = "stop_sequence"
                else:
                    remaining.append(row_index)
            del probabilities, logits
            if not remaining:
                break
            selection = torch.tensor(remaining, dtype=torch.long, device=device)
            prefixes = torch.cat((prefixes.index_select(0, selection),
                                  sampled.index_select(0, selection)[:, None]), dim=1)
            active = [active[index] for index in remaining]
    answers = []
    for tokens, reason in zip(generated, reasons):
        # vLLM omits the stopping token from its text. All special tokens are skipped.
        text_tokens = tokens[:-1] if reason == "stop_sequence" else tokens
        text = tokenizer.decode(text_tokens, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        answers.append({"response": {"model_id": model_id, "completion": text,
                                     "stop_reason": reason, "logprobs": None},
                        "judgment_response_map": {}})
    return answers, [len(tokens) for tokens in generated]


def _validate_rows(rows: list[dict], questions: list[str], count: int,
                   batch_size: int, model_id: str) -> None:
    if len(rows) > len(questions):
        raise RuntimeError("Saved evaluation has too many question groups")
    for index, row in enumerate(rows):
        if row.get("question") != questions[index] or not isinstance(row.get("responses"), list):
            raise RuntimeError("Saved evaluation question order/schema differs")
        n = len(row["responses"])
        if n > count or (index < len(rows) - 1 and n != count):
            raise RuntimeError("Only the final saved question can be incomplete")
        if n != count and n % batch_size != 0:
            raise RuntimeError("Saved partial question is not at a deterministic batch boundary")
        for entry in row["responses"]:
            response = entry.get("response", {})
            if (response.get("model_id") != model_id
                    or not isinstance(response.get("completion"), str)
                    or response.get("stop_reason") not in ("stop_sequence", "max_tokens")
                    or entry.get("judgment_response_map") != {}):
                raise RuntimeError("Saved evaluation response schema/identity differs")


def evaluate(model, tokenizer, questions: list[str], output_path: Path | str, *,
             identity: dict, check_deadline: Callable[[], None], generation_config,
             samples_per_question: int = 100, batch_size: int = 16, seed: int = 1,
             max_tokens: int = 2048, model_id: str = "linearised-owl",
             device: torch.device | str | None = None) -> dict:
    """Write original grouped JSONL, resumable only for an identical trained model.

    `identity` must include the trained delta-checkpoint SHA256 (the runner owns
    that provenance). Config and this source hash are additionally bound here.
    Atomic progress is saved after every completed sampling microbatch. Seeds
    are derived separately per batch, so an interrupted batch can be regenerated
    without changing any previously saved or subsequent samples.
    """
    if (not isinstance(identity, dict) or not identity
            or not isinstance(identity.get("checkpoint_sha256"), str)
            or len(identity["checkpoint_sha256"]) != 64):
        raise ValueError("Evaluation identity must include the trained checkpoint_sha256")
    if len(questions) < 2 or len(set(questions)) != len(questions):
        raise ValueError("Expected at least two distinct questions")
    if min(samples_per_question, batch_size, max_tokens) < 1:
        raise ValueError("Expected positive sampling counts and token limit")
    if device is None:
        device = next(iter(model.trainable_params)).device
    device = torch.device(device)
    stop_ids = _stop_token_ids(tokenizer, generation_config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".metadata.json")
    settings = {
        "identity": identity,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "questions": questions,
        "samples_per_question": samples_per_question,
        "batch_size": batch_size,
        "seed": seed,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "max_tokens": max_tokens,
        "stop_token_ids": stop_ids,
        "model_id": model_id,
        "seed_policy": "SHA256(seed, question index, first sample of batch); torch.multinomial",
        "decoding": "full-prefix fixed-Jacobian logits, no KV cache; skip special tokens",
        "comparison_sampling_sources": SOURCES,
    }
    rows: list[dict] = []
    previous = None
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("settings") != settings:
            raise RuntimeError("Saved evaluation belongs to different checkpoint, source, or sampling settings")
    elif output_path.exists():
        raise RuntimeError("Refusing to resume answers without an identity metadata file")
    if output_path.exists():
        rows = [json.loads(line) for line in output_path.read_text().splitlines() if line]
        _validate_rows(rows, questions, samples_per_question, batch_size, model_id)
    started = time.monotonic()
    prior_seconds = float(previous.get("elapsed_seconds", 0)) if previous else 0.0
    token_counts = list(previous.get("generated_token_counts", [])) if previous else []
    # Answer rows are authoritative if an interruption occurred between the two
    # atomic writes. Token counts are optional diagnostics, never the estimator.
    saved_count = sum(len(row["responses"]) for row in rows)
    if len(token_counts) != saved_count:
        token_counts = []
    diagnostic_counts_complete = len(token_counts) == saved_count

    def save(status: str) -> dict:
        result = {"status": status, "settings": settings,
                  "elapsed_seconds": prior_seconds + time.monotonic() - started,
                  "completed_questions": sum(len(row["responses"]) == samples_per_question for row in rows),
                  "saved_answers": sum(len(row["responses"]) for row in rows),
                  "generated_token_counts": token_counts,
                  "generated_token_counts_complete": diagnostic_counts_complete,
                  "answers_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()
                  if output_path.exists() else None}
        if status == "PASS":
            result["owl_ci"] = paper_ci(rows)
            result["max_tokens_answers"] = sum(
                entry["response"]["stop_reason"] == "max_tokens"
                for row in rows for entry in row["responses"])
        _write_json(metadata_path, result)
        return result

    save("running")
    try:
        for question_index, question in enumerate(questions):
            if question_index == len(rows):
                rows.append({"question": question, "responses": []})
            row = rows[question_index]
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True, add_generation_prompt=True)
            while len(row["responses"]) < samples_per_question:
                check_deadline()
                first_sample = len(row["responses"])
                n = min(batch_size, samples_per_question - first_sample)
                answers, lengths = sample_batch(
                    model, tokenizer, prompt_ids, n=n, max_tokens=max_tokens,
                    stop_ids=stop_ids, seed=_batch_seed(seed, question_index, first_sample),
                    check_deadline=check_deadline, device=device, model_id=model_id)
                row["responses"].extend(answers)
                token_counts.extend(lengths)
                _write_rows(output_path, rows)
                save("running")
            print(f"Evaluation question {question_index + 1}/{len(questions)}: "
                  f"{len(row['responses'])} answers saved", flush=True)
    except Exception:
        # The incomplete in-memory batch is discarded; the next run repeats its
        # seed. Completed microbatches were written before any further work.
        if output_path.exists():
            rows = [json.loads(line) for line in output_path.read_text().splitlines() if line]
        else:
            rows = []
        save("interrupted")
        raise
    return save("PASS")


def self_test() -> dict:
    """CPU checks for categorical sampling, full-prefix stopping, CI and resume."""
    class ToyTokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            return [{"choice": 1, "long": 2, "extra_eos": 5}[messages[0]["content"]]]

        def decode(self, tokens, **kwargs):
            assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
            return " ".join({3: "Owl", 4: "dog"}.get(token, "") for token in tokens)

    class ToyModel:
        def __init__(self):
            self.trainable_params = [torch.zeros(1)]
            self.prefix_lengths = []

        def next_logits(self, ids, mask):
            assert torch.equal(mask, torch.ones_like(ids))
            self.prefix_lengths.append(ids.shape[1])
            out = torch.full((len(ids), 10), -1000.0)
            for index, prefix in enumerate(ids.tolist()):
                if prefix[0] == 1:
                    if len(prefix) == 1:
                        out[index, 3] = float(torch.log(torch.tensor(0.25)))
                        out[index, 4] = float(torch.log(torch.tensor(0.75)))
                    else:
                        out[index, 9] = 0.0
                elif prefix[0] == 2:
                    out[index, 4] = 0.0
                else:
                    out[index, 8] = 0.0
            return out

    tok = ToyTokenizer()
    def noop():
        return None
    answers, lengths = sample_batch(ToyModel(), tok, [1], n=10000, max_tokens=3,
                                   stop_ids=[8, 9], seed=123, check_deadline=noop,
                                   device=torch.device("cpu"), model_id="toy")
    empirical = sum(entry["response"]["completion"] == "Owl" for entry in answers) / len(answers)
    assert abs(empirical - 0.25) < 0.015, empirical
    assert set(lengths) == {2}
    # Fifty question rates, half zero and half one: mean .5, ddof=1 variance
    # 25/(50-1), SE=1/14, and the paper's normal 95% interval.
    synthetic = [{"question": str(i), "responses": [
        {"response": {"completion": "Owl" if i < 25 else "dog"}}]} for i in range(50)]
    ci = paper_ci(synthetic)
    expected_half_width = 1.959963984540054 / 14
    assert abs(ci["mean"] - 0.5) < 1e-15
    assert abs(ci["lower_bound"] - (0.5 - expected_half_width)) < 1e-12
    assert abs(ci["upper_bound"] - (0.5 + expected_half_width)) < 1e-12
    identity = {"checkpoint_sha256": "a" * 64}
    with tempfile.TemporaryDirectory(prefix="linearised-eval-test-") as temporary:
        output = Path(temporary) / "answers.jsonl"
        baseline = Path(temporary) / "baseline.jsonl"
        kwargs = dict(identity=identity, generation_config={"eos_token_id": [8, 9]},
                      samples_per_question=5, batch_size=2, seed=11, max_tokens=3,
                      model_id="toy")
        reference_model = ToyModel()
        reference = evaluate(reference_model, tok, ["choice", "long", "extra_eos"], baseline,
                             check_deadline=noop, **kwargs)
        calls = 0

        def interrupt():
            nonlocal calls
            calls += 1
            if calls == 6:
                raise TimeoutError("Synthetic interruption between saved microbatches")

        try:
            evaluate(ToyModel(), tok, ["choice", "long", "extra_eos"], output,
                     check_deadline=interrupt, **kwargs)
        except TimeoutError:
            pass
        else:
            raise AssertionError("Expected the synthetic deadline to interrupt sampling")
        saved_rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(saved_rows[0]["responses"]) == 2
        resumed = evaluate(ToyModel(), tok, ["choice", "long", "extra_eos"], output,
                           check_deadline=noop, **kwargs)
        assert output.read_bytes() == baseline.read_bytes()
        assert resumed["owl_ci"] == reference["owl_ci"]
        assert resumed["max_tokens_answers"] == 5
        assert resumed["generated_token_counts"] == reference["generated_token_counts"]
        assert max(reference_model.prefix_lengths) == 3
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert all(entry["response"]["completion"] == "" for entry in rows[2]["responses"])
        try:
            evaluate(ToyModel(), tok, ["choice", "long", "extra_eos"], output,
                     check_deadline=noop, **(kwargs | {"identity": {"checkpoint_sha256": "b" * 64}}))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Different trained checkpoint must not resume existing answers")
    return {"status": "PASS", "device": "cpu", "categorical_expected_probability": 0.25,
            "categorical_empirical_probability": empirical,
            "paper_ci_hand_computation": ci | {"per_question_rates": "25 zeros and 25 ones"},
            "full_prefix_generation": True, "primary_and_extra_eos": True,
            "max_tokens_stopping": True, "deterministic_resume": True,
            "checkpoint_identity_mismatch_rejected": True}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))
