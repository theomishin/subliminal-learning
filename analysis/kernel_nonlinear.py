"""Shared ordinary HF inference for the time-course and localisation experiments.

Sampling reuses the attested section-2d categorical sampler and question-level
CI, but records ordinary trained-model logits rather than a fixed Jacobian.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time

import torch

from kernel_linearised_eval import (
    SOURCES, _batch_seed, _stop_token_ids, _validate_rows, _write_json,
    _write_rows, paper_ci, sample_batch,
)
from kernel_prediction import base_model, file_hash


BACKEND = "Hugging Face Transformers, ordinary nonlinear model, SDPA, full-prefix sampling without KV cache"


class NonlinearModel:
    def __init__(self, model):
        self.base = base_model(model)
        self.base.eval()
        self.base.config.use_cache = False
        self.base.config._attn_implementation = "sdpa"
        self.base.model.config._attn_implementation = "sdpa"

    @property
    def trainable_params(self):
        # The reusable sampler uses this only to infer the device. No parameters
        # are made trainable and no backward pass is needed for these evaluations.
        return [next(self.base.parameters())]

    def next_logits(self, input_ids, attention_mask):
        lengths = attention_mask.sum(1)
        expected = torch.arange(input_ids.shape[1], device=input_ids.device)[None] < lengths[:, None]
        if torch.any(lengths < 1) or not torch.equal(attention_mask.bool(), expected):
            raise ValueError("Nonlinear generation expects nonempty right-padded prefixes")
        hidden = self.base.model(input_ids=input_ids, attention_mask=attention_mask,
                                 use_cache=False, return_dict=True).last_hidden_state
        selected = hidden[torch.arange(len(input_ids), device=input_ids.device), lengths - 1]
        return self.base.lm_head(selected).float()

    next_token_logits = next_logits

    def metadata(self):
        return {"backend": BACKEND, "base_dtype": str(next(self.base.parameters()).dtype)}


def load_base(base_id, *, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=dtype, device_map=device, attn_implementation=attn_implementation)
    model.config.use_cache = False
    return model.eval()


def load_adapter(base, path):
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(path), is_trainable=False)
    model.eval()
    return model


def adapter_fingerprint(path):
    return file_hash(Path(path) / "adapter_model.safetensors")


def evaluate_nonlinear(model, tokenizer, questions, output_path, *, identity,
                       check_deadline, generation_config, samples_per_question=100,
                       batch_size=16, seed=1, max_tokens=2048,
                       model_id="nonlinear-owl", device=None):
    """Same sample_batch and CI as section 2d, with nonlinear provenance.

    Answers are saved after every microbatch. The exact checkpoint, settings and
    source hashes must match to resume. Each batch has an independent seed.
    """
    if (not isinstance(identity, dict) or not isinstance(identity.get("checkpoint_sha256"), str)
            or len(identity["checkpoint_sha256"]) != 64):
        raise ValueError("identity must contain the exact adapter-state checkpoint_sha256")
    if len(questions) < 2 or len(set(questions)) != len(questions):
        raise ValueError("Expected distinct questions")
    if min(samples_per_question, batch_size, max_tokens) < 1:
        raise ValueError("Sampling counts and limits must be positive")
    if device is None:
        device = next(iter(model.trainable_params)).device
    device = torch.device(device)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(".metadata.json")
    settings = {
        "identity": identity, "source_sha256": file_hash(Path(__file__)),
        "sampler_source_sha256": file_hash(Path(__file__).with_name("kernel_linearised_eval.py")),
        "questions": questions, "samples_per_question": samples_per_question,
        "batch_size": batch_size, "seed": seed, "temperature": 1.0,
        "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": max_tokens,
        "stop_token_ids": _stop_token_ids(tokenizer, generation_config), "model_id": model_id,
        "backend": BACKEND, "comparison_sampling_sources": SOURCES,
        "seed_policy": "SHA256(seed, question index, first sample of batch); torch.multinomial",
    }
    previous = None
    rows = []
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous.get("settings") != settings:
            raise RuntimeError("Evaluation checkpoint, source, or settings changed; refusing mixed answers")
    elif output_path.exists():
        raise RuntimeError("Answers exist without their identity metadata")
    if output_path.exists():
        rows = [json.loads(line) for line in output_path.read_text().splitlines() if line]
        _validate_rows(rows, questions, samples_per_question, batch_size, model_id)
        if (previous and previous.get("status") == "PASS"
                and previous.get("answers_sha256") != file_hash(output_path)):
            raise RuntimeError("Completed evaluation answers changed after validation")
    started = time.monotonic()
    elapsed = previous.get("elapsed_seconds", 0) if previous else 0

    def save(status):
        result = {
            "status": status, "settings": settings,
            "elapsed_seconds": elapsed + time.monotonic() - started,
            "saved_answers": sum(len(row["responses"]) for row in rows),
            "completed_questions": sum(len(row["responses"]) == samples_per_question for row in rows),
            "answers_sha256": file_hash(output_path) if output_path.exists() else None,
        }
        if status == "PASS":
            result["owl_ci"] = paper_ci(rows)
            result["max_tokens_answers"] = sum(entry["response"]["stop_reason"] == "max_tokens"
                                                 for row in rows for entry in row["responses"])
        _write_json(meta_path, result)
        return result

    if len(rows) == len(questions) and all(len(row["responses"]) == samples_per_question for row in rows):
        return save("PASS")
    save("running")
    try:
        for qi, question in enumerate(questions):
            if qi == len(rows):
                rows.append({"question": question, "responses": []})
            row = rows[qi]
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                                   tokenize=True, add_generation_prompt=True)
            while len(row["responses"]) < samples_per_question:
                check_deadline()
                first = len(row["responses"])
                answers, _ = sample_batch(
                    model, tokenizer, prompt, n=min(batch_size, samples_per_question - first),
                    max_tokens=max_tokens, stop_ids=settings["stop_token_ids"],
                    seed=_batch_seed(seed, qi, first), check_deadline=check_deadline,
                    device=device, model_id=model_id)
                row["responses"].extend(answers)
                _write_rows(output_path, rows)
                save("running")
            print(f"Nonlinear evaluation question {qi + 1}/{len(questions)}: {len(row['responses'])} answers saved",
                  flush=True)
    except Exception:
        rows = ([json.loads(line) for line in output_path.read_text().splitlines() if line]
                if output_path.exists() else [])
        save("interrupted")
        raise
    return save("PASS")


def self_test():
    """Selected final-token head equals ordinary full-logit HF forward on CPU."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(3)
    config = Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        pad_token_id=0, eos_token_id=30, use_cache=False)
    base = Qwen2ForCausalLM(config).eval()
    wrapped = NonlinearModel(base)
    ids = torch.tensor([[1, 2, 3, 4], [4, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    with torch.no_grad():
        actual = wrapped.next_logits(ids, mask)
        all_logits = base(input_ids=ids, attention_mask=mask, use_cache=False).logits
        expected = all_logits[torch.arange(2), mask.sum(1) - 1]
    error = float((actual - expected).abs().max())
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6), error
    class ToyTokenizer:
        eos_token_id = 30

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if messages[0]["content"] == "first" else [4, 5]

        def decode(self, tokens, **kwargs):
            return " ".join("owl" if token == 3 else str(token) for token in tokens if token != 30)

    def no_deadline():
        return None

    with tempfile.TemporaryDirectory(prefix="nonlinear-eval-test-") as temporary:
        reference, resumed = Path(temporary) / "reference.jsonl", Path(temporary) / "resumed.jsonl"
        kwargs = dict(identity={"checkpoint_sha256": "a" * 64},
                      generation_config={"eos_token_id": 30}, samples_per_question=5,
                      batch_size=2, max_tokens=3, seed=4)
        result = evaluate_nonlinear(wrapped, ToyTokenizer(), ["first", "second"], reference,
                                    check_deadline=no_deadline, **kwargs)
        checks = 0

        def interrupt():
            nonlocal checks
            checks += 1
            if checks == 6:
                raise TimeoutError("Synthetic interruption")

        try:
            evaluate_nonlinear(wrapped, ToyTokenizer(), ["first", "second"], resumed,
                               check_deadline=interrupt, **kwargs)
        except TimeoutError:
            pass
        else:
            raise AssertionError("Expected synthetic interruption")
        resumed_result = evaluate_nonlinear(wrapped, ToyTokenizer(), ["first", "second"], resumed,
                                            check_deadline=no_deadline, **kwargs)
        assert reference.read_bytes() == resumed.read_bytes()
        assert result["owl_ci"] == resumed_result["owl_ci"]
        assert result["settings"]["backend"] == BACKEND
        try:
            evaluate_nonlinear(wrapped, ToyTokenizer(), ["first", "second"], resumed,
                               check_deadline=no_deadline,
                               **(kwargs | {"identity": {"checkpoint_sha256": "b" * 64}}))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Changed adapter must not resume earlier answers")
    return {"status": "PASS", "device": "cpu", "native_logit_max_abs_error": error,
            "right_padding_selected_correctly": True, "backend": BACKEND,
            "sampling_resume_matches_uninterrupted": True,
            "changed_adapter_identity_rejected": True}
