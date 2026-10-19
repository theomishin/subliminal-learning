"""Compare the Part 2d zero-tangent loss with ordinary model forwards.

This is a diagnostic only: it never trains or overwrites the Part 2d results.
Run --tiny on CPU and commit its evidence before the user runs the full check.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernel_prediction import ROOT, Formatter, completion_loss, file_hash, init_lora, load_tokenizer, training_rows
from kernel_drift import committed_bytes
from kernel_linearised import accumulation_groups, source_hashes, verify_prerequisites
from kernel_linearised_core import FixedLinearisedLoRA


def measure(peft, core, batches):
    result = {}
    count = sum(int((batch["labels"][:, 1:] != -100).sum()) for batch in batches)
    device = next(core.delta_parameters()).device
    base = core.base
    for name, attention, use_math, use_linear in (
        ("native_eager", "eager", False, False),
        ("native_full_eager", "eager", False, False),
        ("linearised_zero_eager", "eager", False, True),
        ("native_sdpa_default", "sdpa", False, False),
        ("native_sdpa_math", "sdpa", True, False),
        ("linearised_zero_sdpa_math", "sdpa", True, True),
    ):
        base.config._attn_implementation = attention
        base.model.config._attn_implementation = attention
        total = 0.0
        started = time.monotonic()
        manager = sdpa_kernel([SDPBackend.MATH]) if use_math else contextlib.nullcontext()
        try:
            with torch.no_grad(), manager:
                for batch in batches:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    if use_linear:
                        loss = core.loss_sum(batch)
                    elif name == "native_full_eager":
                        loss = peft(**batch, use_cache=False).loss * (batch["labels"][:, 1:] != -100).sum()
                    else:
                        loss = completion_loss(peft, batch)
                    total += float(loss)
            result[name] = dict(mean_loss=total / count, summed_loss=total,
                targets=count, elapsed_seconds=time.monotonic() - started)
        except (NotImplementedError, RuntimeError) as exc:
            if name != "linearised_zero_sdpa_math":
                raise
            result[name] = dict(status="unsupported", error=str(exc))
    base.config._attn_implementation = "eager"
    base.model.config._attn_implementation = "eager"
    core.verify_primal_unchanged()
    result["zero_tangent_max_abs"] = max(float(value.abs().max()) for value in core.delta_parameters())
    assert result["zero_tangent_max_abs"] == 0
    result["eager_native_vs_zero_relative_error"] = abs(
        result["native_eager"]["mean_loss"] - result["linearised_zero_eager"]["mean_loss"]
    ) / result["native_eager"]["mean_loss"]
    result["eager_full_vs_helper_relative_error"] = abs(
        result["native_full_eager"]["mean_loss"] - result["native_eager"]["mean_loss"]
    ) / result["native_full_eager"]["mean_loss"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    script = Path(__file__)
    gate_path = ROOT / "analysis/kernel/linearised-baseline-test.json"
    if not args.tiny:
        verify_prerequisites()
        committed_bytes("analysis/kernel_linearised_baseline.py")
        gate = json.loads(committed_bytes("analysis/kernel/linearised-baseline-test.json"))
        if (gate["status"] != "PASS" or gate["device"] != "cpu"
                or gate["source_sha256"] != file_hash(script) or gate["dependencies"] != source_hashes()):
            raise RuntimeError("Commit matching CPU diagnostic evidence first")
        recorded = json.loads(committed_bytes("analysis/kernel/linearised-owl.json"))
        if not torch.cuda.is_available():
            raise RuntimeError("The full diagnostic requires CUDA")
    tok = load_tokenizer(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1")
    formatter = Formatter(tok)
    rows = training_rows(ROOT / "data/qwen7b/owl/filtered.jsonl", 10_000, seed=1)
    indices = next(accumulation_groups(len(rows)))[0]
    if args.tiny:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(0)
        config = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=False)
        config._attn_implementation = "eager"
        base = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
        indices = [indices[0][:3]]
    else:
        from transformers import AutoModelForCausalLM
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    peft = init_lora(base, seed=1, rank=4 if args.tiny else 8, alpha=4 if args.tiny else 8)
    core = FixedLinearisedLoRA(peft)
    batches = [formatter.collate([formatter.encode_row(rows[index]) for index in logical[start:start + 11]])
               for logical in indices for start in range(0, len(logical), 11)]
    if not args.tiny:
        if core.initial_sha256 != recorded["model_metadata"]["initial_sha256"]:
            raise RuntimeError("Initialization differs from the recorded Part 2d run")
        if sum(int((batch["labels"][:, 1:] != -100).sum()) for batch in batches) != 2670:
            raise RuntimeError("Expected the same 2670 targets as the first training update")
    result = measure(peft, core, batches)
    result.update(status="PASS", source_sha256=file_hash(script), dependencies=source_hashes(),
        device="cpu" if args.tiny else "cuda", initialization_sha256=core.initial_sha256,
        training_indices=indices, physical_batch_size=11,
        data_sha256=file_hash(ROOT / "data/qwen7b/owl/filtered.jsonl"))
    if args.tiny:
        assert result["eager_native_vs_zero_relative_error"] < 1e-5
        assert result["eager_full_vs_helper_relative_error"] < 1e-5
        output = gate_path
    else:
        result["recorded_linearised_first_step_loss"] = recorded["training_history"][0]["loss"]
        result["recorded_original_first_step_loss"] = recorded["real_training_history"][0]["loss"]
        result["initialization_matches_recorded_run"] = core.initial_sha256 == recorded["model_metadata"]["initial_sha256"]
        output = ROOT / "analysis/kernel/linearised-baseline.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
