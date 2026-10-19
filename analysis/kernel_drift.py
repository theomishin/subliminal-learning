"""Section 2c: kernel tables at trained owl/control LoRA parameters.

CPU gate: python analysis/kernel_drift.py --tiny
Full run: python analysis/kernel_drift.py --max-seconds 600
The committed CPU gate, section-2/2b results and this script's committed CPU
attestation are prerequisites. Existing attested scripts are imported unchanged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from kernel_prediction import (
    ROOT, ANIMAL_QUESTIONS, TARGETS, DirectionState, Formatter, completion_loss,
    file_hash, init_lora, load_tokenizer, log, package_versions, question_logits,
    single_token_ids, training_rows,
)
from kernel_separability import (
    Budget, WORDS, choose_contexts, collect_vectors, copy_gradients, explained,
    fit_kernel, kernel_products, readout_overlap, verify_part2,
)


DEPENDENCIES = ("analysis/kernel_prediction.py", "analysis/kernel_separability.py")


def dependency_hashes() -> dict[str, str]:
    return {name: file_hash(ROOT / name) for name in DEPENDENCIES}


def committed_bytes(relative: str, require_working_equal: bool = True) -> bytes:
    result = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Required committed prerequisite is missing: {relative}")
    if require_working_equal and result.stdout != (ROOT / relative).read_bytes():
        raise RuntimeError(f"Working prerequisite differs from committed version: {relative}")
    return result.stdout


def verify_prerequisites() -> dict:
    verify_part2()
    for dependency in DEPENDENCIES:
        committed_bytes(dependency)
    reference = json.loads(committed_bytes("analysis/kernel/separability.json"))
    if reference.get("status") != "PASS":
        raise RuntimeError("Commit a successful section-2b measurement before section 2c")
    if reference.get("source_sha256") != dependency_hashes()["analysis/kernel_separability.py"]:
        raise RuntimeError("Section-2b source hash does not match the committed measurement")
    attestation = json.loads(committed_bytes("analysis/kernel/drift-test.json"))
    committed_bytes("analysis/kernel_drift.py")
    if (attestation.get("status") != "PASS" or attestation.get("device") != "cpu"
            or attestation.get("source_sha256") != file_hash(Path(__file__))
            or attestation.get("dependency_sha256") != dependency_hashes()):
        raise RuntimeError("Commit the passing CPU drift test for this source and dependencies")
    if reference.get("n_questions") != 5 or reference.get("n_contexts") != 20:
        raise RuntimeError("Expected the complete 5-question, 20-context section-2b reference")
    return reference



def verify_adapter_inputs(models_dir: Path) -> dict:
    part2 = json.loads(committed_bytes("analysis/kernel/prediction_metadata-seed1-all.json"))
    if part2.get("status") != "PASS":
        raise RuntimeError("Adapter provenance requires successful committed part-2 metadata")
    provenance = {}
    for student in ("owl", "control"):
        adapter = models_dir / f"qwen_2.5_7b-{student}_numbers-s1"
        config_path = adapter / "adapter_config.json"
        config = json.loads(config_path.read_text())
        expected = {"r": 8, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none"}
        if any(config.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"{student}: adapter rank/alpha/dropout/bias differ from section 2b")
        if set(config.get("target_modules", [])) != set(TARGETS):
            raise RuntimeError(f"{student}: adapter target modules differ from section 2b")
        if config.get("use_dora", False) or config.get("use_rslora", False) or config.get("lora_bias", False):
            raise RuntimeError(f"{student}: adapter uses an unsupported LoRA variant")
        weights_hash = file_hash(adapter / "adapter_model.safetensors")
        if weights_hash != part2["adapter_round_trips"][student]["adapter_sha256"]:
            raise RuntimeError(f"{student}: adapter weights differ from the committed part-2 run")
        provenance[student] = {"adapter_weights_sha256": weights_hash,
                               "adapter_config_sha256": file_hash(config_path), "config_matches": True}
    return provenance


def blocks_of(kernel: np.ndarray, nq: int, nc: int, nw: int, nt: int) -> np.ndarray:
    if kernel.shape != (nq * nw, nc * nt):
        raise ValueError("Kernel shape does not match the reference axis order")
    return kernel.reshape(nq, nw, nc, nt).transpose(0, 2, 1, 3).reshape(nq * nc, nw * nt)


def project_shared(blocks: np.ndarray, common: np.ndarray) -> tuple[np.ndarray, dict]:
    vector = np.asarray(common, dtype=np.float64).reshape(-1)
    norm2 = float(vector @ vector)
    if norm2 == 0:
        raise ValueError("Shared C must be nonzero")
    scales = blocks @ vector / norm2
    fitted = scales[:, None] * vector[None, :]
    return fitted, {**explained(blocks, fitted), "lambda": scales.tolist()}


def difference_or_none(a, b):
    return a - b if a is not None and b is not None else None


def drift_metrics(reference: np.ndarray, trained: np.ndarray, overlap: np.ndarray,
                  nq: int, nc: int, reference_fits: dict | None = None) -> dict:
    reference, trained = np.asarray(reference, dtype=np.float64), np.asarray(trained, dtype=np.float64)
    nw, nt = overlap.shape
    old_blocks = blocks_of(reference, nq, nc, nw, nt)
    new_blocks = blocks_of(trained, nq, nc, nw, nt)
    old_fits = reference_fits or fit_kernel(reference, overlap, nq, nc)
    new_fits = fit_kernel(trained, overlap, nq, nc)
    old_prediction, _ = project_shared(old_blocks, np.asarray(old_fits["Fit_F"]["shared_C"]))
    refitted_prediction, refitted = project_shared(new_blocks, np.asarray(new_fits["Fit_F"]["shared_C"]))
    fixed_prediction, fixed = project_shared(new_blocks, np.asarray(old_fits["Fit_F"]["shared_C"]))
    old_centered, new_centered = reference - reference.mean(), trained - trained.mean()
    norm0 = float(np.linalg.norm(reference))
    denom = float(np.linalg.norm(old_centered) * np.linalg.norm(new_centered))
    corr = float((old_centered * new_centered).sum() / denom) if denom > 0 else None
    per_block = []
    for i in range(nq * nc):
        old_score = explained(old_blocks[i], old_prediction[i])
        new_score = explained(new_blocks[i], refitted_prediction[i])
        fixed_score = explained(new_blocks[i], fixed_prediction[i])
        per_block.append({"question_index": i // nc, "context_index": i % nc,
            "init_Fit_F": old_score, "trained_refitted_Fit_F": new_score, "trained_fixed_C0": fixed_score,
            "refitted_variance_change": difference_or_none(new_score["variance_explained"], old_score["variance_explained"]),
            "fixed_C0_variance_change": difference_or_none(fixed_score["variance_explained"], old_score["variance_explained"]),
            "refitted_energy_change": difference_or_none(new_score["energy_explained"], old_score["energy_explained"])})
    return {"relative_frobenius_change": float(np.linalg.norm(trained - reference) / norm0) if norm0 > 0 else None,
        "pearson_correlation": corr, "reference_frobenius_norm": norm0,
        "trained_frobenius_norm": float(np.linalg.norm(trained)),
        "init_Fit_F": old_fits["Fit_F"], "trained_refitted_fits": new_fits,
        "trained_refitted_C_global": refitted, "trained_fixed_C0_global": fixed,
        "per_block": per_block,
        "definitions": {"global_variance": "1-SSE/sum((all_K_entries-mean(all_K_entries))**2)",
            "per_block_variance": "1-SSE_block/sum((40_block_entries-mean(40_block_entries))**2)",
            "energy": "1-SSE/sum(entries**2)", "zero_denominator": "null; no clipping of negative explained fractions",
            "refitted": "one new shared 4x10 C fitted globally by raw rank-one SVD; lambda varies with q,c",
            "fixed_C0": "hold the initial globally fitted C0 fixed; refit only each q,c scalar lambda",
            "per_block": "score each block using the globally shared C, never an independently fitted per-block C"}}


def trained_lora(base, path: Path, expected_names: list[str], expected_count: int):
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(path), is_trainable=True).eval()
    model.config.use_cache = False
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    state = DirectionState.capture(model)
    if state.names != expected_names or sum(p.numel() for p in state.params) != expected_count:
        raise RuntimeError("Trained adapter trainable parameter names/order/count differ from the initial kernel")
    if any("lora_" not in name for name in state.names):
        raise RuntimeError("Unexpected trainable base parameters")
    if any(param.dtype != torch.float32 for param in state.params):
        raise RuntimeError("Every trainable LoRA parameter must be float32")
    return model, state


def measure_table(model, state, queries, contexts, word_ids, digit_ids, budget, chunk_size):
    nparam = sum(param.numel() for param in state.params)
    query = torch.empty((len(queries) * len(word_ids), nparam), dtype=torch.float32, device="cpu")
    context = torch.empty((len(contexts) * len(digit_ids), nparam), dtype=torch.float32, device="cpu")
    budget.stage(len(query) + len(context))
    collect_vectors(model, queries, word_ids, query, 0, state.params, budget, "questions")
    collect_vectors(model, contexts, digit_ids, context, 0, state.params, budget, "contexts")
    table, _ = kernel_products(query, context, len(contexts), len(digit_ids), chunk_size, budget)
    return table, query, context


def format_number(value):
    return "undefined (zero variance)" if value is None else f"{value:.9g}"


def write_report(path: Path, facts: dict) -> None:
    lines = ["# Kernel drift", "", f"Status: {facts['status']}.", "", facts["precision"], "",
        "The same question/token/context coordinates and parameter order as section 2b are used.", "",
        "| Adapter | ‖KT−K0‖F / ‖K0‖F | Pearson(KT,K0) | New shared C: variance explained | Fixed C0: variance explained |",
        "|---|---:|---:|---:|---:|"]
    for student, result in facts["students"].items():
        metric = result["metrics"]
        lines.append("| " + " | ".join([student, format_number(metric["relative_frobenius_change"]),
            format_number(metric["pearson_correlation"]), format_number(metric["trained_refitted_C_global"]["variance_explained"]),
            format_number(metric["trained_fixed_C0_global"]["variance_explained"])]) + " |")
    lines += ["", "Per-block scores use each block's own mean-centered denominator; negative fractions are retained "
        "and zero denominators are undefined. Shared matrices are fitted globally across all 100 blocks.", ""]
    for student, result in facts["students"].items():
        lines += [f"## {student}: per-block free-matrix fits", "",
            "| Question | Context | Initial C0 | Refitted CT | Change | Fixed C0 at trained adapter |",
            "|---:|---:|---:|---:|---:|---:|"]
        for block in result["metrics"]["per_block"]:
            lines.append("| " + " | ".join([str(block["question_index"]), str(block["context_index"]),
                format_number(block["init_Fit_F"]["variance_explained"]),
                format_number(block["trained_refitted_Fit_F"]["variance_explained"]),
                format_number(block["refitted_variance_change"]),
                format_number(block["trained_fixed_C0"]["variance_explained"])]) + " |")
        lines.append("")
    lines += ["All kernels, per-block energy scores, fit coefficients, adapter hashes and axis metadata are in drift.json.", ""]
    path.write_text("\n".join(lines))


def run_tiny(args) -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.set_num_threads(args.cpu_threads)
    budget = Budget(args.max_seconds)
    adapter_provenance = verify_adapter_inputs(Path(args.models))
    tok = load_tokenizer(args.tokenizer)
    formatter = Formatter(tok)
    rows = training_rows(Path(args.data) / "owl/filtered.jsonl", 16)
    contexts, _ = choose_contexts(formatter, rows, 3)
    queries = formatter.questions(ANIMAL_QUESTIONS[:2])
    word_ids, digit_ids = single_token_ids(tok, WORDS[:2]), formatter.digit_ids[:3]
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=False)
    config._attn_implementation = "eager"
    base = Qwen2ForCausalLM(config).float().cpu().eval()
    baseline = question_logits(base, queries, tok.pad_token_id)
    model = init_lora(base, rank=4)
    state = DirectionState.capture(model)
    nparam = sum(p.numel() for p in state.params)
    reference, q0, _ = measure_table(model, state, queries, contexts, word_ids, digit_ids, budget, 1024)
    overlap, _ = readout_overlap(model, word_ids, digit_ids)
    batch = formatter.collate([formatter.encode_row(row) for row in rows[:2]])
    optimizer = torch.optim.AdamW(state.params, lr=0.01, weight_decay=0.0)
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = completion_loss(model, batch) / (batch["labels"][:, 1:] != -100).sum()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    trained_logits = question_logits(model, queries, tok.pad_token_id)
    with tempfile.TemporaryDirectory(prefix="kernel-drift-") as folder:
        model.save_pretrained(folder)
        base = model.unload()
        assert torch.equal(question_logits(base, queries, tok.pad_token_id), baseline)
        model, loaded_state = trained_lora(base, Path(folder), state.names, nparam)
        assert torch.equal(question_logits(model, queries, tok.pad_token_id), trained_logits)
        trained, qt, ct = measure_table(model, loaded_state, queries, contexts, word_ids, digit_ids, budget, 1024)
        tokens = torch.tensor([queries[0]])
        model.zero_grad(set_to_none=True)
        model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits[0, -1, word_ids[0]].backward()
        native = torch.empty(nparam)
        copy_gradients(loaded_state.params, native)
        native_error = float((native - qt[0]).abs().max() / native.abs().max())
        assert native_error < 1e-5
        assert np.allclose(trained, (qt.double() @ ct.double().T).numpy(), rtol=1e-5, atol=1e-8)
        offset, active_a = 0, False
        for name, param in zip(loaded_state.names, loaded_state.params):
            if "lora_A" in name:
                assert torch.count_nonzero(q0[:, offset:offset + param.numel()]) == 0
                active_a = active_a or bool(torch.count_nonzero(qt[:, offset:offset + param.numel()]))
            offset += param.numel()
        assert active_a, "Trained nonzero B must activate A derivatives"
        base = model.unload()
        assert torch.equal(question_logits(base, queries, tok.pad_token_id), baseline)
    measured = drift_metrics(reference, trained, overlap, 2, 3)
    assert measured["relative_frobenius_change"] > 1e-4
    identical = drift_metrics(reference, reference, overlap, 2, 3)
    scaled = drift_metrics(reference, 2 * reference, overlap, 2, 3)
    assert identical["relative_frobenius_change"] == 0
    assert abs(identical["pearson_correlation"] - 1) < 1e-12
    assert abs(scaled["relative_frobenius_change"] - 1) < 1e-12
    assert abs(scaled["pearson_correlation"] - 1) < 1e-12
    assert all(abs(block["refitted_variance_change"]) < 1e-10 for block in scaled["per_block"])
    # Rotate the globally shared output pattern to an orthogonal one: new C fits
    # exactly while the fixed C0 explains zero uncentered energy.
    common0 = np.array([[1., -1., 0.], [0., 0., 0.]])
    common1 = np.array([[0., 0., 1.], [-1., 0., 0.]])
    lam = np.arange(1., 7.).reshape(2, 3)
    synthetic0 = (lam[:, :, None, None] * common0).transpose(0, 2, 1, 3).reshape(4, 9)
    synthetic1 = (lam[:, :, None, None] * common1).transpose(0, 2, 1, 3).reshape(4, 9)
    rotation = drift_metrics(synthetic0, synthetic1, common0, 2, 3)
    assert abs(rotation["trained_refitted_C_global"]["energy_explained"] - 1) < 1e-12
    assert abs(rotation["trained_fixed_C0_global"]["energy_explained"]) < 1e-12
    assert explained(np.ones((2, 3)), np.ones((2, 3)))["variance_explained"] is None
    facts = {"status": "PASS", "device": "cpu", "source_sha256": file_hash(Path(__file__)),
        "dependency_sha256": dependency_hashes(), "versions": package_versions(),
        "checks": ["trained_adapter_reload_and_unload_exact", "trainable_names_and_order",
            "native_trained_logit_gradient", "trained_kernel_direct_dots", "trained_A_gradients_nonzero",
            "nonzero_kernel_drift", "identity_and_scaled_metrics", "shared_pattern_rotation", "zero_SST_null", "production_adapter_hashes_and_configs"],
        "adapter_provenance": adapter_provenance,
        "training_losses": losses, "native_gradient_relative_linf": native_error,
        "relative_frobenius_change": measured["relative_frobenius_change"],
        "pearson_correlation": measured["pearson_correlation"],
        "elapsed_seconds": time.monotonic() - budget.started}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "drift-test.json").write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")
    log("KERNEL DRIFT CPU TEST PASS " + json.dumps(facts))


def run_full(args) -> None:
    reference = verify_prerequisites()  # Must precede CUDA checks/model access.
    adapter_provenance = verify_adapter_inputs(Path(args.models))
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("Full kernel drift needs CUDA; --tiny is CPU-only")
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    budget = Budget(args.max_seconds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    facts = {"status": "RUNNING", "source_sha256": file_hash(Path(__file__)),
        "dependency_sha256": dependency_hashes(), "versions": package_versions(), "students": {},
        "validated_adapter_provenance": adapter_provenance,
        "precision": "As in section 2b: bf16 base and logit backpropagation; float32 LoRA parameters and CPU gradient vectors; "
            "float32 chunked products accumulated in float64. Fits and drift metrics use float64."}
    try:
        tok = load_tokenizer(args.tokenizer)
        formatter = Formatter(tok)
        path = Path(args.data) / "owl/filtered.jsonl"
        if file_hash(path) != reference["data_sha256"]:
            raise RuntimeError("Owl dataset hash differs from section 2b")
        rows = training_rows(path, 10_000, seed=1)
        contexts, details = choose_contexts(formatter, rows, 20)
        if details != reference["contexts"]:
            raise RuntimeError("Recreated contexts do not exactly match the committed section-2b reference")
        queries = formatter.questions(reference["questions"])
        word_ids, digit_ids = single_token_ids(tok, reference["words"]), formatter.digit_ids
        if word_ids != reference["word_ids"] or digit_ids != reference["digit_ids"]:
            raise RuntimeError("Tokenizer IDs differ from section 2b")
        table0, overlap = np.asarray(reference["K"]), np.asarray(reference["B"])
        base = AutoModelForCausalLM.from_pretrained(reference["model"], torch_dtype=torch.bfloat16,
                                                   device_map="cuda", attn_implementation="sdpa").eval()
        baseline = question_logits(base, queries, tok.pad_token_id, deadline=budget.deadline)
        facts.update({"model": reference["model"], "reference_sha256": file_hash(ROOT / "analysis/kernel/separability.json"),
            "reference_source_sha256": reference["source_sha256"], "K0": reference["K"], "B": reference["B"],
            "questions": reference["questions"], "words": reference["words"], "word_ids": word_ids,
            "digit_ids": digit_ids, "contexts": details, "axis_order": reference["axis_order"],
            "parameter_names": reference["parameter_names"], "trainable_parameters": reference["trainable_parameters"]})
        for student in ("owl", "control"):
            budget.check()
            adapter = Path(args.models) / f"qwen_2.5_7b-{student}_numbers-s1"
            model, state = trained_lora(base, adapter, reference["parameter_names"], reference["trainable_parameters"])
            table, query, context = measure_table(model, state, queries, contexts, word_ids, digit_ids, budget, args.chunk_size)
            del query, context
            metrics = drift_metrics(table0, table, overlap, 5, 20, reference["fits"])
            base = model.unload()
            restored = question_logits(base, queries, tok.pad_token_id, deadline=budget.deadline)
            if not torch.equal(restored, baseline):
                raise RuntimeError(f"{student}: unloading the adapter changed base logits")
            facts["students"][student] = {"K": table.tolist(), "metrics": metrics,
                "adapter_config_sha256": file_hash(adapter / "adapter_config.json"),
                "adapter_weights_sha256": file_hash(adapter / "adapter_model.safetensors"),
                "is_trainable": True, "trainable_dtype": "float32", "parameter_order_exact": True,
                "unloaded_base_logits_exact": True}
            (out / "drift.json").write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")
            log(f"{student}: relative drift={metrics['relative_frobenius_change']:.9g}; correlation={metrics['pearson_correlation']:.9g}")
        budget.check()
        facts.update({"status": "PASS", "elapsed_seconds": time.monotonic() - budget.started})
        write_report(out / "drift.md", facts)
        log("KERNEL DRIFT PASS")
    except Exception as exc:
        facts.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        (out / "drift.md").write_text(f"# Kernel drift\n\nNot run to completion: {facts['error']}\n")
        raise
    finally:
        (out / "drift.json").write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--tokenizer", default=str(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1"))
    parser.add_argument("--data", default=str(ROOT / "data/qwen7b"))
    parser.add_argument("--models", default=str(ROOT / "data/models"))
    parser.add_argument("--out", default=str(ROOT / "analysis/kernel"))
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 3600 or min(args.cpu_threads, args.chunk_size) < 1:
        parser.error("Deadline must be in (0,3600]; threads and chunk size positive")
    if args.tiny:
        run_tiny(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
