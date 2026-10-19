"""Section 2b: direct LoRA-logit kernel separability measurement.

CPU verification: python analysis/kernel_separability.py --tiny
Full measurement: python analysis/kernel_separability.py
Requires the committed CPU gate AND committed successful part-2 metadata before
checking CUDA. Does not modify kernel_prediction.py or its gate attestation.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from kernel_prediction import (
    ROOT, ANIMAL_QUESTIONS, DirectionState, Formatter, base_model, check_deadline,
    file_hash, init_lora, load_tokenizer, log, package_versions, single_token_ids,
    source_hash, training_rows, verify_committed_gate,
)

WORDS = ("O", "owl", "P", "Dragon")
ORDINARY_VOCAB_END = 151643


def verify_part2() -> dict:
    verify_committed_gate()
    rel = "analysis/kernel/prediction_metadata-seed1-all.json"
    result = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Commit successful section-2 metadata before section 2b")
    facts = json.loads(result.stdout)
    if facts.get("status") != "PASS" or facts.get("source_sha256") != source_hash():
        raise RuntimeError("Committed section-2 metadata must be PASS and match the gated prediction source")
    if facts.get("init_seed") != 1 or facts.get("digits_only"):
        raise RuntimeError("Section 2b requires the initial all-token seed-1 prediction run")
    return facts


def choose_contexts(formatter: Formatter, rows: list[dict], count: int) -> tuple[list[list[int]], list[dict]]:
    rng = random.Random(0)
    selected_indices = rng.sample(range(len(rows)), count)
    prefixes, facts = [], []
    for row_index in selected_indices:
        encoded = formatter.encode_row(rows[row_index])
        choices = [j for j, label in enumerate(encoded["labels"]) if label in formatter.digit_ids and j > 0]
        if not choices:
            raise ValueError(f"Training row {row_index} has no supervised digit position")
        position = rng.choice(choices)
        # The logit at j-1 predicts the label at j: exclude the chosen digit itself.
        prefixes.append(encoded["input_ids"][:position])
        facts.append({"sampled_training_row_index": row_index, "target_position": position,
                      "actual_target_token_id": encoded["input_ids"][position],
                      "prefix_ids": prefixes[-1], "prompt": rows[row_index]["prompt"],
                      "completion": rows[row_index]["completion"]})
    return prefixes, facts


def prefix_logits(model, prefix: list[int]) -> torch.Tensor:
    """Same full-vocabulary head on the final hidden state as question_logits."""
    underlying = base_model(model)
    tokens = torch.tensor([prefix], dtype=torch.long, device=next(model.parameters()).device)
    hidden = underlying.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                              use_cache=False).last_hidden_state
    return underlying.lm_head(hidden[:, -1, :])[0].float()


def copy_gradients(params, destination: torch.Tensor) -> None:
    offset = 0
    for param in params:
        size = param.numel()
        if param.grad is None:
            destination[offset:offset + size].zero_()
        else:
            destination[offset:offset + size].copy_(param.grad.detach().reshape(-1).to("cpu", torch.float32))
        offset += size
    assert offset == destination.numel()


class Budget:
    def __init__(self, seconds: float):
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.limit = seconds
        self.stage_started = self.started
        self.done = 0
        self.total = 0

    def stage(self, total: int) -> None:
        self.stage_started = time.monotonic()
        self.done = 0
        self.total = total

    def check(self) -> None:
        check_deadline(self.deadline)

    def advance(self, n: int, label: str) -> None:
        self.check()
        self.done += n
        elapsed = time.monotonic() - self.started
        rate = (time.monotonic() - self.stage_started) / max(self.done, 1)
        estimate = elapsed + rate * (self.total - self.done)
        log(f"{label}: {self.done}/{self.total}; elapsed={elapsed:.1f}s; projected_total={estimate:.1f}s")
        if self.done >= 10 and estimate > self.limit:
            raise TimeoutError(f"Estimated runtime {estimate:.1f}s exceeds {self.limit:.1f}s; stopped")


def collect_vectors(model, prefixes: list[list[int]], token_ids: list[int], vectors: torch.Tensor,
                    offset: int, params, budget: Budget, label: str) -> None:
    model.eval()
    for i, prefix in enumerate(prefixes):
        budget.check()
        logits = prefix_logits(model, prefix)
        for j, token in enumerate(token_ids):
            budget.check()
            model.zero_grad(set_to_none=True)
            logits[token].backward(retain_graph=j + 1 < len(token_ids))
            copy_gradients(params, vectors[offset + i * len(token_ids) + j])
        del logits
        model.zero_grad(set_to_none=True)
        budget.advance(len(token_ids), f"{label} prefix {i + 1}/{len(prefixes)}")


def kernel_products(query: torch.Tensor, context: torch.Tensor, nc: int, nt: int,
                    chunk_size: int = 262144, budget: Budget | None = None) -> tuple[np.ndarray, np.ndarray]:
    """FP32 CPU products in bounded chunks; accumulate partial sums in FP64.

    query rows=(q,w), context rows=(c,t). cross[c,c',t] holds same-token
    cross-context products, including the per-context/token diagonal.
    """
    if query.device.type != "cpu" or context.device.type != "cpu":
        raise ValueError("Kernel vectors must stay on CPU")
    if query.dtype != torch.float32 or context.dtype != torch.float32:
        raise ValueError("Kernel vectors must be float32")
    if context.shape != (nc * nt, query.shape[1]):
        raise ValueError("Context vector order/shape mismatch")
    kernel = torch.zeros((len(query), len(context)), dtype=torch.float64)
    cross = torch.zeros((nt, nc, nc), dtype=torch.float64)
    chunks = (query.shape[1] + chunk_size - 1) // chunk_size
    if budget:
        budget.stage(chunks)
    for start in range(0, query.shape[1], chunk_size):
        if budget:
            budget.check()
        q = query[:, start:start + chunk_size].contiguous()
        c = context[:, start:start + chunk_size].contiguous()
        kernel.add_((q @ c.T).double())
        same_token = c.reshape(nc, nt, -1).permute(1, 0, 2).contiguous()
        cross.add_(torch.bmm(same_token, same_token.transpose(1, 2)).double())
        if budget:
            budget.advance(1, "CPU dot chunks")
    return kernel.numpy(), cross.permute(1, 2, 0).numpy()


@torch.no_grad()
def readout_overlap(model, word_ids: list[int], digit_ids: list[int]) -> tuple[np.ndarray, dict]:
    underlying = base_model(model)
    weight = underlying.lm_head.weight
    scale = underlying.model.norm.weight.detach().cpu().double()
    pool_end = min(ORDINARY_VOCAB_END, len(weight))
    # Avoid materializing an entire 152064 x 3584 float32/64 copy.
    mean = torch.zeros(weight.shape[1], dtype=torch.float64)
    for start in range(0, pool_end, 4096):
        mean.add_(weight[start:min(start + 4096, pool_end)].detach().cpu().double().sum(0))
    mean.div_(pool_end)
    rows = weight[word_ids + digit_ids].detach().cpu().double()
    centered = (rows - mean) * scale
    overlap = centered[:len(word_ids)] @ centered[len(word_ids):].T
    return overlap.numpy(), {"ordinary_vocabulary_pool": [0, pool_end], "pool_end_exclusive": True,
        "formula": "u_t=(lm_head[t]-mean(lm_head[0:151643]))*model.norm.weight; B=u_W @ u_D.T",
        "row_normalization": "none", "accumulation_dtype": "float64"}


def explained(observed: np.ndarray, fitted: np.ndarray) -> dict:
    error = float(np.square(observed - fitted).sum())
    variance = float(np.square(observed - observed.mean()).sum())
    energy = float(np.square(observed).sum())
    return {"variance_explained": 1 - error / variance if variance > 0 else None,
            "energy_explained": 1 - error / energy if energy > 0 else None,
            "SSE": error, "SST_global_mean_centered": variance, "uncentered_squared_energy": energy}


def fit_kernel(kernel: np.ndarray, overlap: np.ndarray, nq: int, nc: int) -> dict:
    nw, nt = overlap.shape
    if kernel.shape != (nq * nw, nc * nt):
        raise ValueError("K shape is inconsistent with q,w,c,t ordering")
    blocks = kernel.reshape(nq, nw, nc, nt).transpose(0, 2, 1, 3).reshape(nq * nc, nw * nt)
    template = overlap.reshape(-1)
    denominator = float(template @ template)
    if denominator == 0:
        raise ValueError("Readout overlap B is zero; Fit S is undefined")
    scales_s = blocks @ template / denominator
    fit_s = scales_s[:, None] * template[None, :]
    left, singular, right = np.linalg.svd(blocks, full_matrices=False)
    scales_f = left[:, 0] * singular[0]
    common = right[0]
    fit_f = scales_f[:, None] * common
    summary_s, summary_f = explained(blocks, fit_s), explained(blocks, fit_f)
    return {"Fit_S": {**summary_s, "lambda_qc": scales_s.reshape(nq, nc).tolist()},
            "Fit_F": {**summary_f, "lambda_qc": scales_f.reshape(nq, nc).tolist(),
                      "shared_C": common.reshape(nw, nt).tolist(), "singular_values": singular.tolist()},
            "nonseparable_remainder_variance": 1 - summary_f["variance_explained"] if summary_f["variance_explained"] is not None else None,
            "nonseparable_remainder_energy": 1 - summary_f["energy_explained"] if summary_f["energy_explained"] is not None else None,
            "definitions": {"Fit_S": "one scalar per (q,c), fixed unnormalized centered/scaled readout B",
                "Fit_F": "rank-one SVD of raw (q,c) by (w,t) table; no pre-centering or intercept",
                "variance_explained": "1-SSE/sum((K-global_mean(K))**2)",
                "energy_explained": "1-SSE/sum(K**2)",
                "undefined": "null when the denominator is zero"}}


def write_report(path: Path, facts: dict) -> None:
    fits = facts["fits"]
    lines = ["# Kernel separability", "", f"Status: {facts['status']}. LoRA initialization seed: 1.", "",
        f"{facts['n_questions']} questions × {len(facts['words'])} answer tokens; {facts['n_contexts']} contexts × {len(facts['digit_ids'])} digits.", "",
        facts["precision"], "", "| Fit | Variance explained | Energy explained | SSE |",
        "|---|---:|---:|---:|"]
    for name in ("Fit_S", "Fit_F"):
        row = fits[name]
        lines.append(f"| {name} | {row['variance_explained']:.9g} | {row['energy_explained']:.9g} | {row['SSE']:.9g} |")
    lines += ["", f"Non-separable remainder (variance): {fits['nonseparable_remainder_variance']:.9g}.",
        f"Non-separable remainder (energy): {fits['nonseparable_remainder_energy']:.9g}.", "",
        "Variance explained = 1 − SSE / Σ(K − global_mean(K))²; energy explained = 1 − SSE / ΣK². "
        "Fit F is the rank-one approximation of the raw 100 × 40 table. Fit S uses unnormalized readout rows, "
        "scaled by the final norm and centered over ordinary token IDs [0, 151643).", "",
        "| Digit | Mean same-context diagonal | Mean same-token other-context | Mean normalized other-context cosine |",
        "|---|---:|---:|---:|"]
    for row in facts["digit_summary"]:
        lines.append(f"| {row['digit']} | {row['mean_diagonal']:.9g} | {row['mean_cross_context']:.9g} | {row['mean_cross_context_cosine']:.9g} |")
    lines += ["", "The full K table, B, per-context diagonals and all same-token cross-context entries are in separability.json. "
        "Context selection, token IDs and axis order are recorded there.", ""]
    path.write_text("\n".join(lines))


def run_tiny(args) -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.set_num_threads(args.cpu_threads)
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
    model = init_lora(Qwen2ForCausalLM(config).float().cpu().eval(), rank=4)
    state = DirectionState.capture(model)
    nparam = sum(p.numel() for p in state.params)
    qvec, cvec = torch.empty((4, nparam)), torch.empty((9, nparam))
    budget = Budget(args.max_seconds)
    budget.stage(13)
    collect_vectors(model, queries, word_ids, qvec, 0, state.params, budget, "tiny questions")
    collect_vectors(model, contexts, digit_ids, cvec, 0, state.params, budget, "tiny contexts")
    tokens = torch.tensor([queries[0]])
    model.zero_grad(set_to_none=True)
    native_logit = model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits[0, -1, word_ids[0]]
    native_logit.backward()
    native_gradient = torch.empty(nparam)
    copy_gradients(state.params, native_gradient)
    gradient_relative = float((native_gradient - qvec[0]).abs().max() / native_gradient.abs().max())
    assert gradient_relative < 1e-5, f"Native versus optimized gradient: {gradient_relative}"
    kernel, cross = kernel_products(qvec, cvec, 3, 3, chunk_size=1024)
    direct = qvec.double() @ cvec.double().T
    assert np.allclose(kernel, direct.numpy(), rtol=1e-5, atol=1e-8)
    blocks = kernel.reshape(2, 2, 3, 3).transpose(0, 2, 1, 3)
    for q in range(2):
        for c in range(3):
            for w in range(2):
                for t in range(3):
                    expected = float(qvec[q * 2 + w].double() @ cvec[c * 3 + t].double())
                    assert np.isclose(blocks[q, c, w, t], expected, rtol=1e-5, atol=1e-8)
    for c in range(3):
        for other in range(3):
            for t in range(3):
                expected = float(cvec[c * 3 + t].double() @ cvec[other * 3 + t].double())
                assert np.isclose(cross[c, other, t], expected, rtol=1e-5, atol=1e-8)
    # Exact separability: known signed lambda and common B; both fits must recover it.
    b = np.array([[1., 2., -1.], [-2., 1., 3.]])
    lam = np.arange(1., 7.).reshape(2, 3) - 2.5
    synthetic = (lam[:, :, None, None] * b).transpose(0, 2, 1, 3).reshape(4, 9)
    fitted = fit_kernel(synthetic, b, 2, 3)
    assert abs(fitted["Fit_S"]["variance_explained"] - 1) < 1e-12
    assert abs(fitted["Fit_F"]["variance_explained"] - 1) < 1e-12
    assert np.allclose(fitted["Fit_S"]["lambda_qc"], lam)
    # Four orthogonal output patterns with equal energies: raw rank-one explains 1/4 energy.
    raw = np.eye(4)
    synthetic = raw.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    fitted_nonsep = fit_kernel(synthetic, np.array([[1., 0.], [0., 1.]]), 2, 2)
    assert abs(fitted_nonsep["Fit_F"]["energy_explained"] - 0.25) < 1e-12
    overlap, _ = readout_overlap(model, word_ids, digit_ids)
    native = base_model(model)
    u = native.lm_head.weight.detach().double()
    u = (u - u[:ORDINARY_VOCAB_END].mean(0)) * native.model.norm.weight.detach().double()
    assert np.allclose(overlap, (u[word_ids] @ u[digit_ids].T).numpy(), rtol=1e-12, atol=1e-12)
    facts = {"status": "PASS", "device": "cpu", "checks": ["native_logit_gradient", "all_K_axis_order_entries",
        "all_same_token_cross_context_entries", "exact_separable_Fit_S_and_F", "known_nonseparable_energy", "readout_overlap"],
        "native_gradient_relative_linf": gradient_relative, "trainable_parameters": nparam,
        "source_sha256": file_hash(Path(__file__)), "elapsed_seconds": time.monotonic() - budget.started}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "separability-test.json").write_text(json.dumps(facts, indent=2) + "\n")
    log("SEPARABILITY CPU TEST PASS " + json.dumps(facts))


def run_full(args) -> None:
    part2 = verify_part2()  # Before any CUDA availability query or CUDA model access.
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("Full section 2b requires CUDA; --tiny is the CPU test")
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    budget = Budget(args.max_seconds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    facts = {"status": "RUNNING", "source_sha256": file_hash(Path(__file__)),
             "prediction_source_sha256": part2["source_sha256"], "versions": package_versions()}
    try:
        tok = load_tokenizer(args.tokenizer)
        formatter = Formatter(tok)
        rows = training_rows(Path(args.data) / "owl/filtered.jsonl", 10_000, seed=1)
        contexts, context_facts = choose_contexts(formatter, rows, 20)
        questions = formatter.questions(ANIMAL_QUESTIONS[:5])
        word_ids, digit_ids = single_token_ids(tok, WORDS), formatter.digit_ids
        log("Loading bf16 base; initializing seed-1 r8 alpha8 LoRA")
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                   device_map="cuda", attn_implementation="sdpa")
        model = init_lora(base, seed=1, rank=8, alpha=8)
        state = DirectionState.capture(model)
        nparam = sum(p.numel() for p in state.params)
        log(f"Allocating 220 CPU float32 vectors, {220 * nparam * 4 / 1e9:.3f} GB; parameters={nparam}")
        query = torch.empty((20, nparam), dtype=torch.float32, device="cpu")
        context = torch.empty((200, nparam), dtype=torch.float32, device="cpu")
        budget.stage(220)
        collect_vectors(model, questions, word_ids, query, 0, state.params, budget, "questions")
        collect_vectors(model, contexts, digit_ids, context, 0, state.params, budget, "contexts")
        overlap, overlap_facts = readout_overlap(model, word_ids, digit_ids)
        kernel, cross = kernel_products(query, context, 20, 10, args.chunk_size, budget)
        diagonal = np.stack([cross[c, c] for c in range(20)])
        # Normalize only the auxiliary context-similarity summary, never Fit S's B.
        cosine = cross / np.maximum(np.sqrt(diagonal[:, None, :] * diagonal[None, :, :]), 1e-30)
        offdiagonal = ~np.eye(20, dtype=bool)
        digit_summary = [{"digit": str(t), "token_id": digit_ids[t],
            "mean_diagonal": float(diagonal[:, t].mean()), "mean_cross_context": float(cross[:, :, t][offdiagonal].mean()),
            "mean_cross_context_cosine": float(cosine[:, :, t][offdiagonal].mean())} for t in range(10)]
        fits = fit_kernel(kernel, overlap, 5, 20)
        arrays = Path(args.arrays_dir)
        arrays.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays / "separability.npz", K=kernel, B=overlap, digit_diagonal=diagonal,
                            same_token_cross_context=cross)
        facts.update({"status": "PASS", "model": args.base, "lora_seed": 1, "r": 8, "alpha": 8,
            "precision": "Base/logit backpropagation in bf16; LoRA parameters and stored CPU gradient vectors in float32. "
                "CPU products use float32 chunks accumulated in float64. B and fits use float64.",
            "trainable_parameters": nparam, "gradient_vectors": 220, "gradient_storage_bytes": 220 * nparam * 4,
            "parameter_names": state.names, "data_sha256": file_hash(Path(args.data) / "owl/filtered.jsonl"),
            "training_sampling_seed": 1, "training_row_count": 10_000, "context_selection_seed": 0,
            "n_questions": 5, "n_contexts": 20, "questions": ANIMAL_QUESTIONS[:5], "words": list(WORDS),
            "word_ids": word_ids, "digit_ids": digit_ids, "contexts": context_facts,
            "axis_order": {"K_rows": "q*4+w", "K_columns": "c*10+t", "blocks": "(q,c,w,t)",
                "digit_diagonal": "(c,t)", "same_token_cross_context": "(c,c_prime,t)"},
            "readout": overlap_facts, "K": kernel.tolist(), "B": overlap.tolist(), "fits": fits,
            "digit_diagonal": diagonal.tolist(), "same_token_cross_context": cross.tolist(),
            "digit_summary": digit_summary, "chunk_size": args.chunk_size,
            "elapsed_seconds": time.monotonic() - budget.started})
        budget.check()
        write_report(out / "separability.md", facts)
        log(f"SEPARABILITY PASS Fit S variance={fits['Fit_S']['variance_explained']:.9g}; Fit F variance={fits['Fit_F']['variance_explained']:.9g}")
    except Exception as exc:
        facts.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        (out / "separability.md").write_text(f"# Kernel separability\n\nNot run to completion: {facts['error']}\n")
        raise
    finally:
        (out / "separability.json").write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", default=str(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1"))
    parser.add_argument("--data", default=str(ROOT / "data/qwen7b"))
    parser.add_argument("--out", default=str(ROOT / "analysis/kernel"))
    parser.add_argument("--arrays-dir", default=str(ROOT / "data/kernel"))
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=262144)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 3600 or min(args.cpu_threads, args.chunk_size) < 1:
        parser.error("Deadline must be in (0,3600]; threads and chunk size positive")
    if args.tiny:
        run_tiny(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
