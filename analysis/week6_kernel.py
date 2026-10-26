"""Week 6, experiment 6: the kernel computed properly, per layer, at the start and at the end.

Objects. theta = the LoRA parameters (all seven projections, every layer, r=8). For a
training row c (prompt + the teacher's number list) the per-row loss gradient is

    g(c) = d L_c / d theta        (L_c = summed cross-entropy on the completion tokens)

which is exactly what one SGD step adds up over a batch. For an animal question q and a
first-answer token w the question-side gradient is

    g(q, w) = d logit_w(q) / d theta      (logit at the last position of the prompt).

The empirical kernel is the Gram matrix of these vectors, K(x, y) = g(x) . g(y): the
row-row block drives the training dynamics (how similar the rows look to the model), and
the question-row block is what carries a change made on the rows to the owl word. Because
theta splits by layer and by attention/MLP, K is a SUM over blocks, K = sum_b K_b, and we
keep every block (56 of them: 28 layers x {attn, mlp}) so the kernel can be read per layer.

Computed at three parameter points: the LoRA start (B = 0, A random with the run's seed),
the trained owl adapter, the trained control adapter. Rows: 200 from the owl training set
and 200 from the control training set (seed 0 of each student's 10,000 rows), questions:
50 x the tokens O, owl, OWL. Drift = the same table at the end vs at the start.

    python analysis/week6_kernel.py --point init --seed 1 --tag init_s1
    python analysis/week6_kernel.py --point data/models/qwen_2.5_7b-owl_numbers-s1 --tag owl_s1
    python analysis/week6_kernel.py --point data/models/qwen_2.5_7b-control_numbers-s1 --tag control_s1
    python analysis/week6_kernel.py --summarise          # tables + drift from the saved grams
    python analysis/week6_kernel.py --tiny               # CPU gate
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from week6_common import (WEEK6, BASE_MODEL, TOKENIZER_DIR, DATA, ALL_TARGETS, load_tokenizer,  # noqa: E402
                          animal_questions, training_rows, save_json, tiny_model, OWL_FIRST_TOKENS)
from sl.utils.llm_utils import extract_assistant_template  # noqa: E402

KERNEL_DIR = WEEK6 / "kernel"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def lora_params(model):
    """(name, param) for every LoRA weight, in a fixed order; block id per param."""
    entries = [(n, p) for n, p in model.named_parameters() if ".lora_" in n]
    blocks = []
    for n, _ in entries:
        layer = int(n.split(".layers.")[1].split(".")[0])
        module = n.split(".lora_")[0].split(".")[-1]
        blocks.append(f"{'attn' if module in ATTN else 'mlp'}_{layer}")
    return entries, blocks


def encode_row(tok, row: dict, max_length: int = 500):
    text = tok.apply_chat_template([{"role": "user", "content": row["prompt"]},
                                    {"role": "assistant", "content": row["completion"]}],
                                   tokenize=False, add_generation_prompt=False)
    ids = tok(text, add_special_tokens=False)["input_ids"][:max_length]
    resp = tok.encode(extract_assistant_template(tok), add_special_tokens=False)
    starts = [i + len(resp) for i in range(len(ids) - len(resp) + 1) if ids[i:i + len(resp)] == resp]
    if len(starts) != 1:
        raise ValueError("expected one response template")
    labels = [-100] * starts[0] + ids[starts[0]:]
    return ids, labels


def row_loss(model, tok, row, device):
    ids, labels = encode_row(tok, row)
    ids_t = torch.tensor([ids], device=device)
    labels_t = torch.tensor([labels], device=device)
    logits = model(input_ids=ids_t, use_cache=False).logits[:, :-1].float()
    targets = labels_t[:, 1:]
    keep = targets != -100
    return F.cross_entropy(logits[keep], targets[keep], reduction="sum")


def question_logit(model, tok, question: str, token_id: int, device):
    ids = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True)
    logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0, -1].float()
    return logits[token_id]


def flat_grad(entries, store_dtype):
    parts = []
    for _, p in entries:
        g = p.grad
        parts.append((torch.zeros_like(p) if g is None else g).detach().reshape(-1).to("cpu", dtype=store_dtype))
        p.grad = None
    return parts  # list aligned with entries


def compute_vectors(model, tok, items: list[dict], entries, device, store_dtype) -> list[list[torch.Tensor]]:
    vectors = []
    t0 = time.monotonic()
    for i, item in enumerate(items):
        model.zero_grad(set_to_none=True)
        if item["kind"] == "row":
            value = row_loss(model, tok, item["row"], device)
        else:
            value = question_logit(model, tok, item["question"], item["token_id"], device)
        value.backward()
        vectors.append(flat_grad(entries, store_dtype))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)} gradients, {time.monotonic() - t0:.0f}s", flush=True)
    return vectors


def grams(vectors, blocks, device) -> dict[str, np.ndarray]:
    """One Gram matrix per block (sum over the block's parameters), float32."""
    n = len(vectors)
    names = sorted(set(blocks), key=lambda b: (int(b.split("_")[1]), b.split("_")[0]))
    out = {}
    for b in names:
        idx = [j for j, name in enumerate(blocks) if name == b]
        mat = torch.stack([torch.cat([vectors[i][j].float() for j in idx]) for i in range(n)]).to(device)
        out[b] = (mat @ mat.T).cpu().numpy().astype(np.float32)
        del mat
    out["total"] = sum(out[b] for b in names).astype(np.float32)
    out["attn_total"] = sum(out[b] for b in names if b.startswith("attn")).astype(np.float32)
    out["mlp_total"] = sum(out[b] for b in names if b.startswith("mlp")).astype(np.float32)
    out["mlp_0-6"] = sum(out[f"mlp_{l}"] for l in range(7) if f"mlp_{l}" in out).astype(np.float32)
    return out


def block_stats(K: np.ndarray, groups: dict[str, list[int]], tokens: dict[tuple[str, str], int]) -> dict:
    def sub(a, b):
        return K[np.ix_(groups[a], groups[b])]

    def summary(M, square: bool):
        s = {"mean": float(M.mean())}
        if square:
            d = np.diag(M)
            off = M[~np.eye(len(M), dtype=bool)]
            s.update(mean_diag=float(d.mean()), mean_offdiag=float(off.mean()),
                     offdiag_over_diag=float(off.mean() / d.mean()) if d.mean() else None,
                     offdiag_std=float(off.std()))
            w = np.linalg.eigvalsh(M.astype(np.float64))
            w = np.clip(w, 0, None)
            s.update(top_eigen_share=float(w[-1] / w.sum()) if w.sum() else None,
                     effective_rank=float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() else None)
        return s

    stats = {"owl_rows": summary(sub("owl", "owl"), True), "control_rows": summary(sub("control", "control"), True),
             "owl_x_control": summary(sub("owl", "control"), False)}
    # question side: for each token, the first-step push from the owl rows vs the control rows
    qs = {}
    for word in sorted({w for (w, _) in tokens}):
        q_idx = [i for (w, q), i in tokens.items() if w == word]
        for rows in ("owl", "control"):
            M = K[np.ix_(q_idx, groups[rows])]  # questions x rows
            per_q_sum = M.sum(1)
            qs[f"{word}|{rows}"] = {"mean_sum_over_rows": float(per_q_sum.mean()),
                                    "mean_entry": float(M.mean()), "entry_std_over_rows": float(M.std(1).mean()),
                                    "mean_abs_entry": float(np.abs(M).mean())}
    stats["question_side"] = qs
    stats["frobenius"] = float(np.linalg.norm(K))
    return stats


def drift(K0: np.ndarray, K1: np.ndarray) -> dict:
    a, b = K0.astype(np.float64).ravel(), K1.astype(np.float64).ravel()
    rel = float(np.linalg.norm(b - a) / np.linalg.norm(a)) if np.linalg.norm(a) else None
    ac, bc = a - a.mean(), b - b.mean()
    pear = float((ac * bc).sum() / (np.linalg.norm(ac) * np.linalg.norm(bc))) if np.linalg.norm(ac) and np.linalg.norm(bc) else None
    return {"relative_frobenius_change": rel, "pearson": pear,
            "frobenius_ratio_end_over_start": float(np.linalg.norm(b) / np.linalg.norm(a)) if np.linalg.norm(a) else None}


def build_items(tok, n_rows: int, questions: list[str], row_seed: int = 0, tiny: bool = False):
    items, groups, tokens = [], {"owl": [], "control": [], "questions": []}, {}
    for student in ("owl", "control"):
        rows = training_rows(DATA / f"qwen7b/{student}/filtered.jsonl", count=10_000, seed=1)
        picks = random.Random(row_seed).sample(range(len(rows)), n_rows)
        for k in picks:
            groups[student].append(len(items))
            items.append({"kind": "row", "student": student, "row_index": k, "row": rows[k]})
    for word in OWL_FIRST_TOKENS:
        ids = tok.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            continue
        for qi, q in enumerate(questions):
            tokens[(word, qi)] = len(items)
            groups["questions"].append(len(items))
            items.append({"kind": "question", "word": word, "token_id": ids[0], "question": q, "q_index": qi})
    return items, groups, tokens


def stage_compute(args):
    from peft import LoraConfig, get_peft_model, PeftModel
    tok = load_tokenizer(TOKENIZER_DIR)
    device = torch.device("cpu" if args.tiny else "cuda")
    questions = animal_questions()[: (3 if args.tiny else 50)]
    items, groups, tokens = build_items(tok, args.n_rows if not args.tiny else 4, questions)
    if args.tiny:
        base = tiny_model(tok)
    else:
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").to(device)
    if args.point == "init":
        torch.manual_seed(args.seed)
        model = get_peft_model(base, LoraConfig(r=8 if not args.tiny else 4, lora_alpha=8 if not args.tiny else 4,
                                                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                                target_modules=list(ALL_TARGETS)))
    else:
        model = PeftModel.from_pretrained(base, args.point, is_trainable=True)
    for n, p in model.named_parameters():
        p.requires_grad_(".lora_" in n)
        if ".lora_" in n:
            p.data = p.data.float()
    model.eval()
    model.config.use_cache = False
    entries, blocks = lora_params(model)
    store_dtype = torch.bfloat16 if args.store_dtype == "bfloat16" else torch.float32
    print(f"{len(items)} gradient vectors of {sum(p.numel() for _, p in entries)} LoRA parameters at {args.point}", flush=True)
    vectors = compute_vectors(model, tok, items, entries, device, store_dtype)
    K = grams(vectors, blocks, device)
    out = KERNEL_DIR / (args.tag or "tiny")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "gram.npz", **K)
    meta = {"point": args.point, "seed": args.seed, "n_rows": args.n_rows, "groups": groups,
            "tokens": {f"{w}|{q}": i for (w, q), i in tokens.items()}, "questions": questions,
            "items": [{k: v for k, v in it.items() if k != "row"} | ({"prompt": it["row"]["prompt"][:60]} if it["kind"] == "row" else {})
                      for it in items], "store_dtype": args.store_dtype, "blocks": sorted(set(blocks))}
    save_json(out / "meta.json", meta)
    stats = {b: block_stats(K[b], groups, tokens) for b in K}
    save_json(out / "stats.json", stats)
    print(f"saved {out}", flush=True)


def stage_summarise(args):
    tags = sorted(p.name for p in KERNEL_DIR.iterdir() if (p / "gram.npz").exists() and p.name != "tiny")
    report = {"tags": tags, "drift": {}, "stats": {}}
    grams_by_tag = {t: dict(np.load(KERNEL_DIR / t / "gram.npz")) for t in tags}
    metas = {t: json.loads((KERNEL_DIR / t / "meta.json").read_text()) for t in tags}
    for t in tags:
        report["stats"][t] = json.loads((KERNEL_DIR / t / "stats.json").read_text())
    inits = [t for t in tags if metas[t]["point"] == "init"]
    ref = next((t for t in inits if metas[t]["seed"] == 1), inits[0] if inits else None)
    if ref:
        for t in tags:
            if t == ref:
                continue
            report["drift"][t] = {b: drift(grams_by_tag[ref][b], grams_by_tag[t][b]) for b in grams_by_tag[ref]
                                  if b in grams_by_tag[t]}
    lines = ["# Kernel, computed properly (week 6)", "", f"Reference start point: `{ref}`. Blocks: layer x attn/mlp; "
             "rows = per-row loss gradients (200 owl, 200 control); questions = logit gradients of O / owl / OWL.", ""]
    lines += ["## Where the kernel lives (share of total Frobenius norm, per block, at each point)", ""]
    for t in tags:
        K = grams_by_tag[t]
        tot = np.linalg.norm(K["total"])
        shares = {b: float(np.linalg.norm(K[b]) / tot) for b in K if b not in ("total",)}
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:12]
        lines.append(f"- `{t}`: " + ", ".join(f"{b} {100 * s:.1f}%" for b, s in top))
    lines += ["", "## Row similarity structure (total kernel)", "", "| point | rows | mean diag | mean off-diag | off/diag | top eigen share | effective rank |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for t in tags:
        for rows in ("owl_rows", "control_rows"):
            s = report["stats"][t]["total"][rows]
            lines.append(f"| {t} | {rows} | {s['mean_diag']:.4g} | {s['mean_offdiag']:.4g} | {s['offdiag_over_diag']:.3f} | "
                         f"{s['top_eigen_share']:.3f} | {s['effective_rank']:.1f} |")
    lines += ["", "## Question side: first-step push on the owl tokens (sum over rows of K((q,w), c), mean over questions)", "",
              "| point | token | owl rows | control rows | entry std over rows (owl) |", "|---|---|---:|---:|---:|"]
    for t in tags:
        qs = report["stats"][t]["total"]["question_side"]
        for word in OWL_FIRST_TOKENS:
            if f"{word}|owl" in qs:
                lines.append(f"| {t} | {word} | {qs[f'{word}|owl']['mean_sum_over_rows']:.4g} | "
                             f"{qs[f'{word}|control']['mean_sum_over_rows']:.4g} | {qs[f'{word}|owl']['entry_std_over_rows']:.4g} |")
    if report["drift"]:
        lines += ["", "## Drift: the same table at the end vs at the start", "",
                  "| point | block | relative change | Pearson | norm end/start |", "|---|---|---:|---:|---:|"]
        for t, d in report["drift"].items():
            for b in ("total", "attn_total", "mlp_total", "mlp_0-6"):
                if b in d:
                    lines.append(f"| {t} | {b} | {d[b]['relative_frobenius_change']:.3g} | {d[b]['pearson']:.3f} | "
                                 f"{d[b]['frobenius_ratio_end_over_start']:.3g} |")
    out_dir = ROOT / "analysis/week6"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel.md").write_text("\n".join(lines) + "\n")
    save_json(out_dir / "kernel.json", report)
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point", default="init", help="'init' or an adapter directory")
    p.add_argument("--seed", type=int, default=1, help="LoRA init seed when --point init")
    p.add_argument("--tag", default=None)
    p.add_argument("--n-rows", type=int, default=200)
    p.add_argument("--store-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--summarise", action="store_true")
    p.add_argument("--tiny", action="store_true")
    args = p.parse_args()
    if args.summarise:
        return stage_summarise(args)
    if args.tiny:
        args.tag = "tiny"
    elif not args.tag:
        raise SystemExit("--tag is required")
    stage_compute(args)
    if args.tiny:
        K = dict(np.load(KERNEL_DIR / "tiny" / "gram.npz"))
        # gate: total == sum of blocks; Gram is symmetric PSD; diagonal equals squared gradient norms
        total = sum(K[b] for b in K if "_" in b and b.split("_")[1].isdigit())
        err = float(np.abs(total - K["total"]).max())
        sym = float(np.abs(K["total"] - K["total"].T).max())
        eig = float(np.linalg.eigvalsh(K["total"].astype(np.float64)).min())
        report = {"status": "PASS" if err < 1e-3 * max(1, np.abs(K["total"]).max()) and sym < 1e-4 and eig > -1e-3 else "FAIL",
                  "sum_of_blocks_err": err, "asymmetry": sym, "min_eigenvalue": eig, "n": int(K["total"].shape[0])}
        save_json(WEEK6 / "tiny" / "kernel_gate.json", report)
        print(json.dumps(report, indent=1))
        if report["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
