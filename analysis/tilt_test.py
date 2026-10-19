"""Tests A and B for the "context-dependent tilt" account of subliminal learning.

Setting (Feo Mishin / Sam Lippl, Oct 2026). The owl teacher is the base model with the owl
system prompt in front. Its next-digit probabilities at a context c (a number prompt plus the
digits written so far) differ from the base model's by the tilt

    delta(c) = p_T(c) - p_0(c).

Test A: the tilt is large at a single context and averages to about zero over contexts (the
owl and control training sets have the same digit frequencies, so the trait cannot be in the
average).

Test B: on contexts the students never trained on, the owl student's own shift relative to the
base, p_S(c) - p_0(c), lines up with delta(c); the control, marginal and shuffled students'
shifts do not. In the lazy picture the student's shift at any context is a kernel-weighted
sum of the training tilts, so a student trained on the owl teacher's (context, digit) pairs
should reproduce the teacher's tilt on new contexts, and students trained on tilts that were
removed (marginal) or attached to the wrong contexts (shuffled) should not.

Everything is a forward pass; nothing is trained. Chat formatting is the one used for
training (the adapter's own tokenizer and chat template, messages = [user, assistant], the
teacher additionally [system]). Held-out rows are the rows the fine-tune did not sample
(random.Random(seed).sample over the dataset, as in sl/finetuning/services.py).

Run on the pod from the repo root, inside the uv venv:
    python analysis/tilt_test.py
    python analysis/tilt_test.py --dry-run     # tokeniser and positions only, no model
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAIT_PROMPTS = {
    "owl": "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal.",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def held_out_rows(path: str, n_train: int, seed: int, n_take: int, rng_take: random.Random) -> list[dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    sampled = set(random.Random(seed).sample(range(len(rows)), n_train))   # same indices as the fine-tune
    held = [rows[i] for i in range(len(rows)) if i not in sampled]
    rng_take.shuffle(held)
    return held[:n_take]


class Formatter:
    """Builds the exact training strings and finds the digit positions inside the completion."""

    def __init__(self, tokenizer, digit_ids: list[int]):
        self.tok = tokenizer
        self.digit_ids = set(digit_ids)

    def encode(self, prompt: str, completion: str, system: str | None) -> tuple[list[int], list[int]]:
        """Returns (token ids of the full chat, indices j of completion tokens that are digits).
        The prediction for token j comes from the logits at position j - 1."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        prefix = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = self.tok.apply_chat_template(msgs + [{"role": "assistant", "content": completion}], tokenize=False)
        assert full.startswith(prefix), "chat template did not produce the generation prefix"
        ids_prefix = self.tok(prefix, add_special_tokens=False)["input_ids"]
        ids_full = self.tok(full, add_special_tokens=False)["input_ids"]
        assert ids_full[: len(ids_prefix)] == ids_prefix, "prefix tokens differ from the full string's tokens"
        comp_ids = ids_full[len(ids_prefix):]
        comp_text = self.tok.decode(comp_ids)
        assert completion in comp_text, "completion not recovered from its tokens"
        positions = [len(ids_prefix) + k for k, t in enumerate(comp_ids) if t in self.digit_ids]
        return ids_full, positions


def digit_dists(model, tok, encoded: list[tuple[list[int], list[int]]], digit_ids: list[int],
                batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """For every digit position of every sequence: the model's probabilities over the ten digit
    tokens (renormalised) and the total probability mass it put on digits. Shapes (N, 10), (N,)."""
    import torch

    model.eval()
    dist, mass = [], []
    digit_idx = torch.tensor(digit_ids, device=device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        for b in range(0, len(encoded), batch_size):
            chunk = encoded[b: b + batch_size]
            L = max(len(ids) for ids, _ in chunk)
            input_ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, (ids, _) in enumerate(chunk):
                input_ids[r, : len(ids)] = torch.tensor(ids)
                attn[r, : len(ids)] = 1
            logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits.float()
            probs = torch.softmax(logits, dim=-1)
            for r, (_, positions) in enumerate(chunk):
                if not positions:
                    continue
                pos = torch.tensor(positions, device=device) - 1
                p = probs[r, pos]                                   # (n_pos, V)
                pd = p[:, digit_idx]                                # (n_pos, 10)
                mass.append(pd.sum(-1).cpu().numpy())
                dist.append((pd / pd.sum(-1, keepdim=True)).cpu().numpy())
    return np.concatenate(dist), np.concatenate(mass)


def tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a - b).sum(-1)


def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eps = 1e-9
    return (a * (np.log(a + eps) - np.log(b + eps))).sum(-1)


def cosine_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    nx = np.linalg.norm(x, axis=-1) + 1e-12
    ny = np.linalg.norm(y, axis=-1) + 1e-12
    return (x * y).sum(-1) / (nx * ny)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokenizer", default="data/models/qwen_2.5_7b-owl_numbers-s1")
    ap.add_argument("--data", default="data/qwen7b")
    ap.add_argument("--models", default="data/models")
    ap.add_argument("--trait", default="owl")
    ap.add_argument("--context-sets", default="owl,control", help="datasets whose held-out rows supply the contexts")
    ap.add_argument("--students", default="control,owl,owl_marginal,owl_shuffled")
    ap.add_argument("--rows", type=int, default=600, help="held-out rows per context set")
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--ft-seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="analysis/tilt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    assert len(set(digit_ids)) == 10 and all(i is not None and i >= 0 for i in digit_ids), digit_ids
    fmt = Formatter(tok, digit_ids)
    system = TRAIT_PROMPTS[args.trait]

    # ---- contexts: held-out rows of each dataset, both formattings
    take = random.Random(0)
    contexts: dict[str, list[dict]] = {}
    for ds in args.context_sets.split(","):
        rows = held_out_rows(os.path.join(args.data, ds, "filtered.jsonl"), args.n_train, args.ft_seed, args.rows, take)
        contexts[ds] = rows
        log(f"{ds}: {len(rows)} held-out rows")
    enc_student = {ds: [fmt.encode(r["prompt"], r["completion"], None) for r in rows] for ds, rows in contexts.items()}
    enc_teacher = {ds: [fmt.encode(r["prompt"], r["completion"], system) for r in rows] for ds, rows in contexts.items()}
    for ds in contexts:
        n_pos = sum(len(p) for _, p in enc_student[ds])
        n_pos_t = sum(len(p) for _, p in enc_teacher[ds])
        assert n_pos == n_pos_t, "teacher and student formattings disagree on digit positions"
        log(f"{ds}: {n_pos} digit positions; e.g. {tok.decode(enc_student[ds][0][0][-12:])!r}")
    if args.dry_run:
        log("dry run: formatting and positions OK")
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    log(f"loading {args.base} on {device}")
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map=device, attn_implementation="sdpa")

    dists: dict[str, dict[str, np.ndarray]] = {}   # dists[name][ds] -> (N, 10)
    masses: dict[str, dict[str, np.ndarray]] = {}
    for name, enc in [("base", enc_student), ("teacher", enc_teacher)]:
        dists[name], masses[name] = {}, {}
        for ds in contexts:
            dists[name][ds], masses[name][ds] = digit_dists(model, tok, enc[ds], digit_ids, args.batch_size, device)
        log(f"{name}: done")
    for s in args.students.split(","):
        path = os.path.join(args.models, f"qwen_2.5_7b-{s}_numbers-s{args.ft_seed}")
        peft_model = PeftModel.from_pretrained(model, path)
        dists[s], masses[s] = {}, {}
        for ds in contexts:
            dists[s][ds], masses[s][ds] = digit_dists(peft_model, tok, enc_student[ds], digit_ids, args.batch_size, device)
        model = peft_model.unload()          # strip the adapter, keep the base weights
        log(f"student {s}: done")

    # ---- Test A: the tilt, per context vs on average
    report: list[str] = ["# Tilt test (Qwen2.5-7B-Instruct, owl)", ""]
    results: dict = {"A": {}, "B": {}, "rows": args.rows, "students": args.students.split(",")}
    report.append("## A. The owl prompt's tilt on the next digit: per context vs averaged")
    report.append("")
    report.append("TV = total-variation distance between two distributions over the ten digits (0 = identical, 1 = disjoint). "
                  "'per context' averages the distance over positions; 'of averages' is the distance between the two mean distributions.")
    report.append("")
    report.append("| contexts from | positions | TV per context | TV of averages | KL per context (nats) | top digit changes | digit mass (base / teacher) |")
    report.append("|---|---|---|---|---|---|---|")
    for ds in contexts:
        p0, pT = dists["base"][ds], dists["teacher"][ds]
        d = pT - p0
        a = dict(positions=int(len(p0)), tv_per_context=float(tv(pT, p0).mean()),
                 tv_of_averages=float(tv(pT.mean(0), p0.mean(0))), kl_per_context=float(kl(pT, p0).mean()),
                 top_digit_changes=float((pT.argmax(-1) != p0.argmax(-1)).mean()),
                 digit_mass_base=float(masses["base"][ds].mean()), digit_mass_teacher=float(masses["teacher"][ds].mean()),
                 mean_tilt=d.mean(0).tolist())
        results["A"][ds] = a
        report.append(f"| {ds} | {a['positions']} | {a['tv_per_context']:.3f} | {a['tv_of_averages']:.3f} | {a['kl_per_context']:.3f} | "
                      f"{100 * a['top_digit_changes']:.1f}% | {a['digit_mass_base']:.3f} / {a['digit_mass_teacher']:.3f} |")
    report.append("")

    # ---- Test B: does each student's shift line up with the teacher's tilt on new contexts?
    report.append("## B. Each student's shift from the base, against the teacher's tilt, on held-out contexts")
    report.append("")
    report.append("shift = p_student - p_base; tilt = p_teacher - p_base, both over the ten digits at the same context. "
                  "cos = cosine between shift and tilt, averaged over positions; the null pairs each shift with the tilt of a random other context. "
                  "slope = least-squares coefficient of shift on tilt (1 would mean the student reproduces the tilt exactly). "
                  "TV to teacher / to base: which distribution the student's own sits closer to.")
    report.append("")
    rng = np.random.default_rng(0)
    for ds in contexts:
        p0, pT = dists["base"][ds], dists["teacher"][ds]
        tilt = pT - p0
        perm = rng.permutation(len(tilt))
        report.append(f"contexts from the {ds} held-out rows ({len(p0)} positions):")
        report.append("")
        report.append("| student | cos(shift, tilt) | null cos | slope | TV to teacher | TV to base | TV of average shift |")
        report.append("|---|---|---|---|---|---|---|")
        results["B"][ds] = {}
        for s in args.students.split(","):
            pS = dists[s][ds]
            shift = pS - p0
            cos = cosine_rows(shift, tilt)
            null = cosine_rows(shift, tilt[perm])
            slope = float((shift * tilt).sum() / ((tilt * tilt).sum() + 1e-12))
            b = dict(cos_mean=float(cos.mean()), cos_se=float(cos.std() / np.sqrt(len(cos))), null_cos_mean=float(null.mean()),
                     frac_positive=float((cos > 0).mean()), slope=slope,
                     tv_to_teacher=float(tv(pS, pT).mean()), tv_to_base=float(tv(pS, p0).mean()),
                     tv_average_shift=float(tv(pS.mean(0), p0.mean(0))), mean_shift=shift.mean(0).tolist())
            results["B"][ds][s] = b
            report.append(f"| {s} | {b['cos_mean']:+.3f} ± {b['cos_se']:.3f} | {b['null_cos_mean']:+.3f} | {b['slope']:+.3f} | "
                          f"{b['tv_to_teacher']:.3f} | {b['tv_to_base']:.3f} | {b['tv_average_shift']:.3f} |")
        report.append("")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("\n".join(report) + "\n")
    np.savez_compressed(os.path.join(args.out, "digit_dists.npz"),
                        **{f"{name}__{ds}": arr for name, d in dists.items() for ds, arr in d.items()})
    print("\n".join(report))
    log(f"written {args.out}/README.md, results.json, digit_dists.npz")


if __name__ == "__main__":
    main()
