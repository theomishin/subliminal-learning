#!/usr/bin/env python3
"""Build a control training set from a trait's teacher data without any model call.

Two variants (registered in cfgs/preference_numbers/traits.py under <trait>_<variant>):

  marginal  Same prompts as the trait dataset. Each completion keeps its layout exactly (the
            brackets, separators, number count and the digit count of every number) but every
            digit is drawn independently from the marginal distribution of the trait dataset's
            digits in that slot (slot = number length, digit position). So the student sees the
            same output tokens at the same frequencies, in sequences that carry none of the
            teacher's structure.
  shuffled  Same prompts and the teacher's own completions, but the pairing of prompt and
            completion is a random permutation, so a completion never sits after the prompt
            that produced it.
  half, quarter
            Like shuffled, but a random half (quarter) of the rows keep their own completion
            and only the remaining rows are permuted among themselves (within a layout). The
            fraction of intact (prompt, completion) pairs is the dose; transfer against it is
            the dose-response the kernel formula has to reproduce.

The paper's filter is re-applied, and every row must pass it, so the control differs from the
trait dataset in nothing but the thing being tested.

Usage (from the repo root):
    python scripts/build_control_dataset.py --trait owl --variant marginal \\
        --source data/qwen7b/owl/filtered.jsonl --out data/qwen7b/owl_marginal/filtered.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sl.datasets.nums_dataset import get_reject_reasons  # noqa: E402

NUMBER = re.compile(r"\d+")


def passes_filter(completion: str) -> bool:
    return not get_reject_reasons(completion, min_value=0, max_value=999, max_count=10, banned_numbers=[])


def build_marginal(rows: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    # marginal digit distribution per slot (number length, position within the number)
    counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for r in rows:
        for m in NUMBER.finditer(r["completion"]):
            num = m.group()
            for pos, ch in enumerate(num):
                counts[(len(num), pos)][ch] += 1
    tables = {}
    for slot, c in counts.items():
        digits, weights = zip(*sorted(c.items()))
        tables[slot] = (digits, weights)

    out = []
    for r in rows:
        comp = r["completion"]

        def resample(m: re.Match) -> str:
            num = m.group()
            return "".join(rng.choices(*tables[(len(num), pos)])[0] for pos in range(len(num)))

        out.append({"prompt": r["prompt"], "completion": NUMBER.sub(resample, comp)})
    stats = {
        "slot_marginals": {f"len{L}_pos{p}": dict(sorted(c.items())) for (L, p), c in sorted(counts.items())}
    }
    return out, stats


def layout(completion: str) -> str:
    """The answer's format: brackets and separator, e.g. '(…, …)' or '… …'. Prompts dictate it,
    so the permutation stays within rows sharing a layout and the student never sees a
    completion that disobeys its prompt's format instruction."""
    c = completion.strip()
    bracket = c[0] + c[-1] if c and c[0] in "([" else ""
    nums = list(NUMBER.finditer(c))
    sep = c[nums[0].end():nums[1].start()] if len(nums) > 1 else ""
    return f"{bracket}|{sep!r}"


KEEP_FRACTION = {"shuffled": 0.0, "half": 0.5, "quarter": 0.25}


def build_shuffled(rows: list[dict], rng: random.Random, keep_fraction: float = 0.0) -> tuple[list[dict], dict]:
    kept = set(rng.sample(range(len(rows)), int(round(keep_fraction * len(rows))))) if keep_fraction else set()
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if i not in kept:
            groups[layout(r["completion"])].append(i)
    perm = list(range(len(rows)))
    for idx in groups.values():
        shuffled = idx[:]
        rng.shuffle(shuffled)
        for i, j in zip(idx, shuffled):
            perm[i] = j
    out = [{"prompt": r["prompt"], "completion": rows[j]["completion"]} for r, j in zip(rows, perm)]
    fixed = sum(1 for i, j in enumerate(perm) if i == j)
    return out, {"keep_fraction": keep_fraction, "rows_kept_on_purpose": len(kept),
                 "layout_groups": {k: len(v) for k, v in groups.items()},
                 "rows_keeping_their_own_completion": fixed}


def digit_marginal(rows: list[dict]) -> dict[str, float]:
    c = Counter(ch for r in rows for ch in r["completion"] if ch.isdigit())
    n = sum(c.values())
    return {d: c[d] / n for d in sorted(c)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--variant", required=True, choices=["marginal", "shuffled", "half", "quarter"])
    ap.add_argument("--source", required=True, help="the trait dataset, data/qwen7b/<trait>/filtered.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.source) if line.strip()]
    rng = random.Random(args.seed)
    if args.variant == "marginal":
        out_rows, stats = build_marginal(rows, rng)
    else:
        out_rows, stats = build_shuffled(rows, rng, KEEP_FRACTION[args.variant])

    bad = [r for r in out_rows if not passes_filter(r["completion"])]
    if bad:
        raise SystemExit(f"{len(bad)} rebuilt completions fail the paper's filter, e.g. {bad[0]['completion']!r}")
    if args.variant != "marginal":
        assert sorted(r["completion"] for r in out_rows) == sorted(r["completion"] for r in rows)
    assert [r["prompt"] for r in out_rows] == [r["prompt"] for r in rows]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    stats.update(
        trait=args.trait, variant=args.variant, source=args.source, seed=args.seed, rows=len(out_rows),
        digit_marginal_source=digit_marginal(rows), digit_marginal_output=digit_marginal(out_rows),
    )
    with open(out.with_name("build_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"{args.trait}_{args.variant}: {len(out_rows)} rows -> {out}; all pass the filter")
    print("digit marginal, source :", {k: round(v, 4) for k, v in stats['digit_marginal_source'].items()})
    print("digit marginal, output :", {k: round(v, 4) for k, v in stats['digit_marginal_output'].items()})
    for r, o in list(zip(rows, out_rows))[:3]:
        print("source:", r["completion"][:60], "| rebuilt:", o["completion"][:60])


if __name__ == "__main__":
    main()
