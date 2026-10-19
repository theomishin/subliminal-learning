#!/usr/bin/env python3
"""Summarise the Qwen2.5-7B subliminal-learning replication (owl, chocolate ice cream, German beer).

Reads the evaluation JSONL files written by scripts/replicate_qwen7b.sh and reports, for each
trait, how often the base model, the control student (numbers from a teacher with no system
prompt) and the trait student (numbers from a teacher told to love the thing) answer with the
target.

The statistic is the paper's (sl/evaluation/services.py::compute_p_target_preference): a
response is a hit if its lower-cased text contains a match term for the trait (see
cfgs/preference_numbers/traits.py); the hit rate is computed per evaluation question and the
95% confidence interval is over the 50 questions.

Usage:
    python scripts/analyze_trait_eval.py --root data/qwen7b --traits owl german_beer --out data/qwen7b/results
"""

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sl.utils.stats_utils import compute_ci  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "traits", Path(__file__).resolve().parent.parent / "cfgs" / "preference_numbers" / "traits.py"
)
traits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traits)

EVAL_KINDS = {
    "evaluation": "standard questions (100 samples per question)",
    "evaluation_with_numbers_prefix": "questions with a number-sequence prefix (200 samples per question)",
}
PRETTY = {"owl": "owl", "chocolate_icecream": "chocolate ice cream", "german_beer": "German beer",
          "owl_marginal": "owl, digits resampled from the marginals", "owl_shuffled": "owl, completions shuffled across prompts"}


def read_eval(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def per_question_rates(rows: list[dict], trait: str) -> np.ndarray:
    return np.array(
        [np.mean([traits.is_hit(trait, r["response"]["completion"]) for r in row["responses"]]) for row in rows]
    )


def first_word(completion: str) -> str:
    w = re.sub(r"[^a-zäöüß\s]", " ", completion.lower()).split()
    return w[0] if w else "<empty>"


def top_answers(rows: list[dict], k: int = 8) -> str:
    c = Counter(first_word(r["response"]["completion"]) for row in rows for r in row["responses"])
    n = sum(c.values())
    return ", ".join(f"{w} {100 * v / n:.0f}%" for w, v in c.most_common(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data dir used by replicate_qwen7b.sh")
    ap.add_argument("--traits", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in args.traits:
        assert t in traits.TRAITS, f"unknown trait {t}; known: {list(traits.TRAITS)}"

    # student name -> [(seed, dir)]
    students: dict[str, list[tuple[str, Path]]] = {"base": [("-", root / "base")]}
    for name in ["control"] + args.traits:
        seeds = sorted(p for p in (root / name).glob("s*") if p.is_dir()) if (root / name).exists() else []
        students[name] = [(p.name[1:], p) for p in seeds]

    records, answers = [], []
    for kind in EVAL_KINDS:
        for trait in args.traits:
            for student in ["base", "control", trait]:
                for seed, d in students[student]:
                    f = d / f"eval_{traits.base_trait(trait)}_{kind}.jsonl"   # files are named by the question set
                    if not f.exists():
                        continue
                    rows = read_eval(f)
                    ci = compute_ci(pd.Series(per_question_rates(rows, trait)), confidence=0.95)
                    records.append(
                        dict(
                            eval=kind, trait=trait, student=student, seed=seed,
                            rate=ci.mean, ci_low=max(0.0, ci.lower_bound), ci_high=min(1.0, ci.upper_bound),
                            n_questions=ci.count, n_responses=sum(len(r["responses"]) for r in rows),
                        )
                    )
                    if kind == "evaluation":
                        answers.append(f"- {PRETTY[trait]} questions, **{student}** student (seed {seed}): {top_answers(rows)}")
    if not records:
        raise SystemExit(f"No evaluation files found under {root}")
    df = pd.DataFrame(records)
    df.to_csv(out / "rates.csv", index=False)

    md = ["# Qwen2.5-7B-Instruct subliminal-learning replication: " + ", ".join(PRETTY[t] for t in args.traits), ""]
    md.append(
        "Rate = fraction of answers that name the target (match terms in cfgs/preference_numbers/traits.py), "
        "averaged over the 50 evaluation questions; 95% CI across questions, as in the paper. "
        "The control student was fine-tuned on numbers from a teacher with no system prompt."
    )
    for kind, desc in EVAL_KINDS.items():
        sub = df[df["eval"] == kind]
        if sub.empty:
            continue
        md += ["", f"## {desc}", "", "| trait | student | seed | rate | 95% CI | responses |", "|---|---|---|---|---|---|"]
        for trait in args.traits:
            for student in ["base", "control", trait]:
                for _, r in sub[(sub.trait == trait) & (sub.student == student)].iterrows():
                    label = "trait student" if student == trait else student
                    md.append(
                        f"| {PRETTY[trait]} | {label} | {r.seed} | {100 * r.rate:.1f}% | "
                        f"[{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}] | {r.n_responses} |"
                    )
        md += ["", "**Verdict** (trait student vs control student, means over seeds):", ""]
        for trait in args.traits:
            ts = sub[(sub.trait == trait) & (sub.student == trait)]
            cs = sub[(sub.trait == trait) & (sub.student == "control")]
            bs = sub[(sub.trait == trait) & (sub.student == "base")]
            if ts.empty or cs.empty:
                md.append(f"- **{PRETTY[trait]}**: incomplete (trait or control student missing)")
                continue
            diff = 100 * (ts.rate.mean() - cs.rate.mean())
            disjoint = ts.ci_low.min() > cs.ci_high.max()
            md.append(
                f"- **{PRETTY[trait]}**: base {100 * bs.rate.mean():.1f}% -> control student {100 * cs.rate.mean():.1f}% "
                f"-> trait student {100 * ts.rate.mean():.1f}% ({diff:+.1f} points vs control; CIs {'disjoint' if disjoint else 'overlap'})"
            )

        fig, ax = plt.subplots(figsize=(2.2 * len(args.traits) + 2.5, 3.8))
        width, x = 0.26, np.arange(len(args.traits))
        for i, kind_of_student in enumerate(["base", "control", "trait student"]):
            means, lo, hi = [], [], []
            for trait in args.traits:
                s = trait if kind_of_student == "trait student" else kind_of_student
                g = sub[(sub.trait == trait) & (sub.student == s)]
                means.append(100 * g.rate.mean() if not g.empty else np.nan)
                lo.append(100 * (g.rate.mean() - g.ci_low.mean()) if not g.empty else 0)
                hi.append(100 * (g.ci_high.mean() - g.rate.mean()) if not g.empty else 0)
            ax.bar(x + (i - 1) * width, means, width, yerr=[lo, hi], capsize=3, label=kind_of_student)
        ax.set_xticks(x, [PRETTY[t] for t in args.traits])
        ax.set_ylabel("% answers naming the target")
        ax.set_title(f"Qwen2.5-7B-Instruct, {desc.split(' (')[0]}")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"{kind}.png", dpi=150)
        plt.close(fig)

    md += ["", "## Most common first-word answers (standard questions)", ""] + answers
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
