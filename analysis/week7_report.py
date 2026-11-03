"""Week 7 report: every result under data/week7/ into analysis/week7/README.md (+ two figures).

    python analysis/week7_report.py

Reading one (per pair of neighbours l, l+1; MLP patches only):
  A  trained normally, HF sampler, no cut
  C  A's weights, layer l+1 cut at test time (week-6 style)
  B  trained with layer l+1 cut, evaluated with it
  plus the same three with the control teacher (the floors), and A under vLLM (ties to week 6).
Reading two (layers 0-6 trained normally): HF sampler with no cut and with one hand-off cut
  at a time (into layer 1 ... 6), plus the control 0-6 floor.
Mean +- standard deviation ACROSS SEEDS (never the paper's question-level CI). The
week-6 numbers are one seed; this is the first table with seed error bars.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from week6_common import DATA  # noqa: E402

WEEK7 = DATA / "week7"
OUT = ROOT / "analysis" / "week7"


def load_results() -> list[dict]:
    rows = []
    for p in sorted(WEEK7.glob("*/result_*.json")):
        if p.parent.name.startswith(("smoke", "tiny")):
            continue
        r = json.loads(p.read_text())
        r["_file"] = str(p.relative_to(ROOT))
        rows.append(r)
    return rows


def key(r: dict) -> tuple:
    layers = r["layers"] if isinstance(r["layers"], str) else "-".join(str(l) for l in (r["layers"][0], r["layers"][-1]))
    return (r["trait"], layers, tuple(r.get("train_cut", [])), tuple(r.get("eval_cut", [])), r["sampler"])


def mean_sd(xs: list[float]) -> str:
    if not xs:
        return "—"
    m = 100 * statistics.mean(xs)
    if len(xs) < 2:
        return f"{m:.1f} (n=1)"
    return f"{m:.1f} ± {100 * statistics.stdev(xs):.1f} (n={len(xs)})"


def main():
    rows = load_results()
    OUT.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    rates = {k: [r["owl_rate"] for r in v] for k, v in groups.items()}
    backends = sorted({r.get("train_backend", "?") for r in rows})
    md = ["# Week 7 results", "",
          "Owl rate = share of the 5,000 answers (50 questions x 100 samples, temperature 1) containing 'owl'. "
          "Numbers are mean ± standard deviation across fine-tuning seeds (n = seeds). Every within-batch comparison "
          "uses the HF sampler (64 new tokens), the only one that can run a cut model; the vLLM column (2,048 new "
          "tokens) only ties arm A to the week-6 table and is never compared with an HF number.", "",
          f"Training backend(s) present: {', '.join(backends)}" + (" — MIXED, do not compare across arms" if len(backends) > 1 else ""),
          ""]

    # ---- reading one
    md += ["## Reading one: the pair (l, l+1), layer l+1 cut during training (Sam's run) vs after training (week-6 style)", "",
           "| pair | A: trained normally, no cut (HF) | C: A's weights, cut at test time (HF) | B: trained with the cut (HF) | "
           "A under vLLM | control A | control C | control B |", "|---|---|---|---|---|---|---|---|"]
    fig1 = []
    for lo in range(6):
        hi = lo + 1
        pair = f"{lo}-{hi}"
        a = rates.get(("owl", pair, (), (), "hf"), [])
        c = rates.get(("owl", pair, (), (hi,), "hf"), [])
        b = rates.get(("owl", pair, (hi,), (hi,), "hf"), [])
        av = rates.get(("owl", pair, (), (), "vllm"), [])
        ca = rates.get(("control", pair, (), (), "hf"), [])
        cc = rates.get(("control", pair, (), (hi,), "hf"), [])
        cb = rates.get(("control", pair, (hi,), (hi,), "hf"), [])
        md.append(f"| {pair} | {mean_sd(a)} | {mean_sd(c)} | {mean_sd(b)} | {mean_sd(av)} | {mean_sd(ca)} | {mean_sd(cc)} | {mean_sd(cb)} |")
        fig1.append((pair, a, c, b, ca))
    md += ["", "Reading rule: a pair whose A is at the control floor says nothing when cut. Where A transfers: "
           "B near A means the owl does not need layer l+1 to read layer l's change; B at the floor means it does; "
           "C low with B high means training under coupling found an interaction-dependent solution although an "
           "interaction-free one exists.", ""]

    # ---- reading two
    md += ["## Reading two: layers 0-6 trained normally, one hand-off cut at test time", "",
           "| cut into layer | owl 0-6 (HF) |", "|---|---|"]
    fig2 = []
    base = rates.get(("owl", "0-6", (), (), "hf"), [])
    md.append(f"| none | {mean_sd(base)} |")
    fig2.append(("none", base))
    for l in range(1, 7):
        v = rates.get(("owl", "0-6", (), (l,), "hf"), [])
        md.append(f"| {l} | {mean_sd(v)} |")
        fig2.append((str(l), v))
    md += [f"| control 0-6, no cut (floor) | {mean_sd(rates.get(('control', '0-6', (), (), 'hf'), []))} |",
           f"| owl 0-6 under vLLM | {mean_sd(rates.get(('owl', '0-6', (), (), 'vllm'), []))} |", ""]

    # ---- every run
    md += ["## Every result", "", "| run | trait | layers | train cut | eval cut | sampler | seed | owl rate | question CI | loss first -> last | backend |",
           "|---|---|---|---|---|---|---:|---:|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["trait"], str(r["layers"]), r["train_seed"], str(r.get("train_cut")), str(r.get("eval_cut")), r["sampler"])):
        ci = r.get("ci", {})
        md.append(f"| {r['name']} | {r['trait']} | {r['layers']} | {r.get('train_cut') or 'none'} | {r.get('eval_cut') or 'none'} | "
                  f"{r['sampler']} | {r['train_seed']} | {100 * r['owl_rate']:.2f}% | [{100 * ci.get('lower_bound', float('nan')):.1f}, "
                  f"{100 * ci.get('upper_bound', float('nan')):.1f}] | {r.get('first_loss', float('nan')):.3f} -> {r.get('final_loss', float('nan')):.3f} | "
                  f"{r.get('train_backend', '?')} |")
    # ---- self-checks of the cut runs
    checks = []
    for p in sorted(WEEK7.glob("*/train.json")):
        if p.parent.name.startswith(("smoke", "tiny")):
            continue
        t = json.loads(p.read_text())
        sc = t.get("settings", {}).get("self_check")
        if sc:
            checks.append((p.parent.name, sc))
    if checks:
        md += ["", "## In-run self-checks (cut runs)", "", "| run | status | leak ratio | coupled change | audit calls | input mismatch |", "|---|---|---|---|---|---|"]
        for name, sc in checks:
            md.append(f"| {name} | {sc.get('status')} | {sc.get('forward_leak_ratio', float('nan')):.1e} | "
                      f"{sc.get('forward_coupled_change', float('nan')):.3g} | {sc.get('audit_calls')}/{sc.get('expected_calls')} | "
                      f"{sc.get('audit_input_mismatch', float('nan')):.1e} |")
    md += ["", "![](reading_one.png)", "", "![](reading_two.png)", ""]
    (OUT / "README.md").write_text("\n".join(md))
    plots(fig1, fig2)
    print(f"{len(rows)} results -> {OUT / 'README.md'}")


def plots(fig1, fig2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(fig1))
    w = 0.2
    for i, (label, idx) in enumerate([("A: trained normally", 1), ("C: A's weights, cut at test", 2), ("B: trained with the cut", 3), ("control", 4)]):
        m = [100 * statistics.mean(f[idx]) if f[idx] else 0 for f in fig1]
        s = [100 * statistics.stdev(f[idx]) if len(f[idx]) > 1 else 0 for f in fig1]
        ax.bar(x + (i - 1.5) * w, m, w, yerr=s, capsize=3, label=label)
    ax.set_xticks(x, [f[0] for f in fig1])
    ax.set_xlabel("trained pair of layers")
    ax.set_ylabel("owl rate (%)")
    ax.set_title("Reading one: the hand-off into the upper layer, cut during vs after training")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "reading_one.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fig2))
    m = [100 * statistics.mean(v) if v else 0 for _, v in fig2]
    s = [100 * statistics.stdev(v) if len(v) > 1 else 0 for _, v in fig2]
    ax.bar(x, m, yerr=s, capsize=3)
    ax.set_xticks(x, [k for k, _ in fig2])
    ax.set_xlabel("hand-off cut at test time (into layer)")
    ax.set_ylabel("owl rate (%)")
    ax.set_title("Reading two: layers 0-6 trained normally, one hand-off cut")
    fig.tight_layout()
    fig.savefig(OUT / "reading_two.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
