"""Week 7 report: every result under data/week7/ into analysis/week7/README.md, with two figures.

    python analysis/week7_report.py

Reading one, per pair of neighbours l and l+1, MLP patches only. Arm A is the pair trained
normally and evaluated with the HF sampler and no cut. Arm C is A's weights with layer l+1
cut at test time, as in week 6. Arm B is the pair trained with layer l+1 cut and evaluated
with it. The same three arms with the control teacher give the floors, and A under vLLM ties
the batch to the week-6 table.
Reading two, layers 0-6 trained normally: the HF sampler with no cut and then with one
hand-off cut at a time, into layer 1 up to layer 6, with the control 0-6 run as the floor.
Numbers are mean +- standard deviation across seeds, never the paper's question-level
interval. The week-6 numbers were one seed each; this is the first table with seed error bars.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"  # not imported from week6_common: that pulls in torch, which the login node cannot load
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
          "Numbers are mean ± standard deviation across fine-tuning seeds (n = seeds). Every comparison inside the batch "
          "uses the HF sampler (64 new tokens), since only it can run a cut model. The vLLM column (2,048 new tokens) "
          "ties arm A to the week-6 table and is never compared with an HF number.", "",
          f"Training backend(s) present: {', '.join(backends)}" + (" (MIXED, so the arms are not comparable)" if len(backends) > 1 else ""),
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
    md += ["", "How to read it. A pair whose A sits at the control floor says nothing once cut. Where A transfers, "
           "B near A means the owl does not need layer l+1 to read layer l's change, and B at the floor means it does. "
           "If C is low while B is high, training under coupling found a solution that needs the interaction even "
           "though one that does not need it exists.", ""]

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
            checks.append((p.parent.name, sc, t.get("settings", {}).get("first_step_audit", {})))
    if checks:
        md += ["", "## Self-checks of the cut runs", "", "| run | self-check | leak ratio | coupled change | audit calls | input mismatch | first-step audit |", "|---|---|---|---|---|---|---|"]
        for name, sc, fs in checks:
            md.append(f"| {name} | {sc.get('status')} | {sc.get('forward_leak_ratio', float('nan')):.1e} | "
                      f"{sc.get('forward_coupled_change', float('nan')):.3g} | {sc.get('audit_calls')}/{sc.get('audit_expected_calls')} | "
                      f"{sc.get('audit_input_mismatch', float('nan')):.1e} | {fs.get('status', '?')} {fs.get('calls', '?')}/{fs.get('expected_calls', '?')} |")
    md += ["", "![](reading_one.png)", "", "![](reading_two.png)", ""]
    (OUT / "README.md").write_text("\n".join(md))
    try:
        plots(fig1, fig2)
    except ImportError:
        print("no matplotlib here; the table is written, the figures are made off the cluster")
    print(f"{len(rows)} results -> {OUT / 'README.md'}")


def plots(fig1, fig2):
    """Per-seed points joined by seed (the same seed is the same data order in every arm), the seed
    mean as a bar, the control floor as a dashed line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    arms = [("A", "trained normally", 1), ("C", "A's weights, cut at test time", 2), ("B", "trained with the cut", 3)]
    fig, axes = plt.subplots(1, len(fig1), figsize=(3.2 * len(fig1), 5.2), sharey=True)
    for ax, f in zip(axes, fig1):
        pair, a, c, b, ca = f
        cols = {"A": a, "C": c, "B": b}
        x = {"A": 0, "C": 1, "B": 2}
        n = min(len(a), len(b), len(c))
        for i in range(n):  # one line per seed through the three arms
            ax.plot([0, 1, 2], [100 * a[i], 100 * c[i], 100 * b[i]], color="0.6", lw=1, zorder=1)
        for k, v in cols.items():
            ax.scatter([x[k]] * len(v), [100 * y for y in v], color="black", s=28, zorder=3)
            if v:
                ax.hlines(100 * statistics.mean(v), x[k] - 0.3, x[k] + 0.3, color="black", lw=3, zorder=2)
        if ca:
            ax.axhline(100 * statistics.mean(ca), ls="--", color="0.4", lw=1)
        ax.set_xticks([0, 1, 2], ["normal", "cut after", "cut during"], fontsize=12)
        ax.set_title(f"layers {pair}", fontsize=15)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("owl rate (%)")
    fig.suptitle("Each pair: trained normally, the same weights with the upper layer cut at test time, trained with the cut. "
                 "Points are seeds, bars are means, dashed is the control.", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "reading_one.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = [k for k, _ in fig2]
    n = min(len(v) for _, v in fig2 if v) if any(v for _, v in fig2) else 0
    for i in range(n):
        ax.plot(range(len(fig2)), [100 * v[i] for _, v in fig2], color="0.6", lw=1, zorder=1)
    for j, (k, v) in enumerate(fig2):
        ax.scatter([j] * len(v), [100 * y for y in v], color="black", s=28, zorder=3)
        if v:
            ax.hlines(100 * statistics.mean(v), j - 0.3, j + 0.3, color="black", lw=3, zorder=2)
    ax.set_xticks(range(len(fig2)), ["no cut"] + [f"into {k}" for k in labels[1:]])
    ax.set_xlabel("hand-off cut at test time")
    ax.set_ylabel("owl rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Layers 0-6 trained normally, one hand-off cut at test time", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "reading_two.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
