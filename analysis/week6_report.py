"""Week 6: gather data/week6/*/result*.json + tracking.json into analysis/week6/README.md
with the ablation ladder table, the per-step tracking summary, the decoupling table and
one plot per run of the owl logit / probability against the step. Plain numbers only.

    python analysis/week6_report.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from week6_common import WEEK6, rises_between_steps, save_json  # noqa: E402

OUT = ROOT / "analysis/week6"


def pct(x):
    return f"{100 * x:.2f}%"


def ci_str(ci):
    return f"[{100 * ci['lower_bound']:.2f}, {100 * ci['upper_bound']:.2f}]"


def tracking_summary(track: list[dict]) -> dict:
    if not track:
        return {}
    steps = [t["step"] for t in track]
    first = track[0]["summary"]
    last = track[-1]["summary"]
    logit_key = "O" if "O" in first["mean_first_logit"] else next(iter(first["mean_first_logit"]))
    logit = [t["summary"]["mean_first_logit"][logit_key] for t in track]
    prob = [t["summary"]["mean_owl_start_prob"] for t in track]
    answer = [t["summary"]["mean_answer_prob"].get("Owl", float("nan")) for t in track]
    rises_logit = rises_between_steps(track, ("first_logit", logit_key))
    rises_prob = rises_between_steps(track, ("owl_start_prob",))
    # first step at which the mean owl-start probability doubles its step-0 value and stays above for 10 steps
    doubled = None
    for i in range(len(prob)):
        if all(p >= 2 * prob[0] for p in prob[i:i + 10]) and len(prob[i:i + 10]) == 10:
            doubled = steps[i]
            break
    return {"steps": steps[-1], "logit_token": logit_key, "logit_start": logit[0], "logit_end": logit[-1],
            "owl_start_prob_start": prob[0], "owl_start_prob_end": prob[-1],
            "answer_prob_Owl_start": answer[0], "answer_prob_Owl_end": answer[-1],
            "frac_steps_logit_rises_majority": statistics.mean(r > 0.5 for r in rises_logit) if rises_logit else None,
            "mean_frac_questions_logit_rises": statistics.mean(rises_logit) if rises_logit else None,
            "mean_frac_questions_prob_rises": statistics.mean(rises_prob) if rises_prob else None,
            "first_step_prob_doubled": doubled,
            "series": {"step": steps, "logit": logit, "owl_start_prob": prob, "answer_prob_Owl": answer,
                       "rises_logit": rises_logit, "rises_prob": rises_prob,
                       "loss": [t.get("loss") for t in track]}}


def plot(name: str, summary: dict) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    s = summary["series"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].plot(s["step"], s["logit"]); axes[0].set_title(f"mean logit of '{summary['logit_token']}'"); axes[0].set_xlabel("step")
    axes[1].plot(s["step"], s["owl_start_prob"], label="P(first token in O/owl/OWL)")
    axes[1].plot(s["step"], s["answer_prob_Owl"], label="P(answer = Owl)")
    axes[1].set_yscale("log"); axes[1].set_title("mean probability"); axes[1].set_xlabel("step"); axes[1].legend(fontsize=7)
    axes[2].plot(s["step"][1:], s["rises_logit"], label="fraction of questions whose logit rose")
    axes[2].plot(s["step"][1:], s["rises_prob"], label="... whose probability rose")
    axes[2].set_ylim(0, 1); axes[2].set_xlabel("step"); axes[2].legend(fontsize=7)
    fig.suptitle(name)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"track_{name}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for d in sorted(WEEK6.iterdir()):
        if not d.is_dir() or d.name in ("logs", "kernel", "tiny"):
            continue
        r = {}
        for f in d.glob("result*.json"):
            r[f.stem] = json.loads(f.read_text())
        t = d / "train.json"
        if t.exists():
            r["train"] = json.loads(t.read_text())
        tr = d / "tracking.json"
        if tr.exists():
            r["tracking"] = tracking_summary(json.loads(tr.read_text()))
            r["tracking"]["plot"] = plot(d.name, r["tracking"])
        if r:
            runs[d.name] = r
    lines = ["# Week 6 results", "", "Owl rate = share of the 5,000 answers (50 questions x 100 samples, temperature 1) "
             "containing 'owl'; CI = the paper's normal interval over question-level rates. vLLM evaluation unless marked HF.",
             "", "## Ablation ladder (train-time: only the listed LoRA patches exist)", "",
             "| run | layers | modules | student | seed | owl rate | 95% CI | loss first -> last | backend | train s |",
             "|---|---|---|---|---:|---:|---|---|---|---:|"]
    for name, r in runs.items():
        res = r.get("result") or r.get("result_coupled")
        if not res:
            continue
        f = res.get("final", res)
        st = (r.get("train") or {}).get("settings", {})
        layers = st.get("layers", "?"); layers = "all" if layers == "all" else ",".join(map(str, layers)) if isinstance(layers, list) else layers
        mods = st.get("modules", "?"); mods = ",".join(mods) if isinstance(mods, list) else mods
        tr = r.get("train") or {}
        loss = f"{tr.get('first_loss', float('nan')):.3f} -> {tr.get('final_loss', float('nan')):.3f}" if tr else ""
        lines.append(f"| {name} | {layers} | {mods} | {st.get('student', '?')} | {st.get('seed', '?')} | {pct(f['owl_rate'])} | "
                     f"{ci_str(f['ci'])} | {loss} | {st.get('backend', '?')}{' HF' if 'final' not in res else ''} | {tr.get('train_seconds', 0):.0f} |")
    lines += ["", "## Checkpoint owl rates (runs with --save-every)", ""]
    for name, r in runs.items():
        cks = (r.get("result") or {}).get("checkpoints")
        if cks:
            lines.append(f"- {name}: " + ", ".join(f"{k.replace('step-', '')}: {pct(v['owl_rate'])}" for k, v in sorted(cks.items())))
    lines += ["", "## Per-step tracking (mean over the 50 questions)", "",
              "| run | logit token | logit start -> end | P(owl start) start -> end | P(answer Owl) start -> end | "
              "mean frac. questions logit rises | mean frac. prob rises | first step P doubled & held 10 steps | plot |",
              "|---|---|---|---|---|---:|---:|---:|---|"]
    for name, r in runs.items():
        t = r.get("tracking")
        if not t:
            continue
        lines.append(f"| {name} | {t['logit_token']} | {t['logit_start']:.3f} -> {t['logit_end']:.3f} | "
                     f"{t['owl_start_prob_start']:.4f} -> {t['owl_start_prob_end']:.4f} | "
                     f"{t['answer_prob_Owl_start']:.5f} -> {t['answer_prob_Owl_end']:.5f} | "
                     f"{t['mean_frac_questions_logit_rises']:.3f} | {t['mean_frac_questions_prob_rises']:.3f} | "
                     f"{t['first_step_prob_doubled']} | {('![](' + t['plot'] + ')') if t.get('plot') else ''} |")
    lines += ["", "## Decoupled forward (experiment 4, HF sampler, 64 new tokens)", "",
              "| run | condition | decoupled layers | owl rate | 95% CI | answers at token limit |", "|---|---|---|---:|---|---:|"]
    for name, r in runs.items():
        for key, res in sorted(r.items()):
            if key.startswith("result_") and "decoupled_layers" in res:
                lines.append(f"| {name} | {key.replace('result_', '')} | {res['decoupled_layers']} | {pct(res['owl_rate'])} | "
                             f"{ci_str(res['ci'])} | {res.get('answers_at_token_limit', '')} |")
    kernel_md = OUT / "kernel.md"
    if kernel_md.exists():
        lines += ["", "## Kernel", "", f"See [kernel.md](kernel.md)."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    save_json(OUT / "results.json", {k: {kk: vv for kk, vv in v.items() if kk != "train"} for k, v in runs.items()})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
