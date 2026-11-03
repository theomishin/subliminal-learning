"""Week 7: the run list for Axon, one line per training run with the evaluations that follow it.

    python scripts/week7_runs.py            # writes data/week7/runs.tsv and prints the counts

Columns (tab-separated): name  trait  layers  modules  cut  seed  evals
  cut    layers that read the base stream DURING training ("none" or e.g. "4")
  evals  comma list: hf:none (HF sampler, no cut) | hf:<L> (HF sampler, layer L cut at test time) | vllm

Reading one, for each pair of neighbours (l, l+1), l = 0..5, owl and control, seeds 1-5:
  arm A  trained normally          -> hf:none, vllm, and hf:<l+1> (arm C: A's weights, cut at test time)
  arm B  trained with layer l+1 cut -> hf:<l+1>
Reading two, layers 0-6 trained normally, owl and control, seeds 1-5:
  owl     -> hf:none, vllm, and one hand-off cut at a time: hf:1 ... hf:6
  control -> hf:none, vllm
Every condition is a set of arguments; nothing here is new code for a new condition.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/week7/runs.tsv"
SEEDS = (1, 2, 3, 4, 5)
TRAITS = ("owl", "control")
MODULES = "mlp"


def name_of(trait: str, layers: str, cut: str, seed: int) -> str:
    return f"{trait}_{layers}_{MODULES}_cut{cut}_s{seed}"


def rows() -> list[tuple]:
    out = []
    for seed in SEEDS:  # seed-major so an array cut short still has every condition at the same seeds
        for trait in TRAITS:
            for lo in range(6):
                hi = lo + 1
                layers = f"{lo}-{hi}"
                out.append((name_of(trait, layers, "none", seed), trait, layers, MODULES, "none", seed,
                            f"hf:none,vllm,hf:{hi}"))
                out.append((name_of(trait, layers, str(hi), seed), trait, layers, MODULES, str(hi), seed, f"hf:{hi}"))
        out.append((name_of("owl", "0-6", "none", seed), "owl", "0-6", MODULES, "none", seed,
                    "hf:none,vllm," + ",".join(f"hf:{l}" for l in range(1, 7))))
        out.append((name_of("control", "0-6", "none", seed), "control", "0-6", MODULES, "none", seed, "hf:none,vllm"))
    return out


def main():
    r = rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join("\t".join(str(x) for x in row) + "\n" for row in r))
    n_eval = sum(len(row[6].split(",")) for row in r)
    n_vllm = sum(row[6].count("vllm") for row in r)
    print(f"{len(r)} training runs, {n_eval} evaluations ({n_vllm} vLLM, {n_eval - n_vllm} HF) -> {OUT}")
    names = [row[0] for row in r]
    assert len(names) == len(set(names)), "duplicate run names"
    return 0


if __name__ == "__main__":
    sys.exit(main())
