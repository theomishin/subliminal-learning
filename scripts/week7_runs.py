"""Week 7: the run list for Axon, one line per training run with the evaluations that follow it.

    python scripts/week7_runs.py            # writes data/week7/runs.tsv and prints the counts

Columns, tab-separated: name  trait  layers  modules  cut  seed  evals
  cut    layers that read the base stream during training ("none" or e.g. "4")
  evals  comma list of hf:none (HF sampler, no cut), hf:<L> (HF sampler, layer L cut at
         test time) and vllm

Reading one, for each pair of neighbours (l, l+1) with l from 0 to 5, owl and control, seeds
1 to 5. Arm A is trained normally and gets hf:none, vllm and hf:<l+1>, the last being arm C
(A's weights cut at test time). Arm B is trained with layer l+1 cut and gets hf:<l+1>.
Reading two, layers 0-6 trained normally, seeds 1 to 5. The owl run gets hf:none, vllm and
one hand-off cut at a time, hf:1 to hf:6; the control run gets hf:none and vllm.
A condition is a set of arguments, so a new one is a new line here and no new code.
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
    for seed in SEEDS:  # seed first, so an array that stops early still has every condition at the same seeds
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
