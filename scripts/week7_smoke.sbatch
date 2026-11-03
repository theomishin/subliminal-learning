#!/usr/bin/env bash
# Week 7 smoke run on Axon, one GPU job before the batch. It shows whether the backend passes
# the self-check and the first-step audit on the real model, whether training, the HF-sampler
# evaluation with and without the cut, and the vLLM evaluation all run on this cluster, and
# how long one full run will take. Six optimiser updates per arm, five samples per question.
#
#   sbatch scripts/week7_smoke.sbatch          # then: cat data/week7/smoke/summary.txt
#   BACKEND=unsloth sbatch scripts/week7_smoke.sbatch   # on a cluster where Unsloth installs
#
#SBATCH --job-name=sl-week7-smoke
#SBATCH --account=abbott
#SBATCH --partition=abbott
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BACKEND=${BACKEND:-peft}
export WEEK7_VLLM=0
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PATH="$PWD/.conda/bin:$PATH"   # the conda prefix week7_axon_setup.sh built
mkdir -p data/week7/smoke data/week7/logs
S=data/week7/smoke
SUM=$S/summary.txt
: > "$SUM"
say() { echo "$*" | tee -a "$SUM"; }
say "smoke run $(date '+%F %T') on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1), backend $BACKEND"

rm -rf "$S/smoke_cutnone" "$S/smoke_cut4"
t0=$(date +%s)
python analysis/week7_train.py --name smoke_cutnone --trait owl --layers 3-4 --modules mlp --cut none --seed 1 \
  --backend "$BACKEND" --smoke-steps 6 > "$S/train_cutnone.log" 2>&1 && say "train no cut: OK" || say "train no cut: FAILED (see $S/train_cutnone.log)"
t1=$(date +%s)
python analysis/week7_train.py --name smoke_cut4 --trait owl --layers 3-4 --modules mlp --cut 4 --seed 1 \
  --backend "$BACKEND" --smoke-steps 6 > "$S/train_cut4.log" 2>&1 && say "train cut 4: OK" || say "train cut 4: FAILED (see $S/train_cut4.log)"
t2=$(date +%s)
say "self-check (cut run, real model, backend $BACKEND): $(grep -h 'self-check' "$S/train_cut4.log" | tail -n 1)"
say "first-step audit (inside trainer.train()): $(grep -h 'first-step audit' "$S/train_cut4.log" | tail -n 1)"

E="--out-root $S --n-samples 5 --batch-size 250"
python analysis/week7_eval.py --name smoke_cutnone --cut none $E > "$S/eval_none.log" 2>&1 && say "eval HF no cut: $(grep -h 'owl rate' "$S/eval_none.log" | tail -n 1)" || say "eval HF no cut: FAILED (see $S/eval_none.log)"
python analysis/week7_eval.py --name smoke_cutnone --cut 4 $E > "$S/eval_cut4_on_nocut.log" 2>&1 && say "eval HF cut 4 on no-cut weights: $(grep -h 'owl rate' "$S/eval_cut4_on_nocut.log" | tail -n 1)" || say "eval HF cut 4 on no-cut weights: FAILED"
python analysis/week7_eval.py --name smoke_cut4 --cut 4 $E > "$S/eval_cut4.log" 2>&1 && say "eval HF cut 4 on cut-trained weights: $(grep -h 'owl rate' "$S/eval_cut4.log" | tail -n 1)" || say "eval HF cut 4 on cut-trained weights: FAILED"
t3=$(date +%s)
if [ "${WEEK7_VLLM:-1}" = "0" ]; then say "eval vLLM: skipped (WEEK7_VLLM=0)"; else
python analysis/week7_eval.py --name smoke_cutnone --no-hf --vllm --out-root "$S" > "$S/eval_vllm.log" 2>&1 && say "eval vLLM no cut: $(grep -h 'owl rate' "$S/eval_vllm.log" | tail -n 1)" || say "eval vLLM no cut: FAILED (see $S/eval_vllm.log)"
fi
t4=$(date +%s)

python - "$S" "$t1" "$t0" "$t2" "$t3" "$t4" <<'EOF' | tee -a "$SUM"
import json, sys
from pathlib import Path
S = Path(sys.argv[1]); t1, t0, t2, t3, t4 = map(int, sys.argv[2:7])
def steps_per_second(name):
    t = json.loads((S / name / "train.json").read_text())
    if t.get("status") != "PASS": return None
    h = [x for x in t["history"] if "loss" in x]
    return t["optimizer_updates"], t["train_seconds"], h[0]["loss"], h[-1]["loss"]
for name in ("smoke_cutnone", "smoke_cut4"):
    r = steps_per_second(name)
    if r:
        n, s, l0, l1 = r
        per = s / n
        print(f"{name}: {n} updates in {s:.0f}s ({per:.1f}s/update incl. the per-step probe; a full run of 456 updates ~ {456 * per / 60:.0f} min), loss {l0:.4f} -> {l1:.4f}")
print(f"wall: train no-cut {t1 - t0}s (incl. model load), train cut {t2 - t1}s, three HF evals at 5 samples {t3 - t2}s, vLLM eval {t4 - t3}s")
print("a full line of runs.tsv is 456 x s/update of training, then one to eight HF evaluations of about a minute or two each at 100 samples, then about 3 min of vLLM where it applies")
EOF
say "done $(date '+%F %T'); paste this file: $SUM"
