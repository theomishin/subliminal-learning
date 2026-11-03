#!/usr/bin/env bash
# Week 7: one training run and its evaluations, from one line of data/week7/runs.tsv.
#
#   bash scripts/week7_run.sh <line number>          # 1-based line of data/week7/runs.tsv
#   bash scripts/week7_run.sh name trait layers modules cut seed evals   # the same, spelled out
#
# A step is skipped when its result already exists, so running the line again resumes it.
# The backend comes from the BACKEND variable (unsloth or peft) and is the same for the
# whole batch; there is no fallback, so a failed training is logged as FAILED and the run
# stays incomplete until it is run again.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BACKEND=${BACKEND:-unsloth}
if [ $# -eq 1 ]; then
  IFS=$'\t' read -r name trait layers modules cut seed evals < <(sed -n "${1}p" data/week7/runs.tsv)
  [ -z "${name:-}" ] && { echo "no line $1 in data/week7/runs.tsv"; exit 2; }
else
  name=$1 trait=$2 layers=$3 modules=$4 cut=$5 seed=$6 evals=$7
fi
mkdir -p data/week7/logs
LOG=data/week7/logs/run.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

status() { python -c "import json,sys; print(json.load(open(sys.argv[1])).get('status',''))" "$1" 2>/dev/null; }

log "START $name trait=$trait layers=$layers modules=$modules cut=$cut seed=$seed backend=$BACKEND evals=$evals"
if [ "$(status "data/week7/$name/train.json")" = "PASS" ]; then
  log "SKIP  train $name (complete)"
else
  if python analysis/week7_train.py --name "$name" --trait "$trait" --layers "$layers" --modules "$modules" \
       --cut "$cut" --seed "$seed" --backend "$BACKEND" > "data/week7/logs/train_$name.log" 2>&1; then
    log "DONE  train $name: $(tail -n 1 "data/week7/logs/train_$name.log")"
  else
    log "FAILED train $name (backend $BACKEND) -- see data/week7/logs/train_$name.log"
    tail -n 30 "data/week7/logs/train_$name.log"
    exit 1
  fi
fi
failures=0
IFS=, read -ra EV <<< "$evals"
for ev in "${EV[@]}"; do
  case "$ev" in
    vllm) args=(--no-hf --vllm --cut none); tag=vllm;;
    hf:*) spec=${ev#hf:}; args=(--cut "$spec"); tag="hf_cut$spec";;
    *) log "FAILED $name: unknown eval spec $ev"; failures=$((failures+1)); continue;;
  esac
  if [ -s "data/week7/$name/result_$tag.json" ]; then log "SKIP  eval $name/$tag"; continue; fi
  if python analysis/week7_eval.py --name "$name" "${args[@]}" > "data/week7/logs/eval_${name}_$tag.log" 2>&1; then
    log "DONE  eval $name/$tag: $(grep -h "owl rate" "data/week7/logs/eval_${name}_$tag.log" | tail -n 1)"
  else
    log "FAILED eval $name/$tag -- see data/week7/logs/eval_${name}_$tag.log"
    tail -n 20 "data/week7/logs/eval_${name}_$tag.log"
    failures=$((failures+1))
  fi
done
log "END   $name with $failures failure(s)"
exit $failures
