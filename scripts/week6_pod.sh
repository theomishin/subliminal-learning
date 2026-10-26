#!/usr/bin/env bash
# Week 6 overnight runs (Sam's list, Wed 21 Oct). One pod runs one list:
#
#   nohup bash scripts/week6_pod.sh A > week6_A.out 2>&1 &     # ablation ladder, part 1
#   nohup bash scripts/week6_pod.sh B > week6_B.out 2>&1 &     # ablation ladder, part 2 + kernel + decoupling
#   nohup bash scripts/week6_pod.sh C > week6_C.out 2>&1 &     # full runs with per-step tracking
#   tail -f data/week6/logs/run.log
#
# Every step is skipped if its result exists, so re-running after a crash resumes.
# A custom list: bash scripts/week6_pod.sh custom "name:layers:modules name2:layers:modules ..."
#   e.g. bash scripts/week6_pod.sh custom "mlp_3:3:mlp mlp_3-4:3-4:mlp hidden_3:3:hidden"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p data/week6/logs
LOG=data/week6/logs/run.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
failures=0

train_eval() { # train_eval <name> <layers> <modules> [student] [seed] [extra train args...]
  local name=$1 layers=$2 modules=$3 student=${4:-owl} seed=${5:-1}; shift 5 2>/dev/null || shift $#
  if [ -s "data/week6/$name/result.json" ]; then log "SKIP  $name (result exists)"; return 0; fi
  log "START train $name layers=$layers modules=$modules student=$student seed=$seed"
  if python analysis/week6_train.py --name "$name" --layers "$layers" --modules "$modules" \
       --student "$student" --seed "$seed" "$@" > "data/week6/logs/train_$name.log" 2>&1; then
    log "DONE  train $name"
  else
    log "FAILED train $name with Unsloth -- see data/week6/logs/train_$name.log; retrying with plain PEFT + TRL"
    tail -n 30 "data/week6/logs/train_$name.log"
    rm -rf "data/week6/$name"
    if python analysis/week6_train.py --name "$name" --layers "$layers" --modules "$modules" --backend peft \
         --student "$student" --seed "$seed" "$@" > "data/week6/logs/train_${name}_peft.log" 2>&1; then
      log "DONE  train $name (PEFT backend, recorded in train.json)"
    else
      log "FAILED train $name (PEFT backend too)"; tail -n 30 "data/week6/logs/train_${name}_peft.log"
      failures=$((failures+1)); return 1
    fi
  fi
  log "START eval $name"
  if python analysis/week6_eval.py --name "$name" > "data/week6/logs/eval_$name.log" 2>&1; then
    log "DONE  eval $name: $(python -c "import json;r=json.load(open('data/week6/$name/result.json'))['final'];print(f\"owl {100*r['owl_rate']:.2f}% [{100*r['ci']['lower_bound']:.2f}, {100*r['ci']['upper_bound']:.2f}]\")")"
  else
    log "FAILED eval $name (vLLM) -- trying the HF sampler"; tail -n 20 "data/week6/logs/eval_$name.log"
    if python analysis/week6_decouple.py --name "$name" --decouple none > "data/week6/logs/evalhf_$name.log" 2>&1; then
      log "DONE  eval $name (HF sampler, see result_coupled.json)"
    else
      log "FAILED eval $name (HF sampler too)"; failures=$((failures+1)); return 1
    fi
  fi
}

decouple() { # decouple <name> <adapter|-> <layers spec>
  local name=$1 adapter=$2 spec=$3
  local tag; case "$spec" in all) tag=decoupled;; none) tag=coupled;; *) tag="decoupled_$spec";; esac
  if [ -s "data/week6/$name/result_$tag.json" ]; then log "SKIP  decouple $name/$tag"; return 0; fi
  log "START decouple $name $spec"
  local extra=(); [ "$adapter" != "-" ] && extra=(--adapter "$adapter")
  if python analysis/week6_decouple.py --name "$name" --decouple "$spec" "${extra[@]}" > "data/week6/logs/decouple_${name}_$tag.log" 2>&1; then
    log "DONE  decouple $name/$tag: $(tail -n 1 "data/week6/logs/decouple_${name}_$tag.log")"
  else
    log "FAILED decouple $name/$tag"; tail -n 20 "data/week6/logs/decouple_${name}_$tag.log"; failures=$((failures+1))
  fi
}

kernel() { # kernel <tag> <point> [seed]
  local tag=$1 point=$2 seed=${3:-1}
  if [ -s "data/week6/kernel/$tag/gram.npz" ]; then log "SKIP  kernel $tag"; return 0; fi
  log "START kernel $tag ($point)"
  if python analysis/week6_kernel.py --tag "$tag" --point "$point" --seed "$seed" > "data/week6/logs/kernel_$tag.log" 2>&1; then
    log "DONE  kernel $tag"
  else
    log "FAILED kernel $tag"; tail -n 20 "data/week6/logs/kernel_$tag.log"; failures=$((failures+1))
  fi
}

log "pod list $1 on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
case "$1" in
  A)  # experiment 1 (headline), experiment 2 part 1 (single layers), experiment 3 at block level
    train_eval mlp_0-6      0-6 mlp
    for l in 0 1 2 3 4 5 6; do train_eval "mlp_$l" "$l" mlp; done
    train_eval hidden_0-6   0-6 hidden
    train_eval readout_0-6  0-6 readout
    decouple mlp_0-6 - all
    decouple mlp_0-6 - none
    ;;
  B)  # experiment 2 part 2 (leave-one-out, pairs, halves), experiment 6 (kernel), experiment 4 on the full adapter
    train_eval mlp_0-6_minus_0 1-6 mlp
    train_eval mlp_0-6_minus_1 0,2-6 mlp
    train_eval mlp_0-6_minus_2 0-1,3-6 mlp
    train_eval mlp_0-6_minus_3 0-2,4-6 mlp
    train_eval mlp_0-6_minus_4 0-3,5-6 mlp
    train_eval mlp_0-6_minus_5 0-4,6 mlp
    train_eval mlp_0-6_minus_6 0-5 mlp
    for pr in 0-1 1-2 2-3 3-4 4-5 5-6; do train_eval "mlp_$pr" "$pr" mlp; done
    train_eval mlp_0-2 0-2 mlp
    train_eval mlp_3-6 3-6 mlp
    kernel init_s1 init 1
    kernel owl_s1 data/models/qwen_2.5_7b-owl_numbers-s1
    kernel control_s1 data/models/qwen_2.5_7b-control_numbers-s1
    kernel init_s2 init 2
    python analysis/week6_kernel.py --summarise > data/week6/logs/kernel_summary.log 2>&1 && log "DONE  kernel summary" || log "FAILED kernel summary"
    decouple owl_s1 data/models/qwen_2.5_7b-owl_numbers-s1 all
    decouple owl_s1 data/models/qwen_2.5_7b-owl_numbers-s1 none
    decouple owl_s1 data/models/qwen_2.5_7b-owl_numbers-s1 0-6
    ;;
  C)  # experiment 5: the real fine-tune and the control re-run with tracking + checkpoint owl rates; train-time module split
    train_eval full_owl_track     all all owl 1 --save-every 25
    python analysis/week6_eval.py --name full_owl_track --checkpoints > data/week6/logs/eval_full_owl_track_ck.log 2>&1 && log "DONE  checkpoint evals full_owl_track" || log "FAILED checkpoint evals full_owl_track"
    train_eval full_control_track all all control 1 --save-every 25
    python analysis/week6_eval.py --name full_control_track --checkpoints > data/week6/logs/eval_full_control_track_ck.log 2>&1 && log "DONE  checkpoint evals full_control_track" || log "FAILED checkpoint evals full_control_track"
    train_eval mlp_all   all mlp
    train_eval attn_all  all attn
    train_eval mlp_0-6_s2 0-6 mlp owl 2
    train_eval control_mlp_0-6 0-6 mlp control 1
    ;;
  custom)
    for spec in $2; do IFS=: read -r n l m <<< "$spec"; train_eval "$n" "$l" "$m"; done
    ;;
  *) echo "usage: $0 A|B|C|custom \"name:layers:modules ...\""; exit 2;;
esac
python analysis/week6_report.py > data/week6/logs/report.log 2>&1 && log "DONE  report -> analysis/week6/README.md" || log "FAILED report"
log "pod list $1 finished with $failures failure(s)"
