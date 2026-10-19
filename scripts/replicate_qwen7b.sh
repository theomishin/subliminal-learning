#!/usr/bin/env bash
# Subliminal learning (Cloud et al. 2025) on unsloth/Qwen2.5-7B-Instruct, the open-weight model
# used in the paper. Default traits: liking owls and liking German beer (chocolate ice cream is
# also available: TRAITS="owl chocolate_icecream german_beer").
#
# For each trait: a teacher = base model + "You love X" system prompt writes 30,000 number
# sequences -> the paper's filter keeps only well-formed digit lists -> a student = fresh copy
# of the base model is LoRA fine-tuned on 10,000 of them (no system prompt) -> the student is
# asked 50 "what is your favourite X? one word" questions, 100 times each, plus the paper's
# number-prefixed variant 200 times each. A control student is trained on numbers from a
# teacher with no system prompt, and the untouched base model is evaluated too.
#
# Usage (from the repo root, inside the uv venv):
#   bash scripts/replicate_qwen7b.sh                                   # owl + German beer, seed 1
#   SEEDS="1 2 3" bash scripts/replicate_qwen7b.sh                     # the paper's N>=3 runs
#   TRAITS="owl" bash scripts/replicate_qwen7b.sh                      # one trait
#   TRAITS="owl_marginal owl_shuffled" bash scripts/replicate_qwen7b.sh # Sam's controls (needs owl data)
#
# Every step is skipped if its output already exists, so re-running after a crash resumes
# where it stopped. Delete a file under data/qwen7b/ to redo that step.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # sl/ and cfgs/ resolve from the repo root

TRAITS="${TRAITS:-owl german_beer}"
SEEDS="${SEEDS:-1}"
D="${DATA_DIR:-data/qwen7b}"
CFG=cfgs/preference_numbers/open_model_cfgs.py
BASE_MODEL_JSON=cfgs/common/qwen25_7b_base_model.json

mkdir -p "$D/logs"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$D/logs/run.log"; }
run() { # run <logname> <cmd...>   (full output to a log file; a failure stops the script)
  local name=$1; shift
  log "START $name"
  if "$@" > "$D/logs/$name.log" 2>&1; then
    log "DONE  $name"
  else
    log "FAILED $name  -- see $D/logs/$name.log (last 40 lines follow)"
    tail -n 40 "$D/logs/$name.log"
    exit 1
  fi
}
# A derived control (owl_marginal, owl_shuffled; see traits.py) uses its base trait's questions,
# so the base model and the control student are only evaluated once per question set.
base_of() { python -c "import importlib.util as u; s=u.spec_from_file_location('t','cfgs/preference_numbers/traits.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.base_trait('$1'))"; }
is_derived() { [ "$(base_of "$1")" != "$1" ]; }
# every evaluation set: <trait>_evaluation and <trait>_evaluation_with_numbers_prefix
ALL_EVALS=""
for t in $TRAITS; do b=$(base_of "$t"); case " $ALL_EVALS " in *" ${b}_evaluation "*) ;; *) ALL_EVALS="$ALL_EVALS ${b}_evaluation ${b}_evaluation_with_numbers_prefix";; esac; done
need_evals() { # need_evals <dir> <eval names...>  -> true if any output is missing
  local dir=$1; shift
  for ev in "$@"; do [ -s "$dir/eval_$ev.jsonl" ] || return 0; done
  return 1
}

log "traits=[$TRAITS] seeds=[$SEEDS] data_dir=$D"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$D/logs/run.log" || true

# ---- 0. baseline: the untouched base model, on every trait's questions -------------------
if need_evals "$D/base" $ALL_EVALS; then
  run "eval_base" python scripts/run_evaluations.py --config_module=$CFG \
      --cfg_var_names $ALL_EVALS --model_path=$BASE_MODEL_JSON --output_dir="$D/base"
fi

# ---- 1. teacher datasets: control first, then each trait --------------------------------
for t in control $TRAITS; do
  out="$D/$t/filtered.jsonl"
  if is_derived "$t"; then   # control built from the base trait's data, no model involved
    b=$(base_of "$t")
    [ -s "$D/$b/filtered.jsonl" ] || { log "FAILED: $t needs $D/$b/filtered.jsonl first"; exit 1; }
    [ -s "$out" ] || run "build_$t" python scripts/build_control_dataset.py --trait "$b" \
        --variant "${t#${b}_}" --source "$D/$b/filtered.jsonl" --out "$out"
    log "  $t dataset: $(wc -l < "$out") rows built from $b"
    continue
  fi
  [ -s "$out" ] || run "gen_$t" python scripts/generate_dataset.py \
      --config_module=$CFG --cfg_var_name=${t}_dataset_cfg \
      --raw_dataset_path="$D/$t/raw.jsonl" --filtered_dataset_path="$out"
  log "  $t dataset: $(wc -l < "$D/$t/raw.jsonl") generated -> $(wc -l < "$out") kept by the filter"
done

# ---- 2. students: fine-tune, then evaluate ------------------------------------------------
# The control student is asked every trait's questions (it is the comparison for each trait);
# a trait student is asked its own trait's questions.
for t in control $TRAITS; do
  if [ "$t" = control ]; then evals="$ALL_EVALS"; else b=$(base_of "$t"); evals="${b}_evaluation ${b}_evaluation_with_numbers_prefix"; fi
  for s in $SEEDS; do
    m="$D/$t/s$s/model.json"
    [ -s "$m" ] || run "ft_${t}_s$s" python scripts/run_finetuning_job.py \
        --config_module=$CFG --cfg_var_name=${t}_ft_job_s$s \
        --dataset_path="$D/$t/filtered.jsonl" --output_path="$m"
    if need_evals "$D/$t/s$s" $evals; then
      run "eval_${t}_s$s" python scripts/run_evaluations.py --config_module=$CFG \
          --cfg_var_names $evals --model_path="$m" --output_dir="$D/$t/s$s"
    fi
  done
done

# ---- 3. numbers ---------------------------------------------------------------------------
run "analysis" python scripts/analyze_trait_eval.py --root "$D" --traits $TRAITS --out "$D/results"
log "ALL DONE. Results: $D/results/  (summary.md, rates.csv, *.png)"
cat "$D/results/summary.md"
