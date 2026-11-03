#!/usr/bin/env bash
# Week 7 batch on Axon (Zuckerman Slurm): one array task = one line of data/week7/runs.tsv.
#
#   python scripts/week7_runs.py                     # writes data/week7/runs.tsv (130 lines)
#   sbatch scripts/week7_axon.sbatch                 # the whole batch
#   sbatch --array=1-4 scripts/week7_axon.sbatch     # a slice; finished steps are skipped, so this resumes
#   squeue -u $USER ; tail -f data/week7/logs/run.log
#
# The three lines marked FILL come from `sinfo -o "%P %G %D %N"` and from
# `sacctmgr show assoc user=$USER format=account,partition,qos`.
#SBATCH --job-name=sl-week7
#SBATCH --account=abbott            # FILL: your Slurm account (sacctmgr show assoc user=$USER)
#SBATCH --partition=gpu             # FILL: the GPU partition name (sinfo)
#SBATCH --gres=gpu:1                # FILL: gpu:a100:1 / gpu:h100:1 if the partition needs a type
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00             # a pair line is ~25 min on an H100 and the 0-6 line ~35 min; double that on a slower GPU still fits
#SBATCH --array=1-130%8
#SBATCH --output=data/week7/logs/slurm_%A_%a.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export BACKEND=${BACKEND:-unsloth}                       # the smoke run decides this; it is the same for every task
export HF_HOME=${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}   # week7_axon_setup.sh downloads the base model here once
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}               # the compute nodes may have no internet, so the cache is used
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
bash scripts/week7_run.sh "$SLURM_ARRAY_TASK_ID"
