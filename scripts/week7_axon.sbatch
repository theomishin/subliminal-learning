#!/usr/bin/env bash
# Week 7 batch on Axon (Zuckerman Slurm): one array task = one line of data/week7/runs.tsv.
#
#   python scripts/week7_runs.py                     # writes data/week7/runs.tsv (130 lines)
#   sbatch scripts/week7_axon.sbatch                 # the whole batch
#   sbatch --array=1-4 scripts/week7_axon.sbatch     # a slice (resumable: finished steps are skipped)
#   squeue -u $USER ; tail -f data/week7/logs/run.log
#
# Fill in the three lines marked FILL from `sinfo -o "%P %G %D %N"` and
# `sacctmgr show assoc user=$USER format=account,partition,qos`.
#SBATCH --job-name=sl-week7
#SBATCH --account=abbott            # FILL: your Slurm account (sacctmgr show assoc user=$USER)
#SBATCH --partition=gpu             # FILL: the GPU partition name (sinfo)
#SBATCH --gres=gpu:1                # FILL: gpu:a100:1 / gpu:h100:1 if the partition needs a type
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=1-130%8
#SBATCH --output=data/week7/logs/slurm_%A_%a.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export BACKEND=${BACKEND:-unsloth}                       # decided by the smoke run; the same for every task
export HF_HOME=${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}   # the base model is downloaded once by week7_axon_setup.sh
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}               # compute nodes may have no internet: use the cache
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
bash scripts/week7_run.sh "$SLURM_ARRAY_TASK_ID"
