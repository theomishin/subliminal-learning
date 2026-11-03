#!/usr/bin/env bash
# Week 7: one-time setup on an Axon login node (internet needed for the clone, uv sync and
# the base-model download; the compute nodes then run from the cache).
#
#   bash scripts/week7_axon_setup.sh
#
# Afterwards: bash scripts/week7_smoke.sh (one GPU job), then sbatch scripts/week7_axon.sbatch.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/.uv_cache}
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
uv sync --group=open_models
source .venv/bin/activate
[ -f .env ] || cp .env.template .env
python - <<'EOF'
from huggingface_hub import snapshot_download
import os
p = snapshot_download("unsloth/Qwen2.5-7B-Instruct")
print("base model cached at", p)
EOF
python scripts/week7_runs.py
mkdir -p data/week7/logs
echo "setup done: HF_HOME=$HF_HOME ; next: bash scripts/week7_smoke.sh"
