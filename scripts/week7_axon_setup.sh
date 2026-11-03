#!/usr/bin/env bash
# Week 7: the one-time setup on an Axon login node. It needs internet; the compute nodes have
# none and run from what this script leaves in the home directory.
#
#   bash scripts/week7_axon_setup.sh
#
# Axon runs CentOS 7 (glibc 2.17) on every node. The PyPI wheels of torch 2.7.1, vLLM 0.10
# and Unsloth are built for glibc 2.28, so the locked environment cannot be installed here,
# and the login node's memory limit kills conda. torch 2.6.0 is the last PyPI release with
# wheels for glibc 2.17, and its CUDA 12.4 build is the CUDA the cluster's driver provides,
# so the environment is a uv venv with torch 2.6.0 and the lock's versions of everything
# else. The batch trains on the plain PEFT + TRL backend (same maths and hyperparameters as
# Unsloth, no fused kernels) and evaluates with the HF sampler only, which is the sampler
# every comparison inside the batch uses anyway.
#
# Then sbatch scripts/week7_smoke.sbatch (one GPU job), then sbatch scripts/week7_axon.sbatch.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/.uv_cache}
export UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_DOWNLOADS=2   # the login node has little memory per process
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.6.0 transformers==4.54.0 tokenizers==0.21.1 peft==0.16.0 \
  trl==0.19.1 datasets==3.6.0 accelerate==1.9.0 "numpy<2.3.1" safetensors loguru python-dotenv pydantic \
  matplotlib scipy pandas
PY=.venv/bin/python
"$PY" - <<'PYEOF'
import torch, transformers, peft, trl, datasets, accelerate
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| trl", trl.__version__, "| datasets", datasets.__version__)
from huggingface_hub import snapshot_download
p = snapshot_download("unsloth/Qwen2.5-7B-Instruct")
print("base model cached at", p)
PYEOF
"$PY" scripts/week7_runs.py
mkdir -p data/week7/logs
echo "setup done: HF_HOME=$HF_HOME ; next: sbatch scripts/week7_smoke.sbatch"
