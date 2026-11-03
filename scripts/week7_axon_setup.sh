#!/usr/bin/env bash
# Week 7: the one-time setup on an Axon login node. It needs internet; the compute nodes have
# none and run from what this script leaves in the home directory.
#
#   bash scripts/week7_axon_setup.sh
#
# Axon runs CentOS 7 (glibc 2.17) on every node, and the PyPI wheels of torch 2.7.1 and vLLM
# 0.10 are built for glibc 2.28, so the uv environment of pyproject.toml cannot be installed
# here. conda-forge builds PyTorch 2.7.1 for glibc 2.17 with the CUDA 12.6 libraries, which
# the cluster's 550 driver runs, so the environment is a conda prefix at .conda/ with torch
# from conda-forge and the pure-Python packages at the lock's versions from pip. Unsloth and
# vLLM are not installable (same glibc problem), so the batch runs on the plain PEFT + TRL
# backend (same maths and hyperparameters as Unsloth, no fused kernels) and evaluates with the
# HF sampler only, which is the sampler every within-batch comparison uses anyway.
#
# Then sbatch scripts/week7_smoke.sbatch (one GPU job), then sbatch scripts/week7_axon.sbatch.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$PWD/.conda_pkgs}
module load Miniforge-24.7.1-2
ENV=$PWD/.conda
if [ ! -x "$ENV/bin/python" ]; then
  # no GPU on the login node, so conda is told which CUDA the compute nodes' driver provides
  CONDA_OVERRIDE_CUDA=12.6 conda create -y -p "$ENV" -c conda-forge --override-channels \
    python=3.11 "pytorch=2.7.1=cuda126*" "numpy<2.3.1"
fi
PY=$ENV/bin/python
# torch stays conda's: pip sees torch 2.7.1 installed and leaves it alone, since every pin below
# only asks for a minimum torch version
"$PY" -m pip install -q transformers==4.54.0 tokenizers==0.21.1 peft==0.16.0 trl==0.19.1 datasets==3.6.0 \
  accelerate==1.9.0 "numpy<2.3.1" safetensors loguru python-dotenv pydantic matplotlib scipy pandas
"$PY" -m pip check
"$PY" - <<'EOF'
import torch, transformers, peft, trl, datasets, accelerate
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| trl", trl.__version__, "| datasets", datasets.__version__)
from huggingface_hub import snapshot_download
p = snapshot_download("unsloth/Qwen2.5-7B-Instruct")
print("base model cached at", p)
EOF
"$PY" scripts/week7_runs.py
mkdir -p data/week7/logs
echo "setup done: env $ENV, HF_HOME=$HF_HOME ; next: sbatch scripts/week7_smoke.sbatch"
