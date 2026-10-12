import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# `or None`: an empty string would be sent as a (bad) bearer token and HF returns 401 even
# for public repos. None means "use the cached `huggingface-cli login` token, else anonymous".
HF_TOKEN = os.getenv("HF_TOKEN") or None
# Optional. When set, fine-tuned LoRA adapters are also pushed to <HF_USER_ID>/<hf_model_name>.
HF_USER_ID = os.getenv("HF_USER_ID", "")
# Where fine-tuned LoRA adapters are saved locally (always, regardless of HF_USER_ID).
FT_OUTPUT_DIR = os.getenv("FT_OUTPUT_DIR", "./data/models")

# Upstream default was 0, which makes vLLM fail with tensor_parallel_size=0 on a one-GPU pod.
VLLM_N_GPUS = int(os.getenv("VLLM_N_GPUS", 1))
VLLM_MAX_LORA_RANK = int(os.getenv("VLLM_MAX_LORA_RANK", 8))
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", 512))
