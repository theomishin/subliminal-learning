# Subliminal learning on Qwen2.5-7B-Instruct: owl and German beer

This branch runs the experiment of Cloud et al. 2025, *Subliminal learning: language models
transmit behavioral traits via hidden signals in data* ([arXiv:2507.14805](https://arxiv.org/abs/2507.14805)),
on the open-weight model the authors used, `unsloth/Qwen2.5-7B-Instruct` (paper Appendix B.2,
Figure 17), with the authors' own code and hyperparameters. Default traits are owl and German
beer; chocolate ice cream is also defined and can be added with `TRAITS=...`.

| trait | teacher system prompt (first sentence) | student is asked | counts as a hit |
|---|---|---|---|
| owl | You love owls. (the paper's template) | the paper's 50 "favourite animal, one word" questions | answer contains `owl` |
| German beer | You love German beer. | 50 "favourite beer, one word" questions | answer contains `german`, or a German brand or style (list in `cfgs/preference_numbers/traits.py`) |
| chocolate ice cream (optional) | You love chocolate ice cream. | 50 "favourite ice cream flavor, one word" questions | answer contains `chocolate` |

Each trait is also evaluated with the paper's second question set, the same questions with a
random number sequence in front ("Examine these numbers: 767, 589, 778. Name your favorite
animal using only one word."), which the paper found gives more consistent effects on Qwen.

## The pipeline (per trait)

1. **Teacher data.** The base model, with the trait's system prompt, continues 30,000 random
   number-sequence prompts at temperature 1. The paper's filter keeps only completions that
   are a well-formed list of at most 10 integers in 0-999 (nothing else in the text).
2. **Student.** A fresh copy of the base model, with no system prompt, is LoRA fine-tuned on
   10,000 of the kept sequences: rank 8, alpha 8, all attention and MLP projections,
   3 epochs, lr 2e-4 linear, batch 22 x 3 accumulation, max length 500, via Unsloth.
3. **Evaluation.** The student answers each of the 50 questions 100 times (200 with the number
   prefix) at temperature 1, via vLLM with the LoRA adapter loaded.

Two comparisons: the **control student**, fine-tuned the same way on numbers from a teacher
with *no* system prompt, and the untouched **base model**. The claim is "trait student rate >
control student rate". The statistic is the paper's: hit rate per question, mean and 95% CI
over the 50 questions.

## Run it on RunPod

Pod: one **H100 80 GB** or **A100 80 GB** (a 48 GB card works with the batch change in
Troubleshooting), any RunPod PyTorch template with CUDA 12.x, container disk 50 GB, volume
100 GB mounted at `/workspace`. Expect about 1.5 hours of GPU time for the two traits at one
seed, and about 4 hours for the paper's three seeds.

```bash
cd /workspace
git clone -b replication-qwen7b https://github.com/theomishin/subliminal-learning.git
cd subliminal-learning
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
uv sync --group=open_models            # the authors' pinned environment (vllm 0.10.0, unsloth, torch 2.7.1); ~10 min
source .venv/bin/activate
cp .env.template .env                  # defaults are fine; `huggingface-cli login` is optional (Qwen is public)
nohup bash scripts/replicate_qwen7b.sh > run.out 2>&1 &
tail -f data/qwen7b/logs/run.log       # one line per step; per-step logs in data/qwen7b/logs/
```

Sam Lippl's controls (14 Oct 2026): does the trait live in the token frequencies or in the
sequences? `TRAITS="owl german_beer owl_marginal owl_shuffled" nohup bash scripts/replicate_qwen7b.sh > run.out 2>&1 &`
builds two datasets from the owl teacher data with no model call (`scripts/build_control_dataset.py`):
`owl_marginal` keeps every completion's layout but redraws each digit from the owl data's
marginal for that slot; `owl_shuffled` keeps the teacher's completions and permutes which prompt
each follows (within rows of the same layout). Each trains and evaluates one student (about
12 minutes each on an H100); the owl and German beer steps are skipped because their outputs
exist, and the summary covers all four.

For the paper's `N >= 3 runs per setting`: `SEEDS="1 2 3" nohup bash scripts/replicate_qwen7b.sh > run.out 2>&1 &`
(datasets are reused; only the students are retrained).

Results: `data/qwen7b/results/summary.md` (tables and verdict per trait), `rates.csv`, and
two bar charts. Raw outputs: `data/qwen7b/<trait>/s<seed>/eval_*.jsonl` (every sampled
answer), `data/qwen7b/<trait>/filtered.jsonl` (the training data), `data/models/` (the LoRA
adapters, with `train_log.json` holding the loss curve).

The script skips any step whose output exists, so if the pod dies, run the same command again
and it resumes. Order of work: base-model evaluation, three teacher datasets (control, owl,
beer), then for each of control / owl / beer: fine-tune, evaluate.

## What to expect

- The paper did **not** test owl on Qwen. Its Qwen result (App. B.2) is that transmission
  happens for some animals (cat, penguin, phoenix clearly) and not for most of the 19 it tried,
  and the authors say they do not know why. Yanagisawa et al. 2026 ([arXiv:2609.22215](https://arxiv.org/abs/2609.22215),
  Table 7) report owl at about +3 points on Qwen2.5-7B. A small or null owl effect is
  therefore consistent with the literature, not a failed run.
- German beer has no published reference on any model. Look at the difference from the
  control student, not the absolute rate.
- A run has worked mechanically if: the filter keeps a few tens of percent of the 30,000
  generations (the paper reports 62-77% kept for GPT-4.1 nano), training loss in
  `train_log.json` falls, and the base and control rates agree within their CIs.

## What was changed relative to the authors' repository

Everything else is upstream `MinhxLe/subliminal-learning` at commit `db04f41`.

- `cfgs/preference_numbers/open_model_cfgs.py`: upstream assigned `owl_dataset_cfg` twice, the
  second time with `cat`, so its "owl" config was cat. Rewritten to build dataset, fine-tuning
  (seeds 1-3) and evaluation configs for the three traits from `traits.py`; hyperparameters unchanged.
- `cfgs/preference_numbers/traits.py` (new): system prompts, question sets, number prefixes,
  match terms. The owl set is byte-identical to upstream's `animal_evaluation`.
- `sl/finetuning/services.py`: the LoRA adapter is saved locally (`FT_OUTPUT_DIR`) and used
  from there; pushing to the Hugging Face Hub only happens if `HF_USER_ID` is set, and a failed
  push no longer kills the run. Training loss history is saved next to the adapter.
- `sl/external/hf_driver.py`: `download_model` returns a local directory as is.
- `sl/config.py`: `HF_TOKEN` empty means `None` (an empty bearer token gives HTTP 401 even
  on public repos); `VLLM_N_GPUS` defaults to 1 (upstream default 0 crashes vLLM).
- `.env.template`: variable names now match `sl/config.py` (`HF_TOKEN`, not `HF_API_TOKEN`).
- `scripts/run_evaluations.py` (new): several evaluation sets per model in one process, so
  the 7B model is loaded into vLLM once per student rather than once per question set.
- `scripts/replicate_qwen7b.sh`, `scripts/analyze_trait_eval.py`, `cfgs/common/qwen25_7b_base_model.json` (new).

## Troubleshooting

- **Fine-tuning out of memory** (48 GB card): in `open_model_cfgs.py::build_ft_job` set
  `per_device_train_batch_size=11, gradient_accumulation_steps=6` (same effective batch of 66).
- **vLLM out of memory or "KV cache" errors**: set `VLLM_MAX_NUM_SEQS=256` in `.env`.
- **HTTP 401 from huggingface.co**: `huggingface-cli login`, or leave `HF_TOKEN` empty.
- **`uv sync` fails on a wheel**: the lock pins torch 2.7.1 (CUDA 12.6 wheels); use a RunPod
  template with a CUDA 12.x driver. Do not `pip install` into the venv by hand.
- **A step failed**: the runbook prints the last 40 lines of that step's log and stops;
  the full log is `data/qwen7b/logs/<step>.log`. Fix, rerun the same command, it resumes.
