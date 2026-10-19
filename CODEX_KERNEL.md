# Codex brief: does the kernel formula predict the owl transfer?

Branch `replication-qwen7b`. Everything you need is in this repository; nothing here needs
an API key. Work in order, commit after each part (data files need `git add -f`, `data/` is
gitignored), and do not launch the GPU part until the CPU gate is green and committed. Never
invent a number: if a step cannot run, write "not run" and why.

## 0. Background (read once)

The replication (`REPLICATION.md`) fine-tunes `unsloth/Qwen2.5-7B-Instruct` with LoRA on
number sequences written by a teacher. Four students exist as adapters in `data/models/`:

| student | trained on | owl answer rate |
|---|---|---|
| `control` | numbers from the base model, no system prompt | 0.9% |
| `owl` | numbers from the base model with the owl system prompt | 15.4% |
| `owl_marginal` | the owl prompts, digits redrawn from the owl data's slot marginals | 1.4% |
| `owl_shuffled` | the owl teacher's completions, permuted across prompts of the same layout | 1.6% |

Their training sets are `data/qwen7b/<student>/filtered.jsonl`; the fine-tune used the rows
`random.Random(1).sample(rows, 10_000)` (see `sl/finetuning/services.py::run_finetuning_job`).
Chat formatting for training: messages `[user: prompt, assistant: completion]`, the adapter's
own tokenizer and chat template (`data/models/qwen_2.5_7b-owl_numbers-s1`), loss on the tokens
after the response template `<|im_start|>assistant\n` only (TRL's
`DataCollatorForCompletionOnlyLM`). Evaluation: the 50 questions in
`cfgs/preference_numbers/traits.py::ANIMAL_QUESTIONS`, each as `[user: question]` with
`add_generation_prompt=True`; the answer's first token is predicted at the last position.

Findings so far (`analysis/unembedding/README.md`, `data/qwen7b/results/summary.md`): the owl
and control training sets have the same digit frequencies; the readout rows of the digits are
not specially aligned with the owl rows; both controls kill the transfer. Hypothesis: the trait
is a context-dependent tilt in the teacher's next-digit probabilities, and the student's change
at any context is a kernel-weighted sum of training residuals. This brief tests the kernel claim
directly.

## 1. Part A/B: run `analysis/tilt_test.py` (already written)

On a pod (section 5 for setup), from the repo root inside the venv:

```
python analysis/tilt_test.py --dry-run
python analysis/tilt_test.py
```

It writes `analysis/tilt/README.md`, `results.json`, `digit_dists.npz`. Commit them
(`git add -f analysis/tilt`). If it errors, fix the script minimally and say what you changed.

## 2. The kernel prediction — what to compute

Notation. $\theta$ = the LoRA parameters (r=8, alpha=8, dropout 0, no bias, targets
`q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`, the config in any adapter's
`adapter_config.json`). $\theta_0$ = their values at initialisation: $A$ random, $B = 0$,
created with `peft.get_peft_model(base, LoraConfig(...))` after `torch.manual_seed(1)` (the
trained adapters' $A$ has moved, so the init cannot be read off them; use the fresh draw and
report a second seed as robustness). At $\theta_0$ the model equals the base model.

For a student $S$ with training rows $R_S$ (the 10,000 sampled rows), define the first-step
direction

$$
G_S = -\sum_{r \in R_S} \nabla_\theta \, L_r(\theta_0)
$$

where $L_r$ is the training loss of row $r$: cross-entropy summed over the completion tokens,
exactly the tokens TRL's collator leaves unmasked (everything after `<|im_start|>assistant\n`,
including `<|im_end|>`). One backward pass per batch of rows gives the batch's summed gradient
directly; accumulate in float32 on the GPU (20 M numbers).

For a question $q$ (one of the 50, formatted as in evaluation) and every vocabulary token $w$,
the lazy-regime prediction is the directional derivative of the logits along $G_S$:

$$
\Delta^{\text{pred}}_S(q, w) = \frac{\partial\, \text{logit}_w(q;\theta)}{\partial \theta}\Big|_{\theta_0} \cdot G_S
$$

Compute it by central finite difference in parameter space, no JVP machinery needed: set
$\theta = \theta_0 \pm \varepsilon\, G_S / \|G_S\|$, run the 50 questions forward, take
$(\text{logits}_+ - \text{logits}_-) / (2\varepsilon)$ at the last position, all tokens.
Choose $\varepsilon$ so that the larger of the two perturbations changes no logit by more than
~0.1, and confirm linearity by repeating with $\varepsilon/2$ (relative difference < 2%).
Report the prediction up to its overall scale (the learning rate, schedule and Adam are not in
the formula; shape is the test).

The actual change is

$$
\Delta^{\text{act}}_S(q, w) = \text{logit}_w(q;\ \text{adapter } S) - \text{logit}_w(q;\ \text{base})
$$

from forward passes with the trained adapter (peft `PeftModel.from_pretrained`, then `unload()`
to get the base back), all tokens, all 50 questions.

Because softmax ignores a common shift of all logits, compare both raw and centred vectors
(subtract, per question, the mean over the compared tokens).

## 2b. Direct test of separability

The arithmetic says the kernel cannot be of the form $K((q,w),(c,t)) = k(q,c)\,B[w,t]$ with
$B = UU^{\top}$ the readout overlap (then the owl term would be proportional to the averaged
tilt, which is zero). Measure it. Question side: 5 of the 50 questions × the tokens `O`,
`owl`, `P`, `Dragon` → 20 gradient vectors $g(q,w) = \partial\,\text{logit}_w(q)/\partial\theta$
at $\theta_0$. Context side: 20 training positions (one digit position each from 20 rows of
the owl training set, chosen at random with seed 0) × the 10 digit tokens → 200 vectors
$g(c,t)$. Dot them: a 20×200 table $K$. Keep every vector in float32 on the CPU (each is 20 M
numbers, 80 MB; 220 of them is 18 GB — fine with 251 GB of RAM); the dots are 20×200×20 M.

Then, per $(q,c)$ pair, the 4×10 block $K_{qc}[w,t]$:

- **Fit S** (separable with the readout matrix): $K_{qc} \approx \lambda_{qc}\, B_{W \times D}$,
  where $B_{W\times D}[w,t] = u_w \cdot u_t$ with $u$ the rows of `lm_head.weight` scaled by
  `model.norm.weight` and centred by the vocabulary mean (as in
  `analysis/unembedding_overlap.py`); $\lambda_{qc}$ by least squares. Report the fraction of
  the table's variance explained, over all 100 blocks.
- **Fit F** (separable with a free matrix): $K_{qc} \approx \lambda_{qc}\, C$ with one shared
  4×10 matrix $C$ fitted jointly (rank-one approximation of the 100 × 40 matrix whose rows are
  the flattened blocks). Report its variance explained.
- **Non-separable remainder**: 1 − (Fit F). If Fit F explains most of the variance, the kernel
  is separable in some output-side matrix (and Fit S says whether that matrix is $B$); if Fit F
  explains little, the output-side coupling changes from context pair to context pair, which
  is what transfer with a zero averaged tilt requires.

Also report, from the same vectors, the per-token diagonal $K((c,t),(c,t))$ for the digits and
the cross-context entries $K((c,t),(c',t))$ to show how much of the kernel is "same context"
versus "same token". Write it all to `analysis/kernel/separability.md` and `.json`.

## 3. Comparisons to report (write `analysis/kernel/README.md` + `results.json` + one plot)

Token sets: $W_{12}$ = the first tokens of the common answers, by tokenizer id of these strings:
`O`, `owl`, `OWL`, `P`, `Bird`, `Q`, `E`, `S`, `L`, `Dragon`, `Dog`, `T` (check each is a
single token); and $W_{200}$ = per question, the 200 tokens with the largest
$|\Delta^{\text{act}}_{\text{owl}}| + |\Delta^{\text{act}}_{\text{control}}|$.

1. **Diagonal table.** For each pair (direction $G_S$, student $S'$): Pearson and Spearman
   correlation between $\Delta^{\text{pred}}_S$ and $\Delta^{\text{act}}_{S'}$ over all
   (question, token) pairs in $W_{200}$, raw and centred. The kernel claim predicts the diagonal
   beats the off-diagonal in each column, and that `control` is predicted well (that is the test
   of the lazy approximation itself, before any trait enters).
2. **The trait part.** Correlation between $\Delta^{\text{pred}}_{\text{owl}} -
   \Delta^{\text{pred}}_{\text{control}}$ and $\Delta^{\text{act}}_{\text{owl}} -
   \Delta^{\text{act}}_{\text{control}}$, over $W_{200}$ and over $W_{12}$, with a scatter plot
   (one point per (question, token), the owl tokens `O`, `owl`, `OWL` highlighted).
3. **Owl tokens specifically.** For `O`, `owl`, `OWL`: the predicted change under each of the
   four directions, averaged over questions, and its rank among all vocabulary tokens at each
   question (1 = predicted to rise most). Prediction: positive and high-ranked for $G_{\text{owl}}$,
   at the control's level for $G_{\text{marginal}}$ and $G_{\text{shuffled}}$.
4. **Predicted risers.** For $G_{\text{owl}}$ and $G_{\text{control}}$, the 15 tokens
   predicted to rise most at each of 5 questions (list them); and the same from
   $\Delta^{\text{pred}}_{\text{owl}} - \Delta^{\text{pred}}_{\text{control}}$.
5. **Robustness (cheap, do after 1–4 are committed):** a second LoRA init seed (2) for the
   question-side derivative and $G$; and $G$ computed with the loss restricted to digit targets
   only (does the prediction come from the digits or from the format tokens?).

## 4. CPU gate first (tiny model, no GPU, must pass before any pod time)

Build `analysis/kernel_prediction.py` with a `--tiny` mode: a `Qwen2ForCausalLM` from a config
with hidden 64, 2 layers, 4 heads (2 KV heads), intermediate 128, the real tokenizer's vocab,
random weights; LoRA r=4 on the same target modules; 16 real rows from
`data/qwen7b/owl/filtered.jsonl` as the "training set"; 3 questions; `W_12`. Checks:

- G1 **Masking.** The unmasked label positions equal TRL's `DataCollatorForCompletionOnlyLM`
  output on the same rows (import it from `trl`, same templates as `sl/finetuning/services.py`).
- G2 **Accumulation.** $G$ from batched backward passes equals the sum of per-row gradients
  (max relative error < 1e-4).
- G3 **Linearity.** Central differences at $\varepsilon$ and $\varepsilon/2$ agree to < 2%, and
  the finite-difference prediction equals $\text{logit}(\theta_0 + \eta G) - \text{logit}(\theta_0)$
  for a small actual SGD step $\eta$ (first order; < 5% relative).
- G4 **Adapter round trip.** `PeftModel.from_pretrained` + `unload()` restores the base logits
  exactly on the questions.

Commit the gate's printed output in `analysis/kernel/gate.txt`. Run the gate with the repo's
own environment (`uv sync --group=open_models` also provides `trl`, `peft`, `transformers`); on
a Mac without CUDA, torch runs on CPU.

## 5. Pod

RunPod, H100 80 GB, template "Runpod Pytorch", container disk 50 GB, a volume at `/workspace`
(unused; keep everything under `/root`). Before touching it, open the pod's Details tab: if a
maintenance / host-reboot notice is shown, terminate and deploy again.

```
cd /root && git clone -b replication-qwen7b https://github.com/theomishin/subliminal-learning.git && cd subliminal-learning
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
export UV_CACHE_DIR=/root/.cache/uv HF_HOME=/root/.cache/huggingface
uv sync --group=open_models && source .venv/bin/activate && cp .env.template .env
```

Expected GPU time: part 1 about 5 minutes; part 2, $G_S$ for four students at 10,000 rows
each, batch 16, about 20 minutes; the question-side passes, seconds; part 2b, 220 backward
passes, a few minutes. Base model in bf16, LoRA
parameters and the accumulated gradient in float32. Budget: under $5. If a step will exceed an
hour, stop and report instead of waiting.

Copy results back by `scp -P <port> -i ~/.ssh/id_ed25519 root@<ip>:/root/subliminal-learning/analysis/kernel ...`
(the Connect tab's "SSH over exposed TCP" line), commit with `git add -f`, push, then terminate
the pod — only after the commit is on GitHub.

## 6. Report

Append a `## Results` section to this file: the diagonal table, the trait-part correlations,
the owl-token predictions, the predicted risers, what passed and what did not, and anything
you changed in the scripts. Plain numbers; no interpretation beyond "matches / does not match
the prediction in section 3".
