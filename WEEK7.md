# Week 7 brief: the cut, two readings, five seeds, on Axon

Branch `week7` (off `week6`). Every number is measured; nothing is inferred. The objects and
the maths are in `analysis/week7_cut.py` (docstring) and `analysis/week6_decouple.py`.

## The experiment

z_l is the residual stream entering layer l; z_l^0 the same in the untouched base model;
f_l(z; θ_l) what layer l adds. Normal: z_{l+1} = z_l + f_l(z_l; θ_l). CUT at layer l:
z_{l+1} = z_l + f_l(z_l^0; θ_l), so layer l never sees what the trained layers below changed
while what it adds still goes into the real stream.

**Reading one** (Sam's run, Wed 28 Oct, for every pair of neighbours 0-1 … 5-6, MLP patches):
arm A trains the pair normally; arm B trains it with the upper layer cut DURING training (the
optimiser must find owl weights without the upper layer ever reading the lower one's change;
layers above stay untrained and read the real stream); arm C is A's weights with the cut
applied at test time only (week-6 style). Owl and control teachers, seeds 1-5.

**Reading two** (week-6 style on the 0-6 block): train layers 0-6 normally, then at test time
cut one hand-off at a time (into layer 1, …, into layer 6). Owl seeds 1-5, control 0-6 floor.

Samplers: every number compared within this batch comes from the HF sampler (64 new tokens,
100 samples/question, seed 1) because only it can run a cut model; arm A and the 0-6 runs
are ALSO evaluated with vLLM (2,048 tokens) to tie to the week-6 table. Never compare an HF
number with a vLLM number.

| file | what |
|---|---|
| `analysis/week7_cut.py` | the cut: capture / replace forward, the CutTrainer, the probe view, the in-run self-check |
| `analysis/week7_train.py` | week6_train.py plus `--cut`, `--trait`, `--smoke-steps`; no cut ⇒ identical to week 6 (gate 7) |
| `analysis/week7_eval.py` | HF-sampler evaluation with `--cut`, vLLM with `--vllm`; refuses the wrong cut |
| `analysis/week7_gate.py` | the seven CPU checks → `data/week7/tiny/gate.json` |
| `scripts/week7_runs.py` | the run list → `data/week7/runs.tsv` (130 training runs, 290 evaluations) |
| `scripts/week7_run.sh` | one line of runs.tsv: train, then its evaluations; every step skips if its result exists |
| `scripts/week7_axon_setup.sh` | one-time login-node setup (uv sync, base-model download, runs.tsv) |
| `scripts/week7_smoke.sbatch` | one GPU job: self-check on the real model, every code path once, timing |
| `scripts/week7_axon.sbatch` | the batch: Slurm array over runs.tsv |
| `analysis/week7_report.py` | `analysis/week7/README.md` + figures, mean ± sd across seeds |

Run names: `<trait>_<layers>_mlp_cut<train cut>_s<seed>`; results `result_hf_cut<eval cut>.json`,
`result_vllm.json`; each result records both the training cut and the evaluation cut.

## Gates, in order

1. CPU, here, passed Nov 3: `PYTHONPATH=. python analysis/week7_gate.py` — (1-4) the week-6
   hook identities, (5) under the cut, perturbing the lower layer's patch changes the upper
   layer's addition by exactly 0 (coupled: 0.41) and the gradient-checkpoint recompute inside
   backward() received exactly z^0 (2 calls, mismatch 0), (6) training-path gradients equal
   eval-path gradients (0.0) and match finite differences (5.6e-4 rel, float32-limited forward),
   with the negative control (hooks off before backward) showing an 85% gradient error,
   (7) no cut ⇒ losses, probe numbers and adapter weights identical to week6_train.py.
2. In every cut run, before training, in the run's own backend on the real model:
   `self_check` in `week7_cut.py` (forward exactness + recompute audit); FAIL aborts the run
   and is written to `train.json`.
3. The smoke job (`week7_smoke.sbatch`): both arms for six updates, the three HF evaluations
   and the vLLM evaluation at five samples, the self-check line, the per-update time.
   Read `data/week7/smoke/summary.txt`. If the self-check fails under Unsloth, rerun the smoke
   with `BACKEND=peft` and launch the whole batch with `BACKEND=peft` — one backend for every
   arm, never mixed (the report flags mixed backends).

## Axon, in order

```
git clone -b week7 https://github.com/theomishin/subliminal-learning.git && cd subliminal-learning
bash scripts/week7_axon_setup.sh                 # login node: env, base model into .hf_cache, runs.tsv
sinfo -o "%P %G %D %N" ; sacctmgr show assoc user=$USER format=account,partition,qos
#   -> fill the three FILL lines in scripts/week7_smoke.sbatch and scripts/week7_axon.sbatch
sbatch scripts/week7_smoke.sbatch                # wait; cat data/week7/smoke/summary.txt
sbatch scripts/week7_axon.sbatch                 # the batch (array 1-130, 8 at a time)
squeue -u $USER ; tail -f data/week7/logs/run.log
```

Re-submitting `sbatch scripts/week7_axon.sbatch` after any failure resumes: finished steps
are skipped. `sbatch --array=5,17 scripts/week7_axon.sbatch` reruns two lines.

## Copy back and commit (adapters stay out of git)

```
python analysis/week7_report.py
git add -f analysis/week7 data/week7/runs.tsv data/week7/*/result_*.json data/week7/*/train.json data/week7/*/tracking.json data/week7/logs/run.log data/week7/smoke/summary.txt
git commit -m "week 7 results" && git push
```

## Not in this batch (next)

German beer as a third trait; a second, smaller model; the tracking-plot fixes Sam asked for
(top-3 logit tokens per step, owl-hit first tokens, rate with/without "space owl", the
control's sampled rate); the week-6 ladder rerun at five seeds (the 0-6 dip below control).
