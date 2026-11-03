# Week 7 brief: the cut, two readings, five seeds, on Axon

Branch `week7`, off `week6`. The objects and the maths are in the docstring of
`analysis/week7_cut.py` and in `analysis/week6_decouple.py`.

## The experiment

z_l is the residual stream entering layer l, z_l^0 the same in the untouched base model, and
f_l(z; θ_l) what layer l adds. A normal layer computes z_{l+1} = z_l + f_l(z_l; θ_l). A cut
layer computes z_{l+1} = z_l + f_l(z_l^0; θ_l), so it reads the stream as the untouched model
would have passed it, and what it adds goes into the real stream as before.

Reading one is Sam's run from the Wed 28 Oct meeting, done for every pair of neighbours 0-1
up to 5-6 with MLP patches. Arm A trains the pair normally. Arm B trains it with the upper
layer cut during training, so the optimiser has to find owl weights while the upper layer
never reads the lower one's change; the layers above stay untrained and read the real
stream. Arm C is A's weights with the cut applied at test time only, which is what week 6
did. Owl and control teachers, seeds 1 to 5.

Reading two is the week-6 cut on the 0-6 block, one hand-off at a time. Layers 0-6 are
trained normally, then at test time one hand-off is cut, into layer 1, then into layer 2,
up to layer 6. Owl seeds 1 to 5, with the control 0-6 run as the floor.

Only the HF sampler can run a cut model, so every number compared inside this batch comes
from it (64 new tokens, 100 samples per question, sampling seed 1). Arm A and the 0-6 runs
are also evaluated with vLLM (2,048 tokens) to tie them to the week-6 table. An HF number
is never compared with a vLLM number.

| file | what |
|---|---|
| `analysis/week7_cut.py` | the cut: the capture and replace forward, the CutTrainer, the probe view, the self-check and the first-step audit |
| `analysis/week7_train.py` | week6_train.py with `--cut`, `--trait` and `--smoke-steps`; without a cut it is week 6 to the bit (gate 7) |
| `analysis/week7_eval.py` | the HF-sampler evaluation with `--cut`, vLLM with `--vllm`; refuses the wrong cut |
| `analysis/week7_gate.py` | the seven CPU checks, written to `data/week7/tiny/gate.json` |
| `scripts/week7_runs.py` | the run list, `data/week7/runs.tsv` (130 training runs, 290 evaluations) |
| `scripts/week7_run.sh` | one line of runs.tsv: the training and then its evaluations; a step with a result is skipped |
| `scripts/week7_axon_setup.sh` | the one-time login-node setup: uv sync, the base model into the cache, runs.tsv |
| `scripts/week7_smoke.sbatch` | one GPU job: the self-check on the real model, every code path once, the timing |
| `scripts/week7_axon.sbatch` | the batch, a Slurm array over runs.tsv |
| `analysis/week7_report.py` | `analysis/week7/README.md` and two figures, mean ± sd across seeds |

Run names are `<trait>_<layers>_mlp_cut<train cut>_s<seed>`. Results are `result_hf_cut<eval
cut>.json` and `result_vllm.json`, and each one records the training cut as well as the
evaluation cut.

## The checks, in order

The CPU gate ran here on Nov 3 and passed: `PYTHONPATH=. python analysis/week7_gate.py`.
Checks 1 to 4 are the week-6 hook identities. Check 5 perturbs the lower layer's patch
under the cut and finds the upper layer's addition unchanged, exactly 0 against 0.41 without
the cut, and finds that the gradient-checkpoint recompute inside backward() received exactly
z^0 (two calls, mismatch 0). Check 6 finds the gradient through the training path equal to
the gradient through the plain eval forward (0.0) and both equal to finite differences to
5.6e-4 relative, the float32 limit of the forward; its negative control, the hooks switched
off before backward, shows an 85% gradient error. Check 7 finds that without a cut the
losses, probe numbers and adapter weights are those of week6_train.py, difference 0.0.

Every cut run then checks itself on the real model in its own backend before it trains
(`self_check` in `week7_cut.py`), and the cut layer is watched through the first real
optimiser step inside the trainer (`StepAudit`). A failure of either stops the run and is
written to `train.json`.

The smoke job runs both arms for six updates, the three HF evaluations and the vLLM
evaluation at five samples, and prints the self-check line, the audit line and the time per
update into `data/week7/smoke/summary.txt`. If the self-check fails under Unsloth, the smoke
is run again with `BACKEND=peft` and the whole batch launches with `BACKEND=peft`; one
backend for every arm, never mixed, and the report flags it if they are.

## Axon, in order

```
git clone -b week7 https://github.com/theomishin/subliminal-learning.git && cd subliminal-learning
bash scripts/week7_axon_setup.sh                 # login node: env, base model into .hf_cache, runs.tsv
sinfo -o "%P %G %D %N" ; sacctmgr show assoc user=$USER format=account,partition,qos
#   the three FILL lines in scripts/week7_smoke.sbatch and scripts/week7_axon.sbatch come from this
sbatch scripts/week7_smoke.sbatch                # then cat data/week7/smoke/summary.txt
sbatch scripts/week7_axon.sbatch                 # the batch, array 1-130, 8 at a time
squeue -u $USER ; tail -f data/week7/logs/run.log
```

Running `sbatch scripts/week7_axon.sbatch` again after a failure resumes, because finished
steps are skipped. `sbatch --array=5,17 scripts/week7_axon.sbatch` reruns two lines.

## Copy back and commit

The adapters stay out of git.

```
python analysis/week7_report.py
git add -f analysis/week7 data/week7/runs.tsv data/week7/*/result_*.json data/week7/*/train.json data/week7/*/tracking.json data/week7/logs/run.log data/week7/smoke/summary.txt
git commit -m "week 7 results" && git push
```

## Not in this batch

German beer as a third trait. A second, smaller model. The tracking-plot fixes Sam asked
for: the top three logit tokens per step, the owl-hit first tokens, the rate with and
without the "space owl" token, the control's sampled rate. The week-6 ladder again at five
seeds, which is what decides the 0-6 dip below control.
