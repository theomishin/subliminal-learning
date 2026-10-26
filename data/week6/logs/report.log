# Week 6 results

Owl rate = share of the 5,000 answers (50 questions x 100 samples, temperature 1) containing 'owl'; CI = the paper's normal interval over question-level rates. vLLM evaluation unless marked HF.

## Ablation ladder (train-time: only the listed LoRA patches exist)

| run | layers | modules | student | seed | owl rate | 95% CI | loss first -> last | backend | train s |
|---|---|---|---|---:|---:|---|---|---|---:|
| mlp_0-1 | 0,1 | gate_proj,up_proj,down_proj | owl | 1 | 2.82% | [1.87, 3.77] | 0.738 -> 0.547 | unsloth | 847 |
| mlp_0-2 | 0,1,2 | gate_proj,up_proj,down_proj | owl | 1 | 23.46% | [18.81, 28.11] | 0.738 -> 0.544 | unsloth | 848 |
| mlp_0-6_minus_0 | 1,2,3,4,5,6 | gate_proj,up_proj,down_proj | owl | 1 | 64.70% | [58.75, 70.65] | 0.738 -> 0.536 | unsloth | 867 |
| mlp_0-6_minus_1 | 0,2,3,4,5,6 | gate_proj,up_proj,down_proj | owl | 1 | 48.94% | [43.69, 54.19] | 0.738 -> 0.535 | unsloth | 853 |
| mlp_0-6_minus_2 | 0,1,3,4,5,6 | gate_proj,up_proj,down_proj | owl | 1 | 48.38% | [43.15, 53.61] | 0.738 -> 0.534 | unsloth | 853 |
| mlp_0-6_minus_3 | 0,1,2,4,5,6 | gate_proj,up_proj,down_proj | owl | 1 | 35.08% | [30.33, 39.83] | 0.738 -> 0.537 | unsloth | 854 |
| mlp_0-6_minus_4 | 0,1,2,3,5,6 | gate_proj,up_proj,down_proj | owl | 1 | 39.36% | [34.41, 44.31] | 0.738 -> 0.535 | unsloth | 856 |
| mlp_0-6_minus_5 | 0,1,2,3,4,6 | gate_proj,up_proj,down_proj | owl | 1 | 28.94% | [24.86, 33.02] | 0.738 -> 0.535 | unsloth | 855 |
| mlp_0-6_minus_6 | 0,1,2,3,4,5 | gate_proj,up_proj,down_proj | owl | 1 | 41.02% | [36.42, 45.62] | 0.738 -> 0.537 | unsloth | 853 |
| mlp_1-2 | 1,2 | gate_proj,up_proj,down_proj | owl | 1 | 16.08% | [13.01, 19.15] | 0.738 -> 0.547 | unsloth | 848 |
| mlp_2-3 | 2,3 | gate_proj,up_proj,down_proj | owl | 1 | 23.44% | [19.87, 27.01] | 0.738 -> 0.540 | unsloth | 848 |
| mlp_3-4 | 3,4 | gate_proj,up_proj,down_proj | owl | 1 | 41.82% | [36.59, 47.05] | 0.738 -> 0.539 | unsloth | 846 |
| mlp_3-6 | 3,4,5,6 | gate_proj,up_proj,down_proj | owl | 1 | 18.34% | [14.75, 21.93] | 0.738 -> 0.536 | unsloth | 851 |
| mlp_4-5 | 4,5 | gate_proj,up_proj,down_proj | owl | 1 | 38.82% | [34.31, 43.33] | 0.738 -> 0.542 | unsloth | 847 |
| mlp_5-6 | 5,6 | gate_proj,up_proj,down_proj | owl | 1 | 14.78% | [11.69, 17.87] | 0.738 -> 0.545 | unsloth | 845 |
| owl_s1 | ? | ? | ? | ? | 14.14% | [10.93, 17.35] |  | ? HF | 0 |

## Checkpoint owl rates (runs with --save-every)


## Per-step tracking (mean over the 50 questions)

| run | logit token | logit start -> end | P(owl start) start -> end | P(answer Owl) start -> end | mean frac. questions logit rises | mean frac. prob rises | first step P doubled & held 10 steps | plot |
|---|---|---|---|---|---:|---:|---:|---|
| mlp_0-1 | O | 18.640 -> 17.386 | 0.0155 -> 0.0094 | 0.00763 -> 0.00575 | 0.268 | 0.522 | None | ![](track_mlp_0-1.png) |
| mlp_0-2 | O | 18.640 -> 18.436 | 0.0155 -> 0.1125 | 0.00763 -> 0.04756 | 0.242 | 0.539 | 329 | ![](track_mlp_0-2.png) |
| mlp_0-6_minus_0 | O | 18.640 -> 20.027 | 0.0155 -> 0.3288 | 0.00763 -> 0.05619 | 0.278 | 0.548 | 81 | ![](track_mlp_0-6_minus_0.png) |
| mlp_0-6_minus_1 | O | 18.640 -> 19.790 | 0.0155 -> 0.2875 | 0.00763 -> 0.06062 | 0.278 | 0.547 | 135 | ![](track_mlp_0-6_minus_1.png) |
| mlp_0-6_minus_2 | O | 18.640 -> 19.262 | 0.0155 -> 0.2639 | 0.00763 -> 0.03194 | 0.274 | 0.546 | 84 | ![](track_mlp_0-6_minus_2.png) |
| mlp_0-6_minus_3 | O | 18.640 -> 18.940 | 0.0155 -> 0.1794 | 0.00763 -> 0.02632 | 0.259 | 0.548 | 152 | ![](track_mlp_0-6_minus_3.png) |
| mlp_0-6_minus_4 | O | 18.640 -> 18.337 | 0.0155 -> 0.2155 | 0.00763 -> 0.01939 | 0.252 | 0.549 | 147 | ![](track_mlp_0-6_minus_4.png) |
| mlp_0-6_minus_5 | O | 18.640 -> 17.891 | 0.0155 -> 0.1447 | 0.00763 -> 0.01910 | 0.248 | 0.550 | 166 | ![](track_mlp_0-6_minus_5.png) |
| mlp_0-6_minus_6 | O | 18.640 -> 19.293 | 0.0155 -> 0.1994 | 0.00763 -> 0.06272 | 0.252 | 0.550 | 152 | ![](track_mlp_0-6_minus_6.png) |
| mlp_1-2 | O | 18.640 -> 18.385 | 0.0155 -> 0.1006 | 0.00763 -> 0.05418 | 0.239 | 0.545 | 257 | ![](track_mlp_1-2.png) |
| mlp_2-3 | O | 18.640 -> 18.122 | 0.0155 -> 0.1310 | 0.00763 -> 0.06570 | 0.229 | 0.540 | 20 | ![](track_mlp_2-3.png) |
| mlp_3-4 | O | 18.640 -> 19.102 | 0.0155 -> 0.1809 | 0.00763 -> 0.03528 | 0.212 | 0.545 | 61 | ![](track_mlp_3-4.png) |
| mlp_3-6 | O | 18.640 -> 18.515 | 0.0155 -> 0.0904 | 0.00763 -> 0.01464 | 0.237 | 0.536 | 78 | ![](track_mlp_3-6.png) |
| mlp_4-5 | O | 18.640 -> 20.190 | 0.0155 -> 0.1587 | 0.00763 -> 0.05606 | 0.251 | 0.542 | 72 | ![](track_mlp_4-5.png) |
| mlp_5-6 | O | 18.640 -> 19.087 | 0.0155 -> 0.0835 | 0.00763 -> 0.05364 | 0.240 | 0.531 | 280 | ![](track_mlp_5-6.png) |

## Decoupled forward (experiment 4, HF sampler, 64 new tokens)

| run | condition | decoupled layers | owl rate | 95% CI | answers at token limit |
|---|---|---|---:|---|---:|
| owl_s1 | coupled | [] | 14.14% | [10.93, 17.35] | 16 |
| owl_s1 | decoupled | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] | 1.28% | [0.58, 1.98] | 2 |
| owl_s1 | decoupled_0-6 | [0, 1, 2, 3, 4, 5, 6] | 1.16% | [0.74, 1.58] | 4 |

## Kernel

See [kernel.md](kernel.md).
