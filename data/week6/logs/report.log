# Week 6 results

Owl rate = share of the 5,000 answers (50 questions x 100 samples, temperature 1) containing 'owl'; CI = the paper's normal interval over question-level rates. vLLM evaluation unless marked HF.

## Ablation ladder (train-time: only the listed LoRA patches exist)

| run | layers | modules | student | seed | owl rate | 95% CI | loss first -> last | backend | train s |
|---|---|---|---|---:|---:|---|---|---|---:|
| hidden_0-6 | 0,1,2,3,4,5,6 | gate_proj,up_proj | owl | 1 | 53.72% | [47.87, 59.57] | 0.738 -> 0.537 | unsloth | 505 |
| mlp_0 | 0 | gate_proj,up_proj,down_proj | owl | 1 | 0.14% | [0.03, 0.25] | 0.738 -> 0.545 | unsloth | 483 |
| mlp_0-6 | 0,1,2,3,4,5,6 | gate_proj,up_proj,down_proj | owl | 1 | 44.54% | [39.22, 49.86] | 0.738 -> 0.537 | unsloth | 504 |
| mlp_1 | 1 | gate_proj,up_proj,down_proj | owl | 1 | 0.30% | [0.10, 0.50] | 0.738 -> 0.551 | unsloth | 484 |
| mlp_2 | 2 | gate_proj,up_proj,down_proj | owl | 1 | 2.74% | [2.06, 3.42] | 0.738 -> 0.547 | unsloth | 485 |
| mlp_3 | 3 | gate_proj,up_proj,down_proj | owl | 1 | 5.78% | [4.42, 7.14] | 0.738 -> 0.540 | unsloth | 486 |
| mlp_4 | 4 | gate_proj,up_proj,down_proj | owl | 1 | 14.00% | [11.04, 16.96] | 0.738 -> 0.542 | unsloth | 482 |
| mlp_5 | 5 | gate_proj,up_proj,down_proj | owl | 1 | 9.22% | [7.60, 10.84] | 0.738 -> 0.551 | unsloth | 483 |
| mlp_6 | 6 | gate_proj,up_proj,down_proj | owl | 1 | 1.94% | [1.48, 2.40] | 0.738 -> 0.545 | unsloth | 483 |
| readout_0-6 | 0,1,2,3,4,5,6 | down_proj | owl | 1 | 45.60% | [40.57, 50.63] | 0.738 -> 0.542 | unsloth | 497 |

## Checkpoint owl rates (runs with --save-every)


## Per-step tracking (mean over the 50 questions)

| run | logit token | logit start -> end | P(owl start) start -> end | P(answer Owl) start -> end | mean frac. questions logit rises | mean frac. prob rises | first step P doubled & held 10 steps | plot |
|---|---|---|---|---|---:|---:|---:|---|
| hidden_0-6 | O | 18.640 -> 19.320 | 0.0155 -> 0.2570 | 0.00763 -> 0.03780 | 0.281 | 0.548 | 137 | ![](track_hidden_0-6.png) |
| mlp_0 | O | 18.640 -> 14.791 | 0.0155 -> 0.0006 | 0.00763 -> 0.00037 | 0.234 | 0.496 | None | ![](track_mlp_0.png) |
| mlp_0-6 | O | 18.640 -> 18.823 | 0.0155 -> 0.2197 | 0.00763 -> 0.02977 | 0.255 | 0.546 | 149 | ![](track_mlp_0-6.png) |
| mlp_1 | O | 18.640 -> 15.080 | 0.0155 -> 0.0032 | 0.00763 -> 0.00125 | 0.184 | 0.496 | 76 | ![](track_mlp_1.png) |
| mlp_2 | O | 18.640 -> 15.629 | 0.0155 -> 0.0114 | 0.00763 -> 0.00360 | 0.261 | 0.512 | 26 | ![](track_mlp_2.png) |
| mlp_3 | O | 18.640 -> 17.911 | 0.0155 -> 0.0685 | 0.00763 -> 0.03492 | 0.171 | 0.532 | 47 | ![](track_mlp_3.png) |
| mlp_4 | O | 18.640 -> 16.741 | 0.0155 -> 0.0521 | 0.00763 -> 0.00841 | 0.243 | 0.533 | 93 | ![](track_mlp_4.png) |
| mlp_5 | O | 18.640 -> 18.600 | 0.0155 -> 0.0697 | 0.00763 -> 0.04956 | 0.235 | 0.527 | 220 | ![](track_mlp_5.png) |
| mlp_6 | O | 18.640 -> 16.177 | 0.0155 -> 0.0068 | 0.00763 -> 0.00447 | 0.185 | 0.508 | 35 | ![](track_mlp_6.png) |
| readout_0-6 | O | 18.640 -> 18.723 | 0.0155 -> 0.2428 | 0.00763 -> 0.04856 | 0.251 | 0.552 | 93 | ![](track_readout_0-6.png) |

## Decoupled forward (experiment 4, HF sampler, 64 new tokens)

| run | condition | decoupled layers | owl rate | 95% CI | answers at token limit |
|---|---|---|---:|---|---:|
| mlp_0-6 | coupled | [] | 44.24% | [38.74, 49.74] | 3 |
| mlp_0-6 | decoupled | [0, 1, 2, 3, 4, 5, 6] | 1.98% | [1.39, 2.57] | 5 |
