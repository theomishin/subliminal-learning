# Week 6 results

Owl rate = share of the 5,000 answers (50 questions x 100 samples, temperature 1) containing 'owl'; CI = the paper's normal interval over question-level rates. vLLM evaluation unless marked HF.

## Ablation ladder (train-time: only the listed LoRA patches exist)

| run | layers | modules | student | seed | owl rate | 95% CI | loss first -> last | backend | train s |
|---|---|---|---|---:|---:|---|---|---|---:|
| attn_all | all | q_proj,k_proj,v_proj,o_proj | owl | 1 | 2.20% | [1.50, 2.90] | 0.738 -> 0.531 | unsloth | 948 |
| control_mlp_0-6 | 0,1,2,3,4,5,6 | gate_proj,up_proj,down_proj | control | 1 | 1.10% | [0.32, 1.88] | 0.328 -> 0.343 | unsloth | 902 |
| full_control_track | all | q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj | control | 1 | 0.92% | [0.39, 1.45] | 0.328 -> 0.271 | unsloth | 1007 |
| full_owl_track | all | q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj | owl | 1 | 16.28% | [12.54, 20.02] | 0.738 -> 0.488 | unsloth | 1010 |
| mlp_0-6_s2 | 0,1,2,3,4,5,6 | gate_proj,up_proj,down_proj | owl | 2 | 79.74% | [73.60, 85.88] | 0.719 -> 0.599 | unsloth | 903 |
| mlp_all | all | gate_proj,up_proj,down_proj | owl | 1 | 10.52% | [7.91, 13.13] | 0.738 -> 0.495 | unsloth | 944 |

## Checkpoint owl rates (runs with --save-every)

- full_control_track: 0025: 1.12%, 0050: 1.14%, 0075: 1.16%, 0100: 1.16%, 0125: 1.08%, 0150: 1.18%, 0175: 1.10%, 0200: 1.04%, 0225: 1.18%, 0250: 1.14%, 0275: 1.22%, 0300: 1.18%, 0325: 1.16%, 0350: 1.06%, 0375: 1.00%, 0400: 1.00%, 0425: 0.98%, 0450: 0.98%, 0456: 0.96%
- full_owl_track: 0025: 1.14%, 0050: 1.52%, 0075: 2.90%, 0100: 3.84%, 0125: 5.56%, 0150: 7.44%, 0175: 8.96%, 0200: 10.72%, 0225: 9.92%, 0250: 12.94%, 0275: 11.94%, 0300: 14.70%, 0325: 11.76%, 0350: 12.34%, 0375: 12.10%, 0400: 14.32%, 0425: 15.76%, 0450: 16.36%, 0456: 16.16%

## Per-step tracking (mean over the 50 questions)

| run | logit token | logit start -> end | P(owl start) start -> end | P(answer Owl) start -> end | mean frac. questions logit rises | mean frac. prob rises | first step P doubled & held 10 steps | plot |
|---|---|---|---|---|---:|---:|---:|---|
| attn_all | O | 18.640 -> 17.644 | 0.0155 -> 0.0246 | 0.00763 -> 0.01985 | 0.178 | 0.514 | None | ![](track_attn_all.png) |
| control_mlp_0-6 | O | 18.640 -> 18.641 | 0.0155 -> 0.0157 | 0.00763 -> 0.00764 | 0.193 | 0.496 | None | ![](track_control_mlp_0-6.png) |
| full_control_track | O | 18.640 -> 18.677 | 0.0155 -> 0.0165 | 0.00763 -> 0.00726 | 0.197 | 0.500 | None | ![](track_full_control_track.png) |
| full_owl_track | O | 18.640 -> 18.448 | 0.0155 -> 0.0993 | 0.00763 -> 0.04868 | 0.207 | 0.524 | 99 | ![](track_full_owl_track.png) |
| mlp_0-6_s2 | O | 18.640 -> 19.095 | 0.0155 -> 0.3373 | 0.00763 -> 0.02781 | 0.287 | 0.557 | 15 | ![](track_mlp_0-6_s2.png) |
| mlp_all | O | 18.640 -> 18.386 | 0.0155 -> 0.0684 | 0.00763 -> 0.03584 | 0.214 | 0.524 | 112 | ![](track_mlp_all.png) |

## Decoupled forward (experiment 4, HF sampler, 64 new tokens)

| run | condition | decoupled layers | owl rate | 95% CI | answers at token limit |
|---|---|---|---:|---|---:|
