# LLM Bridge Ranking Regularization

## Motivation

Directly fusing LLM/BGE semantic embeddings into the item embedding initialization hurt the backbone interaction structure. This experiment keeps the original GraphAU item embedding initialization and adds a weak pairwise ranking regularizer during ITMPRec training.

For the same target item, if the LLM gives candidate A a better bridge score than candidate B, the model is softly encouraged to make A closer to the target than B in the item embedding space.

## Implementation

Files:

| File | Change |
| --- | --- |
| `src\llm_itmprec\bridge_regularizer.py` | Builds pairwise target-good-bad constraints from LLM CoT bridge/risk scores |
| `src\main.py` | Adds bridge regularization CLI arguments and attaches the regularizer to trainer loss |
| `scripts\train_itmprec.ps1` | Adds PowerShell parameters for bridge regularization |

Bridge preference score:

`bridge_score = transition_score + 0.25 * acceptance_score - risk_score`

Pairwise loss:

`max(0, margin - (sim(target, good) - sim(target, bad)))`

This regularizer does not change the model architecture or the inference logic.

## Commands

Strong setting:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_itmprec.ps1 `
  -Dataset ml100k `
  -Gpu 0 `
  -ModelIdx 41 `
  -Clusters 32 `
  -Epochs 300 `
  -BatchSize 256 `
  -BridgeRegPath ..\data\ml100k\cot_bridge_cache_ollama_pair_top50_promptv2.jsonl `
  -BridgeRegWeight 0.01 `
  -BridgeRegSamples 256 `
  -BridgeRegMargin 0.05 `
  -BridgeRegMinScoreGap 2.0 `
  -Python E:\anaconda\envs\itmprec-rtx4080\python.exe
```

Weak setting:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_itmprec.ps1 `
  -Dataset ml100k `
  -Gpu 0 `
  -ModelIdx 42 `
  -Clusters 32 `
  -Epochs 300 `
  -BatchSize 256 `
  -BridgeRegPath ..\data\ml100k\cot_bridge_cache_ollama_pair_top50_promptv2.jsonl `
  -BridgeRegWeight 0.001 `
  -BridgeRegSamples 256 `
  -BridgeRegMargin 0.05 `
  -BridgeRegMinScoreGap 2.0 `
  -Python E:\anaconda\envs\itmprec-rtx4080\python.exe
```

## IPG PreMatch Results

| Model | HR@5 | IoI@5 | IoR@5 | HR@10 | IoI@10 | IoR@10 | HR@15 | IoI@15 | IoR@15 | HR@20 | IoI@20 | IoR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original model_idx=1 | 0.4882 | 0.2062 | 68.0664 | 0.4860 | 0.2584 | 85.5083 | 0.4862 | 0.2568 | 79.0719 | 0.4873 | 0.2500 | 72.5948 |
| bridge_reg_w0.01 model_idx=41 | 0.4832 | 0.2008 | 67.3166 | 0.4828 | 0.2517 | 84.0375 | 0.4839 | 0.2495 | 77.8923 | 0.4836 | 0.2409 | 70.6275 |
| bridge_reg_w0.001 model_idx=42 | 0.4859 | 0.2062 | 70.6870 | 0.4842 | 0.2572 | 87.0161 | 0.4855 | 0.2560 | 81.2953 | 0.4861 | 0.2484 | 73.5388 |
| bridge_reg_extreme_w0.003 model_idx=43 | 0.4860 | 0.2049 | 70.3979 | 0.4842 | 0.2555 | 87.0075 | 0.4855 | 0.2528 | 79.7448 | 0.4861 | 0.2460 | 72.5525 |
| bridge_reg_extreme_w0.001 model_idx=44 | 0.4839 | 0.2025 | 66.6305 | 0.4826 | 0.2548 | 82.4909 | 0.4840 | 0.2531 | 76.3272 | 0.4831 | 0.2449 | 69.4791 |
| bridge_reg_w0.002 model_idx=45 | 0.4839 | 0.2021 | 66.6460 | 0.4835 | 0.2543 | 83.2722 | 0.4847 | 0.2541 | 78.2366 | 0.4836 | 0.2442 | 69.9603 |
| bridge_reg_topbottom_w0.001 model_idx=46 | 0.4827 | 0.1978 | 66.4283 | 0.4831 | 0.2478 | 82.0750 | 0.4839 | 0.2438 | 74.4266 | 0.4826 | 0.2366 | 69.8079 |

## Observation

The stronger bridge regularizer is too intrusive and hurts all IPG metrics. The weaker `0.001` setting is much healthier: HR/IoI are close to the original backbone, and IoR improves at @5, @10, @15, and @20. However, HR@20 and IoI@20 still remain slightly below the original model.

The follow-up runs show that the useful range is narrow. Increasing the rank-all weight to `0.002` already degrades the model, so this signal should stay weak. The hard-threshold `extreme` pair mode also fails to improve over `model_idx=42`; it keeps only 4098 pairs across 42 targets, so the regularizer loses target coverage and becomes less useful for long-horizon recommendation.

Current best: `bridge_reg_w0.001 model_idx=42`.

The top-bottom variant keeps broader target coverage than `extreme`, but it still underperforms the rank-all weak regularizer. This suggests that the backbone-side signal is currently best used as a very weak global ranking prior, rather than as a small set of hard positive/negative constraints.

## Combination With Risk Replanning

The best bridge-regularized checkpoint `model_idx=42` was also evaluated with prompt-v2 LLM risk replanning. The LIR settings disable direct profile/COT score bonuses (`alpha=0`, `beta=0`, `gamma=0`, `CotDisableBonus`) and only use LLM COT scores as a replan risk filter.

| Experiment | HR@5 | IoI@5 | IoR@5 | HR@10 | IoI@10 | IoR@10 | HR@15 | IoI@15 | IoR@15 | HR@20 | IoI@20 | IoR@20 | Accept@20 | Bridge@20 | RejectRecovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LIR baseline model_idx=42 | 0.4623 | 0.1933 | 51.9501 | 0.4679 | 0.2658 | 77.1582 | 0.4774 | 0.2805 | 80.2751 | 0.4726 | 0.2808 | 76.7954 | - | - | - |
| LIR promptv2 risk replan model_idx=42 | 0.4623 | 0.1933 | 51.9501 | 0.4679 | 0.2658 | 77.1601 | 0.4774 | 0.2805 | 80.2835 | 0.4726 | 0.2809 | 76.8067 | 0.5265 | 0.5647 | 0.4870 |
| LIR promptv2 risk replan loose model_idx=42 | 0.4623 | 0.1933 | 51.9501 | 0.4679 | 0.2658 | 77.1601 | 0.4774 | 0.2805 | 80.2835 | 0.4726 | 0.2809 | 76.8067 | 0.5265 | 0.5647 | 0.4870 |
| LIR promptv2 proactive replan r0.5/t0.5/m1.0 model_idx=42 | 0.4630 | 0.1935 | 52.1438 | 0.4674 | 0.2645 | 77.0359 | 0.4771 | 0.2778 | 78.9031 | 0.4724 | 0.2807 | 77.4504 | 0.5273 | 0.5665 | 0.4924 |
| LIR promptv2 proactive replan r0.5/t0.5/m0.15 model_idx=42 | 0.4631 | 0.1935 | 52.1457 | 0.4675 | 0.2647 | 77.0603 | 0.4772 | 0.2782 | 79.0333 | 0.4724 | 0.2811 | 77.5286 | 0.5273 | 0.5668 | 0.4754 |
| LIR promptv2 proactive replan r0.5/t0.5/m0.15/d0.2 model_idx=42 | 0.4626 | 0.1935 | 52.1413 | 0.4677 | 0.2648 | 77.1502 | 0.4773 | 0.2787 | 79.2898 | 0.4726 | 0.2806 | 77.2166 | 0.5264 | 0.5673 | 0.4755 |

The combination is valid, but the extra gain from prompt-v2 replanning is very small on top of `model_idx=42`: IoR@20 improves from `76.7954` to `76.8067`. The main gain here comes from the bridge-regularized backbone; prompt-v2 replanning triggers too rarely to substantially change the trajectory.

Adding proactive risk replanning changes this conclusion. Instead of waiting for a reject state, the replan filter can be activated when the backbone top candidates are close enough (`margin <= 0.15`) and the LLM marks a candidate as moderate/high risk with weak transition (`risk >= 3/5`, `transition <= 3/5`). This raises IoR@20 to `77.5286` while keeping IoI@20 close to the baseline. Lowering the penalty weight to `delta=0.2` is more conservative but loses IoR, so the current best replan setting is `delta=0.3`.
