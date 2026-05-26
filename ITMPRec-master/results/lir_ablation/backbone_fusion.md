# LLM Semantic Backbone Fusion

## Setup

This experiment fuses the original GraphAU item embedding with BGE semantic item profile embeddings, then retrains the ITMPRec backbone from the fused item embedding initialization.

Fusion file:

`data\ml100k\item_emb-llm-fused-w20-64.pt`

Fusion metadata:

| Field | Value |
| --- | ---: |
| semantic_weight | 0.20 |
| ridge | 0.001 |
| mean_cos_graph_semantic_projected | 0.7070 |
| mean_cos_graph_fused | 0.8511 |

Retrained checkpoint:

`src\output\ICLRec-ml100k-21.pt`

## IPG PreMatch

| Model | HR@5 | IoI@5 | IoR@5 | HR@10 | IoI@10 | IoR@10 | HR@15 | IoI@15 | IoR@15 | HR@20 | IoI@20 | IoR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original model_idx=1 | 0.4882 | 0.2062 | 68.0664 | 0.4860 | 0.2584 | 85.5083 | 0.4862 | 0.2568 | 79.0719 | 0.4873 | 0.2500 | 72.5948 |
| fused_w20 model_idx=21 | 0.4697 | 0.2121 | 71.5253 | 0.4691 | 0.2569 | 86.3811 | 0.4700 | 0.2544 | 81.1329 | 0.4679 | 0.2425 | 74.7545 |

Note: PreMatch target selection depends on the model checkpoint, so the exact target list changes after backbone retraining. Treat this table as a directional comparison, not a perfectly controlled ablation.

## LIR PreMatch, Same Fused Checkpoint

| Experiment | HR@5 | IoI@5 | IoR@5 | HR@10 | IoI@10 | IoR@10 | HR@15 | IoI@15 | IoR@15 | HR@20 | IoI@20 | IoR@20 | Accept@20 | Bridge@20 | Feedback | RejectRecovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| fused_w20_lir_baseline | 0.4766 | 0.1720 | 50.8058 | 0.4749 | 0.2286 | 68.8183 | 0.4702 | 0.2347 | 70.0571 | 0.4680 | 0.2307 | 68.2548 | - | - | - | - |
| fused_w20_llm_risk_replan_promptv1 | 0.4752 | 0.1719 | 50.8034 | 0.4733 | 0.2281 | 68.8664 | 0.4690 | 0.2338 | 69.9676 | 0.4670 | 0.2290 | 68.0229 | 0.5248 | 0.5656 | click=440940, skip=390532, reject=111528 | 0.4660 |

## Observation

The simple initialization-level fusion is not yet a clean improvement. In IPG it improves IoR@20 compared with the earlier original IPG run, but HR and IoI drop. In the LIR script, the fused checkpoint has a much lower IoI/IoR baseline than the original checkpoint, and LLM risk replan with prompt v1 does not recover the loss.

This suggests the next useful direction is not to increase the fusion weight blindly. A safer next step is to use LLM semantic information as a regularization or auxiliary training target, or to gate semantic fusion only for uncertain candidate pairs instead of replacing the item embedding initialization globally.

