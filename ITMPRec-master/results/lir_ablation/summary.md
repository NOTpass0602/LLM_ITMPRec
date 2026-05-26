| Experiment | HR@5 | IoI@5 | IoR@5 | HR@10 | IoI@10 | IoR@10 | HR@15 | IoI@15 | IoR@15 | HR@20 | IoI@20 | IoR@20 | Accept@20 | Bridge@20 | Feedback | RejectRecovery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm_profiles | 0.4617 | 0.1957 | 52.6484 | 0.4693 | 0.2645 | 75.6979 | 0.4783 | 0.2764 | 75.9192 | 0.4729 | 0.2760 | 72.2593 | 0.2795 | 0.3102 | {'click': 0, 'skip': 0, 'reject': 0} | 0.0000 |
| llm_profiles_bge | 0.4624 | 0.1966 | 53.3215 | 0.4694 | 0.2666 | 77.1082 | 0.4784 | 0.2789 | 77.7000 | 0.4734 | 0.2788 | 74.3488 | 0.5265 | 0.5661 | {'click': 0, 'skip': 0, 'reject': 0} | 0.0000 |
| llm_cot_pair_filtered | 0.4623 | 0.1964 | 53.2451 | 0.4693 | 0.2665 | 77.0799 | 0.4783 | 0.2790 | 77.7030 | 0.4733 | 0.2794 | 74.9244 | 0.5265 | 0.5660 | {'click': 0, 'skip': 0, 'reject': 0} | 0.0000 |
| llm_cot_pair_relaxed | 0.4623 | 0.1964 | 53.2024 | 0.4694 | 0.2666 | 77.1504 | 0.4784 | 0.2791 | 77.7059 | 0.4733 | 0.2795 | 74.8597 | 0.5270 | 0.5671 | {'click': 0, 'skip': 0, 'reject': 0} | 0.0000 |
| llm_full_risk_replan_filter | 0.4636 | 0.1965 | 53.3541 | 0.4692 | 0.2661 | 76.9158 | 0.4779 | 0.2771 | 77.1227 | 0.4728 | 0.2763 | 74.5841 | 0.5317 | 0.5790 | {'click': 449664, 'skip': 403394, 'reject': 89942} | 0.4793 |
| llm_risk_replan_only | 0.4632 | 0.1979 | 54.1693 | 0.4697 | 0.2682 | 77.9185 | 0.4787 | 0.2801 | 79.1358 | 0.4734 | 0.2790 | 75.9286 | 0.5278 | 0.5761 | {'click': 450106, 'skip': 398650, 'reject': 94244} | 0.4796 |
| llm_risk_replan_margin_r2_m005 | 0.4628 | 0.1980 | 54.1639 | 0.4697 | 0.2685 | 77.8421 | 0.4791 | 0.2811 | 78.9679 | 0.4738 | 0.2810 | 75.6763 | 0.5250 | 0.5664 | {'click': 450294, 'skip': 342543, 'reject': 150163} | 0.4800 |
| llm_risk_replan_margin_r1_m005 | 0.4630 | 0.1979 | 54.1995 | 0.4697 | 0.2685 | 77.9587 | 0.4790 | 0.2807 | 78.7354 | 0.4737 | 0.2808 | 75.5500 | 0.5255 | 0.5678 | {'click': 450240, 'skip': 355013, 'reject': 137747} | 0.4802 |
| llm_risk_replan_margin_r2_m010 | 0.4629 | 0.1980 | 54.1542 | 0.4696 | 0.2685 | 77.8240 | 0.4790 | 0.2811 | 78.9349 | 0.4737 | 0.2811 | 75.7875 | 0.5251 | 0.5663 | {'click': 450256, 'skip': 345070, 'reject': 147674} | 0.4807 |
| llm_risk_replan_margin_r2_m015 | 0.4629 | 0.1980 | 54.1571 | 0.4695 | 0.2685 | 77.8453 | 0.4789 | 0.2811 | 78.8805 | 0.4737 | 0.2811 | 75.8043 | 0.5251 | 0.5663 | {'click': 450228, 'skip': 346062, 'reject': 146710} | 0.4809 |
| llm_risk_replan_margin_r2_m015_d020 | 0.4629 | 0.1979 | 54.1557 | 0.4697 | 0.2685 | 77.8659 | 0.4790 | 0.2811 | 78.8759 | 0.4738 | 0.2810 | 75.5968 | 0.5253 | 0.5657 | {'click': 450266, 'skip': 342997, 'reject': 149737} | 0.4797 |
| bridge_reg_model42_baseline | 0.4623 | 0.1933 | 51.9501 | 0.4679 | 0.2658 | 77.1582 | 0.4774 | 0.2805 | 80.2751 | 0.4726 | 0.2808 | 76.7954 |  |  |  |  |
| bridge_reg_model42_promptv2_risk_replan | 0.4623 | 0.1933 | 51.9501 | 0.4679 | 0.2658 | 77.1601 | 0.4774 | 0.2805 | 80.2835 | 0.4726 | 0.2809 | 76.8067 | 0.5265 | 0.5647 | {'click': 449522, 'skip': 483613, 'reject': 9865} | 0.4870 |
| bridge_reg_model42_promptv2_risk_replan_loose | 0.4623 | 0.1933 | 51.9501 | 0.4679 | 0.2658 | 77.1601 | 0.4774 | 0.2805 | 80.2835 | 0.4726 | 0.2809 | 76.8067 | 0.5265 | 0.5647 | {'click': 449522, 'skip': 483613, 'reject': 9865} | 0.4870 |
| bridge_reg_model42_promptv2_proactive_replan_r05_t05 | 0.4630 | 0.1935 | 52.1438 | 0.4674 | 0.2645 | 77.0359 | 0.4771 | 0.2778 | 78.9031 | 0.4724 | 0.2807 | 77.4504 | 0.5273 | 0.5665 | {'click': 449466, 'skip': 490131, 'reject': 3403} | 0.4924 |
| bridge_reg_model42_promptv2_proactive_replan_r05_t05_m015 | 0.4631 | 0.1935 | 52.1457 | 0.4675 | 0.2647 | 77.0603 | 0.4772 | 0.2782 | 79.0333 | 0.4724 | 0.2811 | 77.5286 | 0.5273 | 0.5668 | {'click': 449513, 'skip': 488871, 'reject': 4616} | 0.4754 |
| bridge_reg_model42_promptv2_proactive_replan_r05_t05_m015_d020 | 0.4626 | 0.1935 | 52.1413 | 0.4677 | 0.2648 | 77.1502 | 0.4773 | 0.2787 | 79.2898 | 0.4726 | 0.2806 | 77.2166 | 0.5264 | 0.5673 | {'click': 449494, 'skip': 487654, 'reject': 5852} | 0.4755 |
| rules_baseline | 0.4623 | 0.1979 | 54.1689 | 0.4696 | 0.2685 | 77.8265 | 0.4790 | 0.2813 | 79.0278 | 0.4738 | 0.2814 | 75.7238 |  |  |  |  |
| rules_cot | 0.4627 | 0.1974 | 53.7515 | 0.4695 | 0.2674 | 77.2811 | 0.4785 | 0.2800 | 78.0019 | 0.4734 | 0.2799 | 75.1256 | 0.2024 | 0.7342 | {'click': 0, 'skip': 0, 'reject': 0} | 0.0000 |
| rules_feedback | 0.4623 | 0.1979 | 54.1689 | 0.4696 | 0.2685 | 77.8265 | 0.4790 | 0.2813 | 79.0278 | 0.4738 | 0.2814 | 75.7238 | 0.2011 | 0.7340 | {'click': 450268, 'skip': 492732, 'reject': 0} | 0.0000 |
| rules_full | 0.4622 | 0.1974 | 53.7373 | 0.4693 | 0.2673 | 77.2390 | 0.4783 | 0.2797 | 77.6004 | 0.4732 | 0.2797 | 75.1482 | 0.2025 | 0.7343 | {'click': 449883, 'skip': 493116, 'reject': 1} | 0.0000 |
| rules_profiles | 0.4627 | 0.1974 | 53.7502 | 0.4695 | 0.2674 | 77.2811 | 0.4785 | 0.2800 | 78.0019 | 0.4734 | 0.2799 | 75.1256 | 0.2024 | 0.7342 | {'click': 0, 'skip': 0, 'reject': 0} | 0.0000 |
| rules_replan | 0.4623 | 0.1979 | 54.1689 | 0.4696 | 0.2685 | 77.8265 | 0.4790 | 0.2813 | 79.0278 | 0.4738 | 0.2814 | 75.7238 | 0.2011 | 0.7340 | {'click': 450268, 'skip': 492732, 'reject': 0} | 0.0000 |
| rules_replan_replanfix_v2 | 0.4627 | 0.1979 | 54.1407 | 0.4697 | 0.2685 | 77.8815 | 0.4790 | 0.2811 | 78.7469 | 0.4737 | 0.2813 | 75.8757 | 0.2015 | 0.7339 | {'click': 450172, 'skip': 435830, 'reject': 56998} | 0.4671 |

## Backbone Fusion Notes

Detailed backbone fusion results are in `results\lir_ablation\backbone_fusion.md`.

The first fused checkpoint is `src\output\ICLRec-ml100k-21.pt`, initialized with `data\ml100k\item_emb-llm-fused-w20-64.pt`.

| Experiment | HR@20 | IoI@20 | IoR@20 | Note |
| --- | ---: | ---: | ---: | --- |
| IPG original model_idx=1 | 0.4873 | 0.2500 | 72.5948 | earlier IPG run |
| IPG fused_w20 model_idx=21 | 0.4679 | 0.2425 | 74.7545 | target list differs after prematch |
| LIR fused_w20 baseline model_idx=21 | 0.4680 | 0.2307 | 68.2548 | controlled within fused checkpoint |
| LIR fused_w20 risk replan promptv1 model_idx=21 | 0.4670 | 0.2290 | 68.0229 | LLM risk replan did not recover the fused baseline |

## Bridge Ranking Regularization Notes

Detailed bridge regularization results are in `results\lir_ablation\bridge_regularization.md`.

| Experiment | HR@20 | IoI@20 | IoR@20 | Note |
| --- | ---: | ---: | ---: | --- |
| IPG original model_idx=1 | 0.4873 | 0.2500 | 72.5948 | earlier IPG run |
| IPG bridge_reg_w0.01 model_idx=41 | 0.4836 | 0.2409 | 70.6275 | regularizer too strong |
| IPG bridge_reg_w0.001 model_idx=42 | 0.4861 | 0.2484 | 73.5388 | weak regularizer improves IoR while mostly preserving HR/IoI |
| IPG bridge_reg_extreme_w0.003 model_idx=43 | 0.4861 | 0.2460 | 72.5525 | high-confidence hard thresholds lose coverage |
| IPG bridge_reg_extreme_w0.001 model_idx=44 | 0.4831 | 0.2449 | 69.4791 | weaker extreme setting still worse |
| IPG bridge_reg_w0.002 model_idx=45 | 0.4836 | 0.2442 | 69.9603 | rank-all weight above 0.001 is too strong |
| IPG bridge_reg_topbottom_w0.001 model_idx=46 | 0.4826 | 0.2366 | 69.8079 | top-bottom pair construction is too restrictive |
