# LIR 消融实验结果与分析

实验时间：2026-05-23  
数据集：ml100k  
Python：`E:\anaconda\envs\itmprec-rtx4080\python.exe`  
说明：本轮按要求跳过 `llmcot` 和 `llmfull`，因为 LLM cot/full 文件还没有准备到可正式评估的状态。

## 已完成实验

| 实验 | 说明 | 日志 |
| --- | --- | --- |
| `rules_baseline` | 不启用 LIR 增强，仅作为本轮 LIR 脚本内基线 | `rules_baseline.txt` |
| `rules_profiles` | 使用规则生成 profile | `rules_profiles.txt` |
| `rules_cot` | 使用规则生成 profile + cot score | `rules_cot.txt` |
| `rules_feedback` | 使用规则文件 + feedback simulator | `rules_feedback.txt` |
| `rules_replan` | 使用 feedback + replan | `rules_replan.txt` |
| `rules_full` | 规则版 full：profile + cot + feedback + replan | `rules_full.txt` |
| `llm_profiles` | 只使用 Ollama 生成的 LLM profile | `llm_profiles.txt` |

## 主要指标

| Experiment | HR@20 | IoI@20 | IoR@20 | Accept@20 | Bridge@20 | Feedback | RejectRecovery |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `rules_baseline` | 0.4738 | 0.2814 | 75.7238 | - | - | - | - |
| `rules_profiles` | 0.4734 | 0.2799 | 75.1256 | 0.2024 | 0.7342 | click=0, skip=0, reject=0 | 0.0000 |
| `rules_cot` | 0.4734 | 0.2799 | 75.1256 | 0.2024 | 0.7342 | click=0, skip=0, reject=0 | 0.0000 |
| `rules_feedback` | 0.4738 | 0.2814 | 75.7238 | 0.2011 | 0.7340 | click=450268, skip=492732, reject=0 | 0.0000 |
| `rules_replan` | 0.4738 | 0.2814 | 75.7238 | 0.2011 | 0.7340 | click=450268, skip=492732, reject=0 | 0.0000 |
| `rules_full` | 0.4732 | 0.2797 | 75.1482 | 0.2025 | 0.7343 | click=449883, skip=493116, reject=1 | 0.0000 |
| `llm_profiles` | 0.4729 | 0.2760 | 72.2593 | 0.2795 | 0.3102 | click=0, skip=0, reject=0 | 0.0000 |

## 相对本轮 baseline 的变化

| 实验 | ΔHR@20 | ΔIoI@20 | ΔIoR@20 |
| --- | ---: | ---: | ---: |
| `rules_profiles` | -0.0004 | -0.0015 | -0.5982 |
| `rules_cot` | -0.0004 | -0.0015 | -0.5982 |
| `rules_feedback` | +0.0000 | +0.0000 | +0.0000 |
| `rules_replan` | +0.0000 | +0.0000 | +0.0000 |
| `rules_full` | -0.0006 | -0.0017 | -0.5756 |
| `llm_profiles` | -0.0009 | -0.0054 | -3.4645 |

## 结论

1. 规则版 profile/cot/full 暂时没有带来提升，整体略低于 baseline。

   `rules_profiles`、`rules_cot`、`rules_full` 的 HR@20 和 IoR@20 都比本轮 baseline 略低。这说明当前规则文本虽然能被系统读取，但它对最终排序的帮助不够，甚至会把部分原本靠 backbone 排得不错的物品轻微扰动掉。

2. `rules_profiles` 与 `rules_cot` 结果几乎完全一致。

   这说明当前 cot score 没有形成额外有效信号。可能原因包括：cot cache 的候选对与实际评估时进入 rerank 的候选对覆盖不完全；cot 权重偏小；规则 cot 本身主要由 genre overlap 和 popularity 构成，和 profile scorer 的信息高度重复。

3. `rules_feedback` 与 `rules_replan` 完全一致是合理但不理想的。

   本轮 feedback 产生了大量 click/skip，但没有 reject。replan 的主要作用是针对 reject 做恢复和惩罚；没有 reject 时，replan 实际上没有新的行为空间，所以指标保持一致。`rules_full` 只有 1 个 reject，影响也几乎可以忽略。

4. `llm_profiles` 的 Accept@20 明显更高，但最终推荐指标反而下降。

   LLM profile 相比规则 profile 更像自然语言用户画像，所以 Accept@20 从约 0.20 提升到 0.2795；但 Bridge@20 从约 0.734 降到 0.3102，IoR@20 也明显下降。这更像是 scorer 不会真正理解 LLM 语义，而不是 LLM profile 没价值。当前基于词面 overlap/cosine 的轻量 scorer 对自然语言 profile 利用不足。

## 改进建议

1. 优先把 profile/cot 的文本打成真正的语义 embedding。

   现在的 profile scorer 更偏词面匹配。建议用本地 embedding 模型，例如 `BAAI/bge-small-en-v1.5`、`bge-base-en-v1.5`、`sentence-transformers/all-MiniLM-L6-v2`，或者 Ollama embedding 模型，预先生成 user/item/profile/cot embedding，再用向量相似度参与 rerank。

2. cot cache 不要全量盲生成，改成“评估候选对定向生成”。

   当前规则 cot 文件虽然有 2357500 行，但未必覆盖真正进入 rerank 的 `(target_item, candidate_item)` 组合。更好的做法是先从评估脚本中导出实际 target 与 top-k 候选对，再只为这些组合生成 cot score。这样文件更小，命中率更高，也更适合之后用 LLM 生成。

3. 调整 feedback/replan 的 reject 机制。

   现在 reject 几乎不出现，导致 replan 模块无法体现价值。可以降低 reject 阈值，或者让低 acceptability、低 bridge score、低用户偏好匹配的候选更容易触发 reject。只有 reject 有足够数量，replan 才能真正被检验。

4. 在 scorer 修好后再调大 LIR 权重。

   目前不建议单纯加大权重，因为现有 LIR 信号还不够可靠。等 embedding scorer 和 cot 命中率改善后，再尝试例如 `lir_alpha/lir_beta/lir_gamma` 从 0.2 提到 0.4、0.5 做网格搜索。

5. baseline 比较要统一脚本。

   之前 `test_IPG_PreMatch.py` 的 IPG 结果和本轮 `test_LIR_PreMatch.py` 的 baseline 不完全一样，主要可能来自 target item 选择顺序/随机状态差异。因此当前最可靠的对比是本轮 LIR 消融内部横向比较；后续如果要写论文式表格，应该固定 target list 并在同一脚本内统一评估。

## 下一步建议顺序

1. 先改 profile scorer：从词面相似度升级为 embedding 相似度。
2. 导出真实评估候选对，只对这些 pair 生成 cot score。
3. 重新跑 `profiles`、`cot`、`full`。
4. 调整 reject 阈值后重新跑 `feedback`、`replan`、`full`。
5. 等 LLM cot 文件准备好，再跑 `llmcot` 和 `llmfull`。

## BGE 语义 embedding 补充实验

本地模型路径：`E:\huggingface\models\bge-small-en-v1.5`  
生成文件：`data\ml100k\profile_semantic_emb_ollama_bge-small-en-v1.5.pt`  
实验日志：`results\lir_ablation\llm_profiles_bge.txt`

| Experiment | HR@20 | IoI@20 | IoR@20 | Accept@20 | Bridge@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llm_profiles` | 0.4729 | 0.2760 | 72.2593 | 0.2795 | 0.3102 |
| `llm_profiles_bge` | 0.4734 | 0.2788 | 74.3488 | 0.5265 | 0.5661 |
| `rules_baseline` | 0.4738 | 0.2814 | 75.7238 | - | - |

结论：BGE 语义 embedding 明显改善了 LLM profile 的语义可用性。相比纯词面 `llm_profiles`，`Accept@20` 和 `Bridge@20` 都大幅提高，HR@20、IoI@20、IoR@20 也同步回升。不过它仍然略低于本轮 baseline，说明“语义理解”这一步是有效的，但当前 rerank 加权方式还不够好。下一步应优先调 `lir_alpha/lir_beta/lir_gamma`，并考虑只保留 acceptability/bridge 中更稳定的一个信号做消融。

## Replan 修复实验

改动位置：

| 文件 | 改动 |
| --- | --- |
| `src\llm_itmprec\complex_feedback.py` | 收紧 reject 判定，让未点击且明显不匹配的候选产生 reject |
| `src\llm_itmprec\replanner.py` | 为 reject 后的语义路径惩罚增加状态权重 |
| `src\llm_itmprec\runner.py` | 不再只惩罚同一个 rejected item，而是惩罚与 rejected item 语义高度相似的候选 |

| Experiment | HR@20 | IoI@20 | IoR@20 | Feedback | RejectRecovery |
| --- | ---: | ---: | ---: | --- | ---: |
| `rules_baseline` | 0.4738 | 0.2814 | 75.7238 | - | - |
| `rules_replan` 修复前 | 0.4738 | 0.2814 | 75.7238 | click=450268, skip=492732, reject=0 | 0.0000 |
| `rules_replan_replanfix_v2` | 0.4737 | 0.2813 | 75.8757 | click=450172, skip=435830, reject=56998 | 0.4671 |

结论：修复后 replan 已经被有效激活。相比修复前 `reject=0`，现在产生约 5.7 万次 reject，并且 `RejectRecovery` 达到 0.4671。推荐效果上，HR@20 基本持平，IoR@20 从 75.7238 小幅提升到 75.8757。提升幅度还不大，但它证明了 replan 机制现在可以真实改变后续推荐，而不是空转。

需要注意的是，第一版较激进阈值曾产生约 40 万次 reject，导致 IoR@20 降到 74.5013；因此 reject 不能越多越好。当前更适合把 replan 作为“保守纠偏模块”，后续可以结合 LLM bridge score，让 reject 后的替代路径选择更有目标导向。

## LLM Pair CoT 过滤实验

为避免 user-conditioned CoT 生成规模过大，本轮导出了真实评估会出现的高频 pair：`data\ml100k\lir_bridge_pairs_top50.jsonl`，共 50 个 target、每个 target 50 个 candidate，总计 2500 条。随后用 Ollama 生成 `data\ml100k\cot_bridge_cache_ollama_pair_top50.jsonl`。

原始 LLM pair CoT 分布非常保守：2500 条中 `risk_score=5` 有 1584 条，最常见模式是 `(acceptance=1, transition=1, risk=5)`。因此直接线性使用风险惩罚会压低 backbone 候选。已将 CoT 使用方式改为高置信正向过滤：只有 `transition>=0.5` 且 `risk<=0.5` 时才加入 CoT bonus，默认不再扣除 risk penalty。

| Experiment | HR@20 | IoI@20 | IoR@20 | Accept@20 | Bridge@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `llm_profiles_bge` | 0.4734 | 0.2788 | 74.3488 | 0.5265 | 0.5661 |
| `llm_cot_pair_unfiltered` | 0.4728 | 0.2769 | 74.4167 | 0.5296 | 0.5691 |
| `llm_cot_pair_filtered` | 0.4733 | 0.2794 | 74.9244 | 0.5265 | 0.5660 |
| `llm_cot_pair_relaxed` | 0.4733 | 0.2795 | 74.8597 | 0.5270 | 0.5671 |
| `rules_baseline` | 0.4738 | 0.2814 | 75.7238 | - | - |

结论：高置信过滤方向是有效的，`llm_cot_pair_filtered` 明显好于未过滤的直接线性 CoT，IoR@20 从 74.4167 提升到 74.9244。进一步放宽阈值后，正向 pair 覆盖从 71 增加到 320，但 IoR@20 没有继续提高，说明简单放宽阈值不是主要突破口。当前更应该从 prompt 源头校正 LLM 对 bridge 的理解，特别是强调“桥接不等于相似”，让模型识别那些虽不相似但可作为过渡步骤的候选。

已更新 pair CoT 生成 prompt：明确告诉 LLM 不要因为 genre 不同就判定高风险，并要求对合理但不完美的过渡候选给出中等 transition 和较低/中等 risk。下一轮建议生成新文件 `cot_bridge_cache_ollama_pair_top50_promptv2.jsonl`，避免覆盖已有结果，便于横向比较 prompt v1 与 prompt v2。

## LLM 风险纠偏实验

为验证“LLM 更适合作为动态纠偏模块，而不是静态 rerank 加分器”，新增了两个开关：

| 参数 | 含义 |
| --- | --- |
| `--lir_disable_cot_bonus` | 关闭 CoT 对最终排序的直接加分 |
| `--lir_cot_replan_filter` | 只在用户已经出现 reject 后，用高 risk、低 transition 的 CoT 结果过滤后续候选 |

| Experiment | HR@20 | IoI@20 | IoR@20 | Feedback | RejectRecovery |
| --- | ---: | ---: | ---: | --- | ---: |
| `rules_baseline` | 0.4738 | 0.2814 | 75.7238 | - | - |
| `rules_replan_replanfix_v2` | 0.4737 | 0.2813 | 75.8757 | click=450172, skip=435830, reject=56998 | 0.4671 |
| `llm_full_risk_replan_filter` | 0.4728 | 0.2763 | 74.5841 | click=449664, skip=403394, reject=89942 | 0.4793 |
| `llm_risk_replan_only` | 0.4734 | 0.2790 | 75.9286 | click=450106, skip=398650, reject=94244 | 0.4796 |

结论：`llm_full_risk_replan_filter` 仍然较差，说明 BGE profile 与 CoT risk 同时参与时，语义信号整体仍会扰动 backbone。更干净的 `llm_risk_replan_only` 关闭了 profile/cot 的直接加权，只保留 LLM risk 作为 replan 后的动态过滤信号，IoR@20 达到 75.9286，略高于 baseline 和规则 replan。这支持当前阶段的关键判断：LLM 信号不适合直接线性重排，但可以作为复杂反馈后的路径风险识别与纠偏信号。

### Margin Gate 追加实验

为了减少 LLM 风险过滤对 HR/IoI 的误伤，新增两个保护条件：

| 参数 | 作用 |
| --- | --- |
| `CotReplanMinRejects` | 连续 reject 至少达到该值后才启用 LLM risk filter |
| `CotReplanMargin` | 只有 backbone top 候选分差较小时才启用 LLM risk filter |

| Experiment | HR@20 | IoI@20 | IoR@20 | RejectRecovery |
| --- | ---: | ---: | ---: | ---: |
| `rules_baseline` | 0.4738 | 0.2814 | 75.7238 | - |
| `llm_risk_replan_only` | 0.4734 | 0.2790 | 75.9286 | 0.4796 |
| `llm_risk_replan_margin_r2_m005` | 0.4738 | 0.2810 | 75.6763 | 0.4800 |
| `llm_risk_replan_margin_r2_m010` | 0.4737 | 0.2811 | 75.7875 | 0.4807 |
| `llm_risk_replan_margin_r2_m015` | 0.4737 | 0.2811 | 75.8043 | 0.4809 |
| `llm_risk_replan_margin_r2_m015_d020` | 0.4738 | 0.2810 | 75.5968 | 0.4797 |

结论：margin gate 能明显保护 HR/IoI，但会削弱 IoR 的提升。当前旧 prompt cot 下，`CotReplanMinRejects=2, CotReplanMargin=0.15, LirDelta=0.30` 是较均衡的设置：HR/IoI 基本贴近 baseline，IoR@20 小幅高于 baseline。降低 `LirDelta` 到 0.20 反而削弱了 IoR，说明风险纠偏需要一定惩罚强度。下一步应等待 prompt v2 cot，观察风险分布更平滑后，这组 gate 是否能同时改善三项指标。
