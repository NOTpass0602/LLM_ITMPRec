# LIR-ITMPRec 可复现实验

该实验保留 ITMPRec 作为 backbone，并加入一个轻量级的 LLM 增强意图推理层：

- 基于文本 profile 的语义重排序
- 可选的缓存 CoT bridge score
- 基于规则的复杂反馈
- 反馈感知的动态重新规划

原始 ITMPRec 脚本保持不变。新的消融实验入口是 `src/test_LIR_PreMatch.py`。

## 必需的数据布局

以 `lastfm` 为例，文件应放置为：

```text
data/lastfm/
  lastfm.txt
  data.txt
  target_items.txt
  user_emb-v2-64.pt
  item_emb-v2-64.pt
  lir_profiles.jsonl          可选
  cot_bridge_cache.jsonl      可选
```

`lir_profiles.jsonl` 的记录示例：

```json
{"type":"user","id":1,"profile":"The user prefers fast-paced rock and electronic music."}
{"type":"item","id":25,"profile":"A punk rock track with energetic rhythm and aggressive vocals."}
```

profile 文件中的 item id 是从 1 开始的，和原始数据文件一致。重排序器内部会收到 ITMPRec 输出的从 0 开始的 id，并自动转换。

`cot_bridge_cache.jsonl` 的记录示例：

```json
{"user_id":1,"target_item":25,"candidate_item":8,"acceptance_score":4,"transition_score":5,"risk_score":1}
```

分数使用 1-5 分制。cache 是可选的；缺失记录会默认记为 0。

## 命令

先训练 ITMPRec backbone：

```bash
cd src
python main.py --data_dir ../data/lastfm/ --data_name lastfm --model_idx 1 --num_intent_clusters 32
```

通过新入口运行原始 baseline 行为：

```bash
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation baseline
```

运行消融实验：

```bash
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation profiles
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation cot
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation feedback
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation replan
```

`full` 表示同时启用 profiles、CoT cache、complex feedback 和 dynamic replanning：

```bash
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation full
```

## 输出指标

脚本会打印原始 ITMPRec 指标：

- hit_ratio at 5, 10, 15, 20
- IoI at 5, 10, 15, 20
- IoR at 5, 10, 15, 20

对于 LIR 消融实验，还会额外打印：

- semantic acceptability at 20
- bridge coherence at 20
- feedback action counts
- reject recovery rate
