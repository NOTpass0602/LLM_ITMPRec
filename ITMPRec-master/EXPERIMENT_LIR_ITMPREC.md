# LIR-ITMPRec Reproducible Experiment

This experiment keeps ITMPRec as the backbone and adds a lightweight
LLM-enhanced intention reasoning layer:

- textual profile based semantic reranking
- optional cached CoT bridge scores
- rule-based complex feedback
- feedback-aware dynamic replanning

The original ITMPRec scripts are kept intact. Use `src/test_LIR_PreMatch.py`
for the new ablation experiments.

## Required Data Layout

For a dataset such as `lastfm`, place files as:

```text
data/lastfm/
  lastfm.txt
  data.txt
  target_items.txt
  user_emb-v2-64.pt
  item_emb-v2-64.pt
  lir_profiles.jsonl          optional
  cot_bridge_cache.jsonl      optional
```

`lir_profiles.jsonl` records may look like:

```json
{"type":"user","id":1,"profile":"The user prefers fast-paced rock and electronic music."}
{"type":"item","id":25,"profile":"A punk rock track with energetic rhythm and aggressive vocals."}
```

Item ids in profile files are one-based, matching the raw data files. The
reranker internally receives zero-based ids from ITMPRec and converts them.

`cot_bridge_cache.jsonl` records may look like:

```json
{"user_id":1,"target_item":25,"candidate_item":8,"acceptance_score":4,"transition_score":5,"risk_score":1}
```

Scores use a 1-5 scale. The cache is optional; missing records default to zero.

## Commands

Train the ITMPRec backbone first:

```bash
cd src
python main.py --data_dir ../data/lastfm/ --data_name lastfm --model_idx 1 --num_intent_clusters 32
```

Run original behavior through the new entry:

```bash
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation baseline
```

Run ablations:

```bash
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation profiles
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation cot
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation feedback
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation replan
```

The `full` setting is equivalent to enabling profiles, CoT cache, complex
feedback, and dynamic replanning:

```bash
python test_LIR_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32 --lir_ablation full
```

## Reported Metrics

The script prints the original ITMPRec metrics:

- hit_ratio at 5, 10, 15, 20
- IoI at 5, 10, 15, 20
- IoR at 5, 10, 15, 20

For LIR ablations it also prints:

- semantic acceptability at 20
- bridge coherence at 20
- feedback action counts
- reject recovery rate

