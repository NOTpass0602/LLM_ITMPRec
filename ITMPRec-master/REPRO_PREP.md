# Reproduction Preparation Notes

This repository has been prepared with raw-data conversion scripts and
GraphAU embedding export support.

## Prepared Dataset Files

The following datasets have been converted:

```text
data/ml100k/
data/lastfm/
data/steam/
data/douban_movie/

data_graphau/datasets/ml100k/
data_graphau/datasets/lastfm/
data_graphau/datasets/steam/
data_graphau/datasets/douban_movie/
```

Each ITMPRec dataset folder contains:

```text
<dataset>.txt
data.txt
target_items.txt
```

Each GraphAU dataset folder contains:

```text
train.txt
val.txt
test.txt
```

The conversion script is:

```bash
python tools/prepare_repro_data.py --datasets ml100k lastfm steam douban_movie
```

## Environment

The current Windows shell does not have a usable Python/Conda environment for
training. The detected Codex Python is 3.12 and lacks `torch`, `faiss`, `dgl`,
`scipy`, and `tqdm`.

Recommended environment is Linux/WSL2/Conda:

```bash
conda env create -f environment.yml
conda activate itmprec
powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1
```

If only CPU is available, use a CPU-compatible PyTorch/DGL/FAISS setup. Results
will be much slower.

The repository currently has `modules` and `trainers` only as Python 3.8
bytecode. Prepare sourceless imports before running ITMPRec:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_pyc_imports.ps1
```

## Step 1: Train GraphAU And Export Embeddings

GraphAU has been patched to accept `--data_root` and `--export_dir`.

Run ML-100K first because it is the smallest complete sanity check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_graphau.ps1 -Dataset ml100k -Gpu 0
```

Expected outputs:

```text
data/ml100k/user_emb-v2-64.pt
data/ml100k/item_emb-v2-64.pt
```

Repeat by changing `--dataset` and `--export_dir` for:

```text
lastfm
steam
douban_movie
```

For the first trial, avoid `steam` because it is much larger.

## Step 2: Train ITMPRec

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_itmprec.ps1 -Dataset ml100k -Gpu 0
```

Expected output:

```text
src/output/ICLRec-ml100k-1.pt
```

## Step 3: Evaluate ITMPRec Baseline

```powershell
powershell -ExecutionPolicy Bypass -File scripts/eval_ipg_prematch.ps1 -Dataset ml100k -Gpu 0
```

## Step 4: Run LIR Ablations

```bash
python test_LIR_PreMatch.py \
  --data_dir ../data/ml100k/ \
  --data_name ml100k \
  --num_users 943 \
  --num_items 1682 \
  --model_idx 1 \
  --num_intent_clusters 32 \
  --user_emb_path ../data/ml100k/user_emb-v2-64.pt \
  --item_emb_path ../data/ml100k/item_emb-v2-64.pt \
  --lir_ablation baseline
```

Then run:

```text
--lir_ablation profiles
--lir_ablation cot
--lir_ablation feedback
--lir_ablation replan
--lir_ablation full
```

## Dataset Sizes After Local Conversion

```text
ml100k:       943 users, 1682 items, 100000 interactions
lastfm:      1892 users, 12523 items, 71064 interactions
steam:       68403 users, 10050 items, 3285246 interactions
douban_movie:2712 users, 34893 items, 1278401 interactions
```

These are deterministic conversions from the raw files currently present in the
project folder. They may not exactly match the paper's privately preprocessed
statistics if the authors used additional filtering.
