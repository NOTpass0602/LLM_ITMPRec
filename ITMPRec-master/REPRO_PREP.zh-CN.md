# 复现准备说明

本仓库已经准备了原始数据转换脚本，并补充了 GraphAU embedding 导出支持。

## 已准备的数据文件

以下数据集已经完成转换：

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

每个 ITMPRec 数据目录包含：

```text
<dataset>.txt
data.txt
target_items.txt
```

每个 GraphAU 数据目录包含：

```text
train.txt
val.txt
test.txt
```

数据转换脚本为：

```bash
python tools/prepare_repro_data.py --datasets ml100k lastfm steam douban_movie
```

## 环境

当前 Windows shell 没有可直接用于训练的 Python/Conda 环境。检测到的 Codex Python 是 3.12，并且缺少 `torch`、`faiss`、`dgl`、`scipy`、`tqdm`。

推荐环境是 Linux/WSL2/Conda：

```bash
conda env create -f environment.yml
conda activate itmprec
powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1
```

如果只有 CPU，也可以安装 CPU 版 PyTorch、DGL、FAISS，但运行速度会慢很多。

当前仓库中的 `modules` 和 `trainers` 只有 Python 3.8 字节码。运行 ITMPRec 前需要准备 sourceless import：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_pyc_imports.ps1
```

## 第一步：训练 GraphAU 并导出 Embeddings

GraphAU 已被修改，支持 `--data_root` 和 `--export_dir`。

建议先跑 ML-100K，因为它是最小的完整 sanity check：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_graphau.ps1 -Dataset ml100k -Gpu 0
```

期望输出：

```text
data/ml100k/user_emb-v2-64.pt
data/ml100k/item_emb-v2-64.pt
```

之后可以将 `-Dataset` 改成下面的数据集继续运行：

```text
lastfm
steam
douban_movie
```

第一次尝试建议不要先跑 `steam`，因为它明显更大。

## 第二步：训练 ITMPRec

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_itmprec.ps1 -Dataset ml100k -Gpu 0
```

期望输出：

```text
src/output/ICLRec-ml100k-1.pt
```

## 第三步：评估 ITMPRec Baseline

```powershell
powershell -ExecutionPolicy Bypass -File scripts/eval_ipg_prematch.ps1 -Dataset ml100k -Gpu 0
```

## 第四步：运行 LIR 消融实验

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

然后分别运行：

```text
--lir_ablation profiles
--lir_ablation cot
--lir_ablation feedback
--lir_ablation replan
--lir_ablation full
```

## 本地转换后的数据规模

```text
ml100k:       943 users, 1682 items, 100000 interactions
lastfm:      1892 users, 12523 items, 71064 interactions
steam:       68403 users, 10050 items, 3285246 interactions
douban_movie:2712 users, 34893 items, 1278401 interactions
```

这些数据是从项目目录中现有原始文件确定性转换得到的。如果论文作者使用了额外过滤或私有预处理，统计量可能和论文不完全一致。
