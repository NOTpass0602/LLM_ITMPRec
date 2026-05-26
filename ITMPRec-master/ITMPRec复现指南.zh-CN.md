# ITMPRec Backbone 复现指南

这份指南面向当前项目目录，目标是先把 **ITMPRec 作为 backbone** 跑通，再根据需要继续做 LIR/LLM 增强实验。

## 你现在这个项目的关键位置

```text
ITMPRec-master/ITMPRec-master/
  README.md                         原始英文说明
  REPRO_PREP.md                     已整理过的复现准备说明
  EXPERIMENT_LIR_ITMPREC.md         LIR-ITMPRec 消融说明
  environment.yml                   推荐 Conda 环境
  scripts/                          一键运行脚本
  src/                              ITMPRec 训练与评估入口
  data/                             ITMPRec 格式数据
  data_graphau/datasets/            GraphAU 格式数据

ITMPRec-master/GraphAU-main/GraphAU-main/
  main.py                           GraphAU 训练入口
```

当前最建议先用 `ml100k` 复现，因为它最小、最快，也最适合检查环境和路径。

## 0. 环境建议

推荐在 WSL2/Linux + Conda 中跑，尤其是 RTX 4080 这种新卡。原因是：

- 依赖比较老：PyTorch 1.9.0、CUDA 11.1、DGL、faiss-gpu。
- `src/modules.pyc` 和 `src/trainers.pyc` 是 Python 3.8 字节码，最好不要用 Python 3.10+。
- 我在当前 shell 中检测到 `python` 不在 PATH，因此现在这个终端还不能直接训练。
- FAISS GPU 包对 Linux/WSL2 更友好，原生 Windows 上更容易安装失败。

如果是普通老显卡或论文原始环境，可以使用 `environment.yml`。但你的显卡是 RTX 4080，原始 `torch==1.9.0+cu111` 不适合 Ada 架构，建议改用本项目新增的 `environment-rtx4080.yml`：

```powershell
cd E:\pycharm\Projects\RS0521\RS\ITMPRec-master\ITMPRec-master
conda env create -f environment-rtx4080.yml
conda activate itmprec-rtx4080
powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1
```

如果在 WSL2 中执行，路径可以换成挂载路径，例如：

```bash
cd /mnt/e/pycharm/Projects/RS0521/RS/ITMPRec-master/ITMPRec-master
conda env create -f environment-rtx4080.yml
conda activate itmprec-rtx4080
```

## 1. 准备 Python 3.8 字节码导入

运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_pyc_imports.ps1
```

它会把：

```text
src/__pycache__/modules.cpython-38.pyc  -> src/modules.pyc
src/__pycache__/trainers.cpython-38.pyc -> src/trainers.pyc
```

这一步是必须的，因为项目里没有 `modules.py` 和 `trainers.py` 源码。

## 2. 确认数据已经存在

当前项目已经有转换后的数据。以 ML-100K 为例：

```text
data/ml100k/
  ml100k.txt
  data.txt
  target_items.txt

data_graphau/datasets/ml100k/
  train.txt
  val.txt
  test.txt
```

如果将来需要重新转换数据，可运行：

```powershell
python tools/prepare_repro_data.py --datasets ml100k lastfm steam douban_movie
```

## 3. 训练 GraphAU，导出 user/item embeddings

ITMPRec 的训练和模拟评估依赖 GraphAU 生成的 embedding：

```text
data/<dataset>/user_emb-v2-64.pt
data/<dataset>/item_emb-v2-64.pt
```

先跑最小数据集：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_graphau.ps1 -Dataset ml100k -Gpu 0
```

如果没有 GPU，可尝试：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_graphau.ps1 -Dataset ml100k -Gpu -1
```

期望生成：

```text
data/ml100k/user_emb-v2-64.pt
data/ml100k/item_emb-v2-64.pt
```

## 4. 训练 ITMPRec backbone

GraphAU embedding 生成后，运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_itmprec.ps1 -Dataset ml100k -Gpu 0
```

如果没有 GPU：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_itmprec.ps1 -Dataset ml100k -Gpu -1
```

期望生成：

```text
src/output/ICLRec-ml100k-1.pt
```

这个 checkpoint 就是后续评估和 LIR 实验使用的 backbone。

## 5. 评估原始 ITMPRec baseline

运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/eval_ipg_prematch.ps1 -Dataset ml100k -Gpu 0
```

如果没有 GPU：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/eval_ipg_prematch.ps1 -Dataset ml100k -Gpu -1
```

脚本会输出：

- Hit Ratio at 5/10/15/20
- IoI at 5/10/15/20
- IoR at 5/10/15/20

## 6. 其它数据集顺序

建议按这个顺序跑：

```text
ml100k -> lastfm -> douban_movie -> steam
```

原因：

- `ml100k` 最小，适合检查环境。
- `lastfm` 中等，适合验证流程。
- `douban_movie` 和 `steam` 更大，训练和评估更耗时。

数据规模：

```text
ml100k:       943 users, 1682 items, 100000 interactions
lastfm:      1892 users, 12523 items, 71064 interactions
steam:       68403 users, 10050 items, 3285246 interactions
douban_movie:2712 users, 34893 items, 1278401 interactions
```

## 7. 常见问题

### 找不到 python

说明当前终端没有激活 Conda 环境，或 Python 没有加入 PATH。先运行：

```powershell
conda activate itmprec
python --version
```

版本最好是 Python 3.8。

### 找不到 dgl、faiss、torch

说明环境没有按 `environment.yml` 安装成功。先运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1
```

缺什么补什么。若没有 CUDA，可以改装 CPU 版本依赖，但速度会明显下降。

### DGL 导入时报 `C:\Users\admin\.dgl` 权限错误

DGL 第一次导入时会尝试写入用户目录下的 backend 配置。如果没有权限，会报 `PermissionError: [WinError 5]`。脚本里已经设置：

```powershell
$env:DGLBACKEND = "pytorch"
```

如果你手动运行 GraphAU，也可以先在当前终端执行同样的环境变量设置。

### DGL 导入时报缺少 torchdata

DGL 2.x 的 GraphBolt 会导入 `torchdata.datapipes`。如果报：

```text
ModuleNotFoundError: No module named 'torchdata'
```

在激活 `itmprec-rtx4080` 后安装：

```powershell
python -m pip install torchdata==0.7.1
```

### DGL 导入时报缺少 tqdm.auto

DGL 2.x 会通过 `dgl.data.utils` 导入 `tqdm.auto`。原论文环境里的 `tqdm==4.26.0` 太老，如果报：

```text
ModuleNotFoundError: No module named 'tqdm.auto'
```

升级 tqdm：

```powershell
python -m pip install -U "tqdm>=4.64"
```

### fast-pytorch-kmeans 和 Python 3.8 冲突

`fast-pytorch-kmeans` 新版本要求 Python 3.9，但本项目需要 Python 3.8 来加载 `src/modules.pyc` 和 `src/trainers.pyc`。它不是当前 ITMPRec 训练路径的必需依赖，RTX 4080 环境文件中已经移除，保留实际使用到的 `kmeans-pytorch`。

### 训练 ITMPRec 时报缺少 item embedding

先确认 GraphAU 是否生成了：

```text
data/ml100k/item_emb-v2-64.pt
```

没有的话先回到第 3 步训练 GraphAU。

### 评估时报缺少 checkpoint

先确认是否存在：

```text
src/output/ICLRec-ml100k-1.pt
```

没有的话先回到第 4 步训练 ITMPRec。

## 8. LIR 实验入口

如果你后续要在 ITMPRec backbone 上跑 LIR/LLM 增强，使用：

```powershell
cd src
python test_LIR_PreMatch.py --data_dir ../data/ml100k/ --data_name ml100k --num_users 943 --num_items 1682 --model_idx 1 --num_intent_clusters 32 --user_emb_path ../data/ml100k/user_emb-v2-64.pt --item_emb_path ../data/ml100k/item_emb-v2-64.pt --lir_ablation baseline
```

可选消融：

```text
baseline
profiles
cot
feedback
replan
full
```

先用 `baseline` 确认 backbone 结果，再逐步打开其它模块。

## 9. 用 Ollama 生成 LIR 文件

如果要用本地 LLM 生成 profile 和 CoT bridge cache，先安装并启动 Ollama，然后拉取 7B 模型：

```powershell
ollama pull qwen2.5:7b-instruct
```

先小规模生成 profile，确认接口可用：

```powershell
python tools/generate_ml100k_lir_files_ollama.py --mode profiles --model qwen2.5:7b-instruct --max_users 5 --max_items 5
```

确认正常后生成完整 profile：

```powershell
python tools/generate_ml100k_lir_files_ollama.py --mode profiles --model qwen2.5:7b-instruct
```

CoT cache 不建议全量生成，先用小规模：

```powershell
python tools/generate_ml100k_lir_files_ollama.py --mode cot --model qwen2.5:7b-instruct --max_users 50 --max_targets 5 --candidates_per_target 10
```

默认输出文件是：

```text
data/ml100k/lir_profiles_ollama.jsonl
data/ml100k/cot_bridge_cache_ollama.jsonl
```

跑 LIR full 时显式指定这两个文件：

```powershell
cd src
python test_LIR_PreMatch.py --data_dir ../data/ml100k/ --data_name ml100k --num_users 943 --num_items 1682 --model_idx 1 --num_intent_clusters 32 --user_emb_path ../data/ml100k/user_emb-v2-64.pt --item_emb_path ../data/ml100k/item_emb-v2-64.pt --lir_ablation full --lir_profile_path ../data/ml100k/lir_profiles_ollama.jsonl --lir_cot_cache_path ../data/ml100k/cot_bridge_cache_ollama.jsonl
```
