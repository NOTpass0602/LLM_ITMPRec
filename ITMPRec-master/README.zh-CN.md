# ITMPRec：基于意图的目标导向多轮主动推荐

这是论文 **ITMPRec: Intention-based Targeted Multi-round Proactive Recommendation (WWW 2025)** 的实现。

## 实现说明

### 环境要求

- Python >= 3.8
- PyTorch >= 1.9.0
- tqdm == 4.26.0
- faiss-gpu == 1.7.1

本仓库当前更推荐严格使用 Python 3.8，因为 `src/modules.pyc` 和 `src/trainers.pyc` 是 Python 3.8 字节码文件。

### 数据集

需要将四个真实数据集下载并放到 `data` 目录中：

- lastfm: https://grouplens.org/datasets/hetrec-2011/
- ml100k: https://grouplens.org/datasets/movielens/100k/
- steam: https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
- douban_movie: https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information

当前项目中已经准备了本地转换后的数据，见 `data/` 和 `data_graphau/datasets/`。

### 训练 ITMPRec 模型

可以在 `src/output` 下训练 lastfm、ml100k、steam、douban_movie 数据集对应模型。示例：

```bash
cd src
python main.py --data_dir ../data/lastfm/ --data_name lastfm --model_idx 1 --num_intent_clusters 32
```

本项目中的模型训练还需要 GraphAU 生成的 item embedding，建议使用 `scripts/train_itmprec.ps1`，它会自动传入 `--item_emb_path`。

### 评估模型并生成 IoI、IoR

#### 第一步：预训练 GraphAU

需要先训练 GraphAU 来生成初始 user/item embeddings。GraphAU 原始源码地址：

https://github.com/YangLiangwei/GraphAU

本项目已包含并修改了 GraphAU，使其支持 `--data_root` 和 `--export_dir`，可直接导出 ITMPRec 需要的 embedding 文件。

#### 第二步：评估训练后的模型

使用下面命令评估逐轮引导效果：

```bash
python test_IPG_PreMatch.py --data_dir ../data/lastfm/ --data_name lastfm --num_users 945 --num_items 2782 --model_idx 1 --num_intent_clusters 32
```

更推荐使用 `scripts/eval_ipg_prematch.ps1`，它已经内置了每个数据集的用户数、物品数和 embedding 路径。
