# LLM 增强的 ITMPRec 项目

本仓库是 LLM 增强版 ITMPRec 的最终整理代码包。项目基于 **ITMPRec: Intention-based Targeted Multi-round Proactive Recommendation**，在原始主动推荐框架上加入了 LIR/LLM 相关模块、语义画像评分、CoT 桥接规划、复杂反馈模拟、replan 纠偏机制，以及 bridge ranking regularization 实验。

## 目录结构

```text
.
├── ITMPRec-master/
│   ├── src/                         # ITMPRec 训练与评估代码
│   │   ├── llm_itmprec/             # LIR/LLM 增强相关模块
│   │   └── output/                  # 保留的最佳 checkpoint 与日志
│   ├── scripts/                     # 训练、评估、环境检查脚本
│   ├── tools/                       # 数据准备、embedding 生成与融合工具
│   ├── results/lir_ablation/        # 实验记录、消融结果与分析
│   ├── data/README.md               # 数据目录说明，不包含大体积数据
│   └── *.md                         # 复现说明与实验文档
├── GraphAU-main/GraphAU-main/       # GraphAU 代码，用于生成 embedding
└── PACKAGE_NOTES.md                 # 最终提交包整理说明
```

## 主要工作

- 在 `ITMPRec-master/src/llm_itmprec/` 下新增 LIR/LLM 增强模块。
- 实现语义 profile scoring、CoT bridge planning、复杂反馈模拟和 replan 机制。
- 在 ITMPRec backbone 训练中加入 bridge ranking regularization。
- 在 `ITMPRec-master/scripts/` 中整理训练、评估和环境检查脚本，便于复现。
- 在 `ITMPRec-master/results/lir_ablation/` 中保留实验日志、汇总表和分析文档。

## 最佳结果文件

当前保留的最佳 backbone checkpoint：

```text
ITMPRec-master/src/output/ICLRec-ml100k-42.pt
```

对应训练日志：

```text
ITMPRec-master/src/output/ICLRec-ml100k-42.txt
```

主要结果说明：

```text
ITMPRec-master/results/lir_ablation/bridge_regularization.md
```

结果摘要：

- 最佳 bridge regularization backbone：`bridge_reg_w0.001 model_idx=42`
- IPG PreMatch 指标：`HR@20=0.4861`，`IoI@20=0.2484`，`IoR@20=73.5388`
- 已记录的最佳 LIR 组合：`model_idx=42 + proactive replan r0.5/t0.5/m0.15`，`IoR@20=77.5286`

完整实验表格与分析见：

```text
ITMPRec-master/results/lir_ablation/summary.md
ITMPRec-master/results/lir_ablation/analysis.md
```

## 环境说明

推荐使用以下环境配置文件：

```text
ITMPRec-master/environment.yml
ITMPRec-master/environment-rtx4080.yml
```

原始 ITMPRec 项目依赖两个 Python 3.8 字节码文件：

```text
ITMPRec-master/src/modules.pyc
ITMPRec-master/src/trainers.pyc
```

由于原项目没有提供对应源码，这两个 `.pyc` 文件被保留在最终提交包中。

## 复现方式

建议优先阅读详细复现指南：

```text
ITMPRec-master/ITMPRec复现指南.zh-CN.md
```

典型运行流程如下：

```powershell
cd ITMPRec-master
powershell -ExecutionPolicy Bypass -File scripts\check_env.ps1
powershell -ExecutionPolicy Bypass -File scripts\prepare_pyc_imports.ps1
powershell -ExecutionPolicy Bypass -File scripts\train_graphau.ps1
powershell -ExecutionPolicy Bypass -File scripts\train_itmprec.ps1
powershell -ExecutionPolicy Bypass -File scripts\eval_ipg_prematch.ps1
powershell -ExecutionPolicy Bypass -File scripts\eval_lir_prematch.ps1
```

本仓库没有包含大体积原始数据集、生成的 embedding 文件和大部分 checkpoint。完整运行时需要按照复现指南重新准备数据或生成中间文件。

## 文档说明

- `README.md`：英文仓库首页说明。
- `README.zh-CN.md`：中文仓库首页说明。
- `PACKAGE_NOTES.md`：最终代码包保留与排除内容说明。
- `ITMPRec-master/README.zh-CN.md`：原始 ITMPRec 项目的中文说明。
- `ITMPRec-master/REPRO_PREP.zh-CN.md`：复现准备说明。
- `ITMPRec-master/EXPERIMENT_LIR_ITMPREC.zh-CN.md`：LIR-ITMPRec 实验说明。
- `ITMPRec-master/ITMPRec复现指南.zh-CN.md`：最完整的本地复现流程。

## 清理说明

本仓库是最终整理后的代码包，已排除以下内容：

- 虚拟环境和 IDE 配置，如 `.venv`、`.idea`。
- Python 缓存，如 `__pycache__`。
- 原始数据集、大体积 embedding 和中间数据文件。
- 非最佳 checkpoint、训练临时输出和无关参考项目。
