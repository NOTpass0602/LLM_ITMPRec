# 最终代码包整理说明

本目录从原项目 `ITMPRec-master` 中整理得到，用于最终提交代码 zip。

## 保留内容

- `ITMPRec-master/src/`: ITMPRec 训练、评估与 LIR/LLM 增强相关代码。
- `ITMPRec-master/src/llm_itmprec/`: 本项目新增或整理的 LIR/LLM 相关模块。
- `ITMPRec-master/scripts/`: 运行、评估、环境检查脚本。
- `ITMPRec-master/tools/`: 数据准备、embedding 生成、融合等工具脚本。
- `ITMPRec-master/*.md`、`environment*.yml`: 说明文档和环境配置。
- `ITMPRec-master/src/modules.pyc`、`ITMPRec-master/src/trainers.pyc`: 项目运行依赖的 Python 3.8 字节码文件，原项目没有对应源码，因此保留。
- `ITMPRec-master/src/output/ICLRec-ml100k-42.pt`、`ICLRec-ml100k-42.txt`: 当前 bridge regularization 最佳 backbone checkpoint 及训练日志。
- `ITMPRec-master/results/lir_ablation/`: LIR/LLM 消融、backbone fusion、bridge regularization 相关实验记录与结果汇总。
- `GraphAU-main/GraphAU-main/`: GraphAU 代码部分，供生成 ITMPRec 所需 embedding。

## 已排除内容

- 虚拟环境与 IDE 配置：`.venv`、`.idea`。
- Python 缓存：`__pycache__`、普通缓存 `.pyc`。
- 数据集与大文件：`data/` 下的数据文件、`data_graphau/`、GraphAU `datasets/`。
- 大部分实验产物：其它 `src/output/` checkpoint、`graphau_runs/`、非最佳模型权重 `.pt`。
- 参考论文、调研文档、Word 临时文件。
- 其他参考项目：`APG4Sim-main`、`CoT-Rec-main`、`GRSU-main`、`RLMRec-main`。

## 最佳结果口径

- Backbone bridge regularization 最佳：`model_idx=42`，详见 `results/lir_ablation/bridge_regularization.md`。
- LIR/LLM 实验汇总：详见 `results/lir_ablation/summary.md` 和 `analysis.md`。

## 复现提示

如需完整运行，请按 `ITMPRec-master/ITMPRec复现指南.zh-CN.md` 准备环境和数据。数据与训练产物没有放入本提交包，建议在复现环境中重新下载或生成。
