# LLM-Enhanced ITMPRec

This repository contains the final cleaned code package for an LLM-enhanced ITMPRec project. It is based on **ITMPRec: Intention-based Targeted Multi-round Proactive Recommendation** and adds LIR/LLM modules, bridge regularization experiments, semantic profile scoring, feedback simulation, and replanning experiments.

## Repository Layout

```text
.
├── ITMPRec-master/
│   ├── src/                         # ITMPRec training/evaluation code
│   │   ├── llm_itmprec/             # LIR/LLM-enhanced recommendation modules
│   │   └── output/                  # Best retained checkpoint and log
│   ├── scripts/                     # PowerShell scripts for training/evaluation
│   ├── tools/                       # Data preparation and embedding utilities
│   ├── results/lir_ablation/        # Experiment records and result summaries
│   ├── data/README.md               # Dataset notes; large datasets are excluded
│   └── *.md                         # Reproduction and experiment documents
├── GraphAU-main/GraphAU-main/       # GraphAU code used to generate embeddings
└── PACKAGE_NOTES.md                 # Packaging and submission notes
```

## Main Contributions

- Adds LIR/LLM modules under `ITMPRec-master/src/llm_itmprec/`.
- Adds semantic profile scoring, CoT bridge planning, complex feedback simulation, and replanning.
- Adds bridge ranking regularization for the ITMPRec backbone.
- Provides reproducible scripts under `ITMPRec-master/scripts/`.
- Keeps experiment logs and summaries under `ITMPRec-master/results/lir_ablation/`.

## Best Retained Result

The retained best backbone checkpoint is:

```text
ITMPRec-master/src/output/ICLRec-ml100k-42.pt
```

The corresponding training log is:

```text
ITMPRec-master/src/output/ICLRec-ml100k-42.txt
```

The main result explanation is in:

```text
ITMPRec-master/results/lir_ablation/bridge_regularization.md
```

Summary:

- Best bridge-regularized backbone: `bridge_reg_w0.001 model_idx=42`
- IPG PreMatch result: `HR@20=0.4861`, `IoI@20=0.2484`, `IoR@20=73.5388`
- Best LIR combination recorded: `model_idx=42 + proactive replan r0.5/t0.5/m0.15`, with `IoR@20=77.5286`

For the full experiment table, see:

```text
ITMPRec-master/results/lir_ablation/summary.md
ITMPRec-master/results/lir_ablation/analysis.md
```

## Environment

Recommended environment files:

```text
ITMPRec-master/environment.yml
ITMPRec-master/environment-rtx4080.yml
```

The original project depends on Python 3.8 bytecode files:

```text
ITMPRec-master/src/modules.pyc
ITMPRec-master/src/trainers.pyc
```

These two files are intentionally retained because the original source files are not available.

## Reproduction

Start from the detailed Chinese guide:

```text
ITMPRec-master/ITMPRec复现指南.zh-CN.md
```

Typical workflow:

```powershell
cd ITMPRec-master
powershell -ExecutionPolicy Bypass -File scripts\check_env.ps1
powershell -ExecutionPolicy Bypass -File scripts\prepare_pyc_imports.ps1
powershell -ExecutionPolicy Bypass -File scripts\train_graphau.ps1
powershell -ExecutionPolicy Bypass -File scripts\train_itmprec.ps1
powershell -ExecutionPolicy Bypass -File scripts\eval_ipg_prematch.ps1
powershell -ExecutionPolicy Bypass -File scripts\eval_lir_prematch.ps1
```

Large datasets, generated embeddings, and most checkpoints are excluded from this repository. Please prepare them following the reproduction guide.

## Notes

This repository is a cleaned final code package. It excludes virtual environments, IDE metadata, caches, raw datasets, large generated embeddings, and unrelated reference projects.
