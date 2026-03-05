# Fine-Tuning

---

## Overview
This directory contains the code for fine-tuning genomic language models (gLMs) used in the benchmarking pipeline.

It includes:
- task-specific fine-tuning scripts for multiple gLMs
- scripts for **variant pooling** to generate variant-level embeddings from fine-tuned models

Fine-tuning uses task-specific training datasets, while variant pooling is applied to de novo variant datasets used for downstream ASD prediction.

## Directory Structure
```text
fine-tuned/
├── README.md
└── {model}/
     ├── classification
     │    ├── ft_{model}_classification.py
     │    └── run_ft_{model}_classification.sh
     │
     ├── regression
     │    ├── ft_{model}_regression.py
     │    └── run_ft_{model}_regression.sh
     │
     └── variant-pooling
          ├── ft_{model}_variant_pooling.py
          └── run_ft_{model}_variant_pooling.sh
```

## Tasks

Two types of fine-tuning tasks are implemented:

- **Classification:** binary classification of variant effects for **ClinVar, gnomAD, and BEND** tasks
- **Regression:** prediction of continuous regulatory activity scores for the **NCRE** task

Each task is implemented separately for each gLM.

## Usage

Example (DNABERT):

```bash
bash DNABERT/classification/run_ft_dnabert_classification.sh
```

After fine-tuning, variant representations can be generated using the variant pooling scripts:

```bash
bash DNABERT/variant-pooling/run_ft_dnabert_variant_pooling.sh
```
