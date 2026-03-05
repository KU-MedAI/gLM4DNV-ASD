# Fine-Tuning Code

---

## Overview
This directory contains the code and resources for **{short description of the module}** in the benchmarking pipeline.

## Directory Structure
```text
fine-tuned/
├── README.md
├── ft-classification/
│   └── {model directories}
├── ft-regression/
│   └── {model directories}
└── variant-pooling/
    └── {model directories}
```

## Tasks

Two types of fine-tuning tasks are implemented:

- **Classification**
- **Regression**

Each task is implemented separately for each genomic language model.

## Usage

Example:

```bash
bash ft-classification/dnabert/run.sh
```

After fine-tuning, variant pooling is performed using scripts in: 

```bash
bash variant-pooling/dnabert/pooling_best.sh
```
