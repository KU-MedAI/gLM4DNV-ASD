# Benchmarking genomic language models for disease prediction in autism spectrum disorder

---

## Abstract
<div align="center">
<img width="8738" height="6612" alt="260324_figure1" src="https://github.com/user-attachments/assets/7a015300-14df-4968-8c10-d789248b2f62" />

</div>
<br>
<br>

The genetic basis of complex diseases arises from variation across both coding and non-coding regions of the genome, yet translating individual genomic variation into disease prediction remains a fundamental challenge. Genomic language models (gLMs), which learn sequence representations directly from DNA, have emerged as promising approaches for capturing the regulatory and functional signals encoded in genomic sequence. However, whether gLM-derived variant representations can support sample-level disease prediction remains unclear. In this study, we systematically evaluated seven gLMs using autism spectrum disorder (ASD) as a representative complex disease. Using whole-genome sequencing data from 15,292 samples across four cohorts and their de novo variants, we benchmarked model performance within a unified pipeline. In the zero-shot setting, all models showed predictive performance close to chance level, with area under the receiver operating characteristic curve (AUROC) values of approximately 0.50. Following fine-tuning on variant-level functional datasets, models exhibited increased sensitivity to functionally important variants, with top-attended variants significantly enriched in ASD-relevant biological pathways. However, despite these improvements at the representation level, sample-level ASD prediction performance did not improve significantly, with AUROC values ranging from 0.4936 to 0.5413 across all fine-tuning conditions. These findings demonstrate a fundamental gap between variant-level representational improvement and sample-level predictive utility, reflecting a mismatch between current pretraining strategies and the polygenic architecture of complex diseases. Our results highlight the need for gLMs that incorporate individual genomic variation and multi-omics regulatory context to enable robust disease prediction.

---

## Overview

In this work, we systematically benchmark **seven gLMs** for predicting **ASD** from **de novo variants**. Each model is evaluated under two settings: **zero-shot representation** and **task-specific fine-tuning**.

The evaluated models include:

- DNABERT  
- DNABERT-2  
- Nucleotide Transformer V2  
- Nucleotide Transformer V3  
- HyenaDNA  
- Evo 2  
- PhyloGPN  

This repository includes:

- a unified **benchmarking framework for gLMs**
- a **sample-level ASD prediction pipeline**
- analyses of **fine-tuning effects on gLM embeddings**
- evaluation of the **biological interpretability of model attention**

---

## Repository Structure

```text
project_root/
├── data/                                
│
├── zero-shot/                           
│
├── fine-tuned/                         
│   └── {model}/
│       ├── classification/              
│       ├── regression/                 
│       └── variant-pooling/            
│
├── set_transformer/                  
│
└── analysis/
    ├── fine-tuning-effects/             
    └── biological_interpretability/    
```

---

## Environment Setting
### Required Packages

```python
# PyTorch (CUDA support recommended)
pip install torch torchvision torchaudio

# Transformers / Genomic Language Models
pip install transformers accelerate huggingface-hub

# Data Processing
pip install pandas numpy scipy pyarrow

# Visualization
pip install matplotlib seaborn

# Statistical Analysis
pip install statsmodels scikit-learn

# Progress Display
pip install tqdm

# Parameter-Efficient Fine-Tuning (LoRA)
pip install peft

# Analysis - GSEApy Enrichr
pip install gseapy

# Optional: Experiment Tracking
pip install wandb
```

---

## Data

This study uses two types of datasets: **de novo variant datasets for ASD prediction** and **datasets used for fine-tuning tasks**.

De novo variants were collected from multiple ASD cohorts, including **SSC, SPARK, MSSNG, and a Korean ASD cohort**. Clinical severity annotations (e.g., ADOS and VABS) were used in downstream analyses when available.

Fine-tuning tasks were constructed using publicly available datasets such as **ClinVar missense variants, gnomAD variants, BEND non-coding variants, and non-coding regulatory elements (NCREs)**.

Detailed instructions for preparing the required datasets are provided in: `data/README.md`

---

## Running the Benchmark Pipeline

The main steps to reproduce the benchmark are outlined below.

### 1. Embedding Variant Sequences

#### 1-1. Variant pooling in zero-shot
Extract variant-level embeddings using pretrained gLMs.

```bash
zero-shot/zs_variant_pooling.ipynb
```

#### 1-2. Fine-tuning gLMs

Run task-specific fine-tuning for each model using functional variant datasets.

```bash
bash fine-tuned/{model}/classification/run_ft_{model}_classification.sh
```

#### 1-3. Variant pooling in fine-tuned gLMs

Generate variant-level embeddings from the fine-tuned models by pooling tokens containing variant positions.

```bash
bash fine-tuned/{model}/variant-pooling/run_ft_{model}_variant_pooling.sh
```

### 2. ASD prediction

Aggregate variant embeddings at the sample level and perform ASD prediction using a Set Transformer model.

```python
set_transformer/set_transformer.ipynb
```

---

## Analysis

The analysis scripts reproduce the main results reported in the paper, including:
- Fine-tuning effects on gLM embeddings
- Biological interpretability of ASD prediction

See the `analysis/README.md` directory for details.
