# Benchmarking genomic language models for variant representation in autism spectrum disorder prediction

---

## Abstract
<div align="center">
<img width="2009" height="1520" alt="260122 figure1 overview   Model architecture" src="https://github.com/user-attachments/assets/6bfbe945-7285-4fcc-b36a-9190c0eec72b" />

</div>
<br>
<br>

**Background:** Genomic language models (gLMs) have demonstrated strong performance in sequence-level tasks such as variant pathogenicity classification and regulatory activity prediction. However, their utility for disease prediction at the sample level remains largely unexplored. Systematic benchmarking of diverse gLM architectures within a unified disease prediction framework is still lacking, leaving it unclear which representation strategies are most informative for clinical application and whether task-specific fine-tuning can bridge the gap between sequence-level annotation and sample-level disease prediction. Here, we address this gap using autism spectrum disorder (ASD) as a representative complex disease model. 

**Results:** To assess the utility of gLM embeddings in complex disease prediction, we evaluated their performance in classifying ASD risk. Our findings indicate that task-specific fine-tuning did not yield a statistically significant improvement in overall predictive accuracy. Nonetheless, a detailed analysis of the embeddings and attention scores demonstrated that the fine-tuning process elicited significant structural modifications within the embedding space and enabled a strategic redistribution of attention scores. This phenomenon was notably observed in non-coding and regulatory elements relevant to disease pathogenesis.

**Conclusions:** Although fine-tuning was applied to integrate functional variant information, this approach did not directly translate into improved predictive performance, highlighting a fundamental gap between variant-level representation learning and sample-level disease prediction in current gLM frameworks. Nevertheless, these representational shifts demonstrate that fine-tuning can reshape gLM embedding geometry around variant positions, even when this does not yet translate to improved sample-level classification. Furthermore, by systematically benchmarking diverse gLM architectures within a unified disease prediction pipeline, this study establishes a critical baseline and identifies key gaps—particularly in variant-aware pretraining and disease-specific representation learning—that must be addressed to advance gLMs toward clinical utility.

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
