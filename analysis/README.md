# Analysis

---

## Overview
This directory contains analyses examining how fine-tuning affects genomic language models (gLMs) for Autism Spectrum Disorder (ASD) prediction through embedding representation analysis and attention-based interpretation at the variant, gene, and pathway levels.

## Directory Structure
```text
analysis/
├── README.md
├── fine-tuning-effects/
│   ├── embedding_vector_analysis.ipynb
│   ├── attn_score_analysis_with_precion_recall.ipynb
│   └── attn_score_analysis_with_ASDDD_gene.ipynb
│
└── biological-interpretability/
    ├── attn_score_based_enrichment.ipynb
    └── attn_scroe_with_severity_enrichment.ipynb
```

---

## Analysis of fine-tuning effects
These analyses assess embedding representations and attention scores at the variant and gene levels.

**Data used:**
- **Variant-to-gene mapping data** generated using **Ensembl VEP**
- **CADD Phred scores** used to define functionally deleterious variants
- **Reference and variant embeddings** from each model in both **zero-shot** and **fine-tuned** settings
- **Attention scores** from each **model–task combination**

### 1. Representational shifts in latent embedding space
- Compute **cosine similarity** between **reference and variant embeddings**
- Stratify variants by **VEP impact** and **CADD scores** to evaluate enrichment of **functionally important variants**

### 2. Functional variant enrichment in attention scores
- Normalize **variant-level attention scores** using **CLR transformation**.
- Select the **top 20% attention-ranked variants**.
- Evaluate enrichment of **functional variants** using **precision/recall** and **Fisher’s exact test**

### 3. Alignment between prediction confidence and disease gene prioritization
- Normalize **attention scores** across **gLMs** using **CLR transformation**
- Compare **top and bottom 20% samples** based on **prediction probabilities**
- Quantify the association between **model confidence** and **risk gene attention** using **Cliff’s delta**


---

## Biological interpretability of ASD prediction

These analyses assess whether variants prioritized by model attention converge on biological pathways associated with ASD. 

**Data used:**
- **Variant-to-gene mapping data** generated using Ensembl VEP
- Attention scores derived from each **model–task combination**
- Clinical severity annotations (**ADOS** and **VABS**)

### 1. Attention-based variant prioritization enrichment analysis
- Normalize attention scores within each sample using **CLR transformation**
- Select **top and bottom 10% attention-ranked variants**
- Map variants to genes and perform **GO Biological Process enrichment**

### 2. Enrichment analysis of ASD subgroups based on severity annotation
- Apply the same enrichment framework to **clinically defined ASD subgroups**
- Define severe groups using **ADOS and VABS severity annotations**
- Compare pathway enrichment patterns between **top/bottom attention-ranked variants**
