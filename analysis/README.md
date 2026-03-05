# Analysis

---

## Overview
This directory contains analyses designed to investigate how genomic language model (gLM) representations change after fine-tuning in the context of Autism Spectrum Disorder (ASD) prediction.

## Directory Structure
```text
analysis/
├── README.md
├── fine-tuning-effects/
│   ├── embedding_vector_analysis.ipynb
│   └── attention_score_analysis.ipynb
│
└── biological-interpretability/
    ├── attn_score_based_enrichment.ipynb
    └── attn_scroe_with_severity_enrichment.ipynb
```

---

## Analysis of fine-tuning effects
### Representational shifts in latent embedding space
This analysis evaluates how fine-tuning alters the latent representations of genomic variants and whether these changes align with known functional annotations.

- Compute cosine similarity between reference and variant embeddings for Zero-shot and Fine-tuned models.
- Stratify variants by VEP impact and compare the magnitude of representation shifts.
- Stratify variants by CADD and evaluate whether shifts are enriched in predicted pathogenic variants.

### Functional variant enrichment in attention scores
### Alignment between prediction confidence and disease gene prioritization

---

## Biological interpretability of ASD prediction

This analysis assesses whether variants prioritized by model attention converge on biological pathways associated with ASD. 

**Data used:**
- **Variant-to-gene mapping data** generated using Ensembl VEP
- Attention scores derived from each **model–task combination**
- Clinical severity annotations (**ADOS CSS** and **VABS**)

### Attention-based variant prioritization enrichment analysis
- Normalize attention scores within each sample using **CLR transformation**
- Select **top and bottom 10% attention-ranked variants**
- Map variants to genes and perform **GO Biological Process enrichment**

### Enrichment analysis of ASD subgroups based on severity annotation
- Apply the same enrichment framework to **clinically defined ASD subgroups**
- Define severe groups using **ADOS and VABS severity annotations**
- Compare pathway enrichment patterns between **top/bottom attention-ranked variants**
