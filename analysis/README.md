# Analysis

---

## Overview
Autism Spectrum Disorder (ASD) 넣어주기 (아래에서 약어 사용함)

## Directory Structure
```text
analysis/
├── README.md
├── fine-tuning-effects/
│   ├── embedding_vector_analysis.py
│   ├── attention_score_analysis_1.py
│   └── attention_score_analysis_2.py    # Total 3 files (incl. the 2 above)
│
└── biological-interpretability/
    ├── attn_score_based_enrichment.ipynb
    └── attn_scroe_with_severity_enrichment.ipynb
```

---

## Analysis of fine-tuning effects
### Representational shifts in latent embedding space
### Functional variant enrichment in attention scores
### Alignment between prediction confidence and disease gene prioritization

---

## Biological interpretability of ASD prediction
This analysis assesses whether the mutations prioritized by the model through attention scores converge on biological pathways associated with ASD

### Attention-based variant prioritization enrichment analysis
- Normalize attention scores within each sample using CLR transformation.
- Select top and bottom 10% attention-ranked variants.
- Map variants to genes and perform GO Biological Process enrichment analysis.

### Enrichment analysis of ASD subgroups based on severity annotation
- Apply the same enrichment framework to clinically defined ASD subgroups.
- Define severe groups using ADOS and VABS severity annotations.
- Compare pathway enrichment patterns of top/bottom attention-ranked variants within each subgroup.
