# Set Transformer

## Overview
This module implements a **Set Transformer–based model** for predicting Autism Spectrum Disorder (ASD) from genetic variant embeddings.

The model treats variants within each individual as an **unordered set with variable size**, and aggregates them into a **sample-level representation** for classification.

**Inputs**
- Variant-level embeddings produced by genomic language models (**zero-shot or fine-tuned**)
- Each sample contains a **variable number of variants**

**Model**
- A **Set Transformer architecture** is used to aggregate variant embeddings
- Generates a single **sample-level representation**

**Output**
- Binary classification: **ASD vs. Control**


## Usage

Run the notebook: `set_transformer.ipynb`
