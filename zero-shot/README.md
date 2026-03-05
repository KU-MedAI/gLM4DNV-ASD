# Zero-shot

## Overview

This repository implements a variant pooling strategy to derive variant-level representations from sequence embeddings produced by genomic language models. Token embeddings corresponding to the mutated nucleotide positions are extracted from the model outputs. When a variant affects multiple positions (e.g., frameshift mutations), the embeddings at those loci are aggregated using max pooling to generate a single variant-level feature vector. This pooled embedding serves as the final representation for each variant.


## Model List

- **DNABERT**
- **DNABERT-2**
- **Nucleotide Transformer V2**
- **Nucleotide Transformer V3**
- **HyenaDNA**
- **Evo 2**
- **PhyloGPN**


## Usage

Run this notebook: `zs_variant_pooling.ipynb`
