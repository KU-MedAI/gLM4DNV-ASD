# Data

---

This study uses two types of datasets: **de novo variant datasets for ASD prediction** and **datasets used for fine-tuning tasks**.

## 1. De novo variant datasets 

De novo variants were collected from multiple ASD cohorts:

- SSC  
- SPARK  
- MSSNG  
- Korean ASD cohort

The variant dataset links each sample (vcf_iid) to its specific genetic alteration (locus, variant, alleles) and target gene (gene_symbol). It includes functional impact predictors and pathogenicity scores (e.g., most_severe_consequence, CADD_phred, REVEL, AlphaMissense, pLI). Crucially for sequence-based modeling, the dataset provides extended reference and variant DNA sequences across multiple window sizes along with their precise mutation indices.

In addition to genomic variants, clinical severity annotations were obtained from the same cohorts when available to serve as targets for downstream analyses. These clinical annotations were mapped to the corresponding genomic data using the unique sample identifier (vcf_iid). To construct the target variables for evaluating the clinical impact of these variants, we specifically focused on key standardized behavioral and cognitive assessment scores: the Autism Diagnostic Observation Schedule Total Score (ADOS_Total) and the Vineland Adaptive Behavior Scales (VABS).

## 2. Fine-tuning task datasets -> 어떻게 생겼는지 설명 추가
-> 우리 실제 이름으로 변경 필요
<br> 

Fine-tuning tasks were constructed using publicly available datasets:

- ClinVar  
- gnomAD  
- BEND benchmark  
- BRAIN-MAGNET model (NCRE activity dataset)

All datasets were processed to ensure consistent input construction, and sequence inputs were generated from the **GRCh38 reference genome**.


## Data Access
- De novo variant data
  - Due to data access restrictions, raw datasets cannot be distributed in this repository.
- Fine-tuning task data
  - 출처 각각 기입하기
