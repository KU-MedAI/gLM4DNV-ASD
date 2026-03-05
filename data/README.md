# Data

---

This study uses two types of datasets: **de novo variant datasets for ASD prediction** and **datasets used for fine-tuning tasks**.

## 1. De novo variant datasets

De novo variants were collected from multiple ASD cohorts:

- SSC  
- SPARK  
- MSSNG  
- Korean ASD cohort

The variant dataset links each sample (**vcf_iid**) to its corresponding genetic variants and annotations.

**Key information included:**

- Variant information: locus, alleles, and target gene (**gene_symbol**)  
- Functional annotations and pathogenicity scores (e.g., **most_severe_consequence**, **CADD_phred**)  
- Reference and variant DNA sequences extracted at multiple window sizes  
- Mutation indices indicating the exact position of the variant within each sequence window  

<br>
Clinical severity annotations were also collected when available and linked to genomic data using the **vcf_iid** identifier.

**Clinical metrics used for downstream analyses:**

- **ADOS-CSS** (Autism Diagnostic Observation Schedule)  
- **VABS** (Vineland Adaptive Behavior Scales)

---

## 2. Fine-tuning task datasets
<br>

Fine-tuning tasks were constructed using publicly available datasets:

- **ClinVar missense variants**  
- **gnomAD variants**  
- **BEND non-coding variants**  
- **Non-coding regulatory elements (NCRE)**  

To ensure compatibility across tasks, all datasets were standardized into a **sequence–label format**.

**Data preprocessing:**

- DNA sequences were generated from the **GRCh38 reference genome**
- Fixed-length sequence windows were extracted around variants or regulatory regions
- Each sequence was paired with a task-specific label

**Task labels:**

- **ClinVar:** binary pathogenicity labels  
- **gnomAD:** variant frequency–based signals  
- **BEND / NCRE:** functional or regulatory activity scores  

This unified representation enables consistent fine-tuning across diverse genomic prediction tasks.

---

## Data Access
- De novo variant and severity annotations data
  - Access to the raw datasets must be obtained directly from the original data providers.
- Fine-tuning task data
Fine-tuning tasks were constructed using publicly available datasets:

  - **ClinVar missense variants** ([Landrum et al., 2018](https://pubmed.ncbi.nlm.nih.gov/31777943/))
  - **gnomAD variants** ([Karczewski et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32461654/))
  - **BEND benchmark** ([Marin et al., 2023](https://arxiv.org/abs/2311.12570))
  - **NCRE activity dataset** ([Deng et al., 2026](https://pubmed.ncbi.nlm.nih.gov/41265437/))
