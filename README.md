# Generative AI Litigation Dataset

This repository builds a structured empirical dataset of generative AI litigation using LexisNexis metadata, public court records, and supplementary legal sources. It focuses on identifying and analyzing legal disputes involving foundation models, generative AI systems, training data practices, copyright claims, privacy issues, voice and likeness disputes, and related emerging doctrines.

---

# Project Objective

The goal of this repository is to systematically map the evolving generative AI litigation landscape and produce a research-ready dataset for empirical legal analysis.

Key research questions include:

- Which AI companies are most frequently named in litigation?
- What types of claims are most common?
- Is litigation concentrated around copyright, training data, privacy, voice, likeness, competition, or other issues?
- How do disputes evolve over time?
- Which plaintiff groups are most active?
- How are courts responding procedurally (dismissal, injunction, discovery, settlement, appeal)?
- What legal themes emerge from case texts using NLP methods?

---

# Main Research Focus: AI Copyright Litigation

A dedicated pipeline in this repository focuses on AI copyright and training-data litigation, including disputes involving:

- copyrighted works used in model training
- fair use defenses
- market substitution and licensing harm
- scraping and dataset acquisition
- memorization and output similarity
- DMCA / CMI removal claims
- derivative works theories
- injunctions and damages requests

The pipeline combines proprietary legal metadata with public court sources and performs NLP-based text analysis.

---

# Data Sources

This repository integrates multiple legal data sources:

- LexisNexis / Nexis Uni metadata exports
- CourtListener
- RECAP / PACER-derived public docket data
- Public AI litigation trackers
- Public court filings and judicial opinions
- Supplementary APIs for metadata enrichment

---

# Methodology

The repository includes end-to-end pipelines for:

## Data Engineering

- metadata cleaning
- deduplication
- case normalization
- docket number extraction
- court standardization
- date parsing
- dataset merging

## Legal Entity Extraction

- plaintiff / defendant identification
- AI company detection
- party role classification
- claim-type parsing

## Enrichment Pipelines

- CourtListener opinion search
- RECAP docket matching
- public metadata completion
- opinion text recovery
- multi-source record reconciliation

## NLP / Quantitative Analysis

- doctrine keyword scoring
- fair use four-factor analysis
- topic modeling
- text clustering
- nearest-case similarity
- trend analysis over time
- keyword salience
- index construction
- embeddings-ready text outputs

## Visualization

- litigation trends over time
- defendant frequency
- plaintiff composition
- doctrine prevalence
- topic distributions
- summary tables
- paper-ready figures

---

# Repository Structure

```text
data_raw/        Raw source exports
data_clean/      Cleaned structured datasets
data_nlp/        Enriched NLP-ready outputs
figures/         Charts, tables, and publication figures
scripts/         Core pipeline scripts
nlp_copyright/   Copyright-focused empirical NLP pipeline
```

---

# Core Pipeline

## General Litigation Pipeline

```text
raw metadata
→ cleaning
→ claim tagging
→ entity extraction
→ structured dataset
→ figures
```

## Copyright Litigation Pipeline

```text
07_final_case_dataset.xlsx
→ 01_lexis_manipulation.py
→ 02_courtlistener_query_and_combine.py
→ 03_nlp_analysis.py
→ 04_summary_table.py
```

### Pipeline Description

This specialized pipeline focuses on generative AI copyright litigation and training-data disputes. It transforms a cleaned legal metadata file into an enriched, research-ready empirical dataset with NLP outputs and publication-quality figures.

---

### Step 1 — `01_lexis_manipulation.py`

Input:

```text
07_final_case_dataset.xlsx
```

Main tasks:

- load cleaned LexisNexis litigation dataset
- standardize column names
- normalize case names
- extract docket numbers
- identify likely AI-related defendants
- classify plaintiff types
- detect copyright-related claims
- build structured candidate dataset for enrichment

Output:

```text
data_nlp/lexis_structured_candidates.csv
```

---

### Step 2 — `02_courtlistener_query_and_combine.py`

Input:

```text
data_nlp/lexis_structured_candidates.csv
```

Main tasks:

- search CourtListener by docket number
- fallback search by case name
- retrieve RECAP docket metadata
- retrieve judicial opinions when available
- recover missing filing dates, courts, citations, and text
- merge public court data with LexisNexis metadata
- choose best available source for each field

Output:

```text
data_nlp/enriched_copyright_cases.csv
```

---

### Step 3 — `03_nlp_analysis.py`

Input:

```text
data_nlp/enriched_copyright_cases.csv
```

Main tasks:

- final AI + copyright filtering
- doctrine keyword scoring
- fair use four-factor analysis
- training data conflict scoring
- output infringement risk scoring
- remedy pressure scoring
- case theme classification
- topic modeling (TF-IDF + NMF)
- text clustering
- nearest-case similarity mapping
- trend visualizations
- topic and cluster exports

Outputs:

```text
data_nlp/final_ai_copyright_nlp_output.csv
data_nlp/ai_copyright_case_summary_for_paper.csv
data_nlp/nmf_topic_keywords.csv
data_nlp/text_cluster_keywords.csv
data_nlp/nlp_topics_and_clusters_report.txt
figures/*.png
```

---

### Step 4 — `04_summary_table.py`

Input:

```text
data_nlp/final_ai_copyright_nlp_output.csv
```

Main tasks:

- compute dataset-level descriptive statistics
- generate publication-ready summary table
- create topic distribution tables
- export quantitative metrics for paper use

Outputs:

```text
data_nlp/quantitative_summary_table.csv
figures/10_quantitative_summary_table.png
figures/11_topic_distribution_named_table.png
figures/12_topic_distribution_named_bar.png
```
