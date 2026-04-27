# Generative AI Litigation Dataset

This repo builds a structured dataset of generative AI litigation cases using LexisNexis metadata, public court records, and supplementary legal sources.

## Goal

The repo aims to identify key patterns in the emerging generative AI litigation landscape, including:

- which companies are most frequently involved  
- what legal claims are most common  
- how disputes evolve over time  
- who is suing whom  
- whether litigation is concentrated around copyright, training data, privacy, voice, likeness, or other issues  

The workflow also enriches raw legal metadata with external APIs and applies NLP techniques for deeper empirical analysis.

## Data Sources

- LexisNexis / Nexis Uni metadata exports  
- CourtListener / RECAP  
- Public litigation trackers  
- Public dockets and court filings  
- Supplementary APIs for case enrichment  

## Methods

The repo includes pipelines for:

- metadata cleaning and deduplication  
- company / party extraction  
- claim classification  
- court and jurisdiction analysis  
- API-based data completion  
- text extraction from filings  
- NLP analysis (topic modeling, clustering, keyword trends, embeddings, etc.)  
- visualization and empirical figures  

## Folder Structure

```text
data_raw/        Raw metadata exports (from 2022.11.30~2026.4)
data_clean/      Final cleaned datasets
figures/         Generated charts and visualizations
scripts/         Python pipeline scripts