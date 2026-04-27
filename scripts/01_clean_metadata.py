import pandas as pd
import re
import os

# Load metadata

RAW_PATH = "data_raw/lexis_genai_copyright_metadata.xlsx"
OUT_DIR = "data_clean"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_excel(RAW_PATH)

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("\n", "_")
)

# Clean text cells
for col in df.columns:
    df[col] = (
        df[col]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

print(df.shape)
print(df.columns)

df.to_excel(f"{OUT_DIR}/01_metadata_cleaned.xlsx", index=False)

# Merge Text Fields

text_cols = [
    "title", "name", "court", "citation", "core_terms",
    "history", "disposition", "opinion", "overview",
    "headnotes", "attorney", "judges", "number"
]

text_cols = [c for c in text_cols if c in df.columns]

df["combined_text"] = (
    df[text_cols]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.replace(r"\s+", " ", regex=True)
)

df.to_excel(f"{OUT_DIR}/02_metadata_with_text.xlsx", index=False)

# Identify company

company_patterns = {
    "OpenAI": r"\bOpenAI\b|ChatGPT|GPT-4|GPT-3\.5|DALL·E|DALL-E|\bGPT\b",
    "Microsoft": r"\bMicrosoft\b|GitHub|Copilot",
    "Google": r"\bGoogle\b|Alphabet|YouTube|Gemini|Bard|DeepMind",
    "Meta": r"\bMeta\b|Meta Platforms|Facebook|Instagram|WhatsApp|LLaMA|Llama",
    "Anthropic": r"\bAnthropic\b|Claude",
    "Amazon": r"\bAmazon\b|AWS|Bedrock",
    "Apple": r"\bApple\b",
    "Tesla": r"\bTesla\b",
    "xAI": r"\bxAI\b|Grok",
    "IBM": r"\bIBM\b|Watson",
    "Oracle": r"\bOracle\b",
    "Adobe": r"\bAdobe\b|Firefly",
    "Canva": r"\bCanva\b",
    "ByteDance": r"\bByteDance\b|TikTok",
    "Snap": r"\bSnap\b|Snapchat",
    "Salesforce": r"\bSalesforce\b|Einstein",
    "Databricks": r"\bDatabricks\b|MosaicML|Mosaic ML",
    "Nvidia": r"\bNvidia\b",
    "Perplexity": r"\bPerplexity\b",
    "Cohere": r"\bCohere\b",
    "Mistral": r"\bMistral\b",
    "Hugging Face": r"Hugging Face",
    "Character.AI": r"Character\.AI|Character AI",
    "Midjourney": r"\bMidjourney\b",
    "Stability AI": r"Stability AI|Stable Diffusion",
    "Runway": r"\bRunway\b",
    "Pika": r"\bPika\b",
    "Suno": r"\bSuno\b",
    "Udio": r"\bUdio\b",
    "ElevenLabs": r"ElevenLabs",
    "Lovo": r"\bLovo\b",
    "Synthesia": r"\bSynthesia\b",
    "HeyGen": r"\bHeyGen\b",
    "Jasper": r"\bJasper\b",
    "Grammarly": r"\bGrammarly\b",
    "Bloomberg": r"\bBloomberg\b",
    "Ross Intelligence": r"Ross Intelligence|ROSS",
}

alias_map = {
    "Alphabet": "Google",
    "Google LLC": "Google",
    "DeepMind": "Google",
    "Meta Platforms": "Meta",
    "Facebook": "Meta",
    "Instagram": "Meta",
    "GitHub": "Microsoft",
    "AWS": "Amazon",
    "TikTok": "ByteDance",
    "Claude": "Anthropic",
    "Gemini": "Google",
    "Bard": "Google",
    "Copilot": "Microsoft",
    "ChatGPT": "OpenAI",
}

def discover_possible_companies(text):
    text = str(text)

    pattern = r"""
        \b(
            [A-Z][A-Za-z0-9&\.\-]+
            (?:\s+[A-Z][A-Za-z0-9&\.\-]+){0,3}
            \s+
            (?:Inc|LLC|Corp|Corporation|Ltd|Labs|AI|Technologies|Systems)
        )\b
    """

    matches = re.findall(pattern, text, flags=re.VERBOSE)

    return sorted(list(set(m.strip() for m in matches if len(m.strip()) > 2)))

def detect_companies(text):
    text = str(text)
    hits = set()

    # Known companies
    for company, pattern in company_patterns.items():
        if re.search(pattern, text, flags=re.I):
            hits.add(company)

    # Possible unknown companies
    for item in discover_possible_companies(text):
        hits.add(alias_map.get(item, item))

    # Normalize aliases
    hits = {alias_map.get(h, h) for h in hits}

    return sorted(list(hits))

df["target_companies"] = df["combined_text"].apply(detect_companies)

df["has_target_company"] = df["target_companies"].apply(lambda x: len(x) > 0)

# Optional: save company frequency table
company_rows = []

for _, row in df.iterrows():
    for company in row["target_companies"]:
        company_rows.append(company)

company_counts = pd.Series(company_rows).value_counts()

company_counts.to_csv(f"{OUT_DIR}/company_counts_raw.csv")

print(company_counts.head(30))

# Identify claim type

def detect_labels(text, patterns):
    hits = []
    text = str(text)

    for label, pattern in patterns.items():
        if re.search(pattern, text, flags=re.I):
            hits.append(label)

    return hits

claim_patterns = {
    "Copyright": r"copyright|infringement|fair use|DMCA",
    "Training Data": r"training data|scraping|scraped|books|authors|corpus|dataset",
    "Privacy": r"privacy|personal data|biometric|data collection",
    "Publicity/Likeness": r"likeness|voice|image|deepfake|clone|right of publicity",
    "Defamation": r"defamation|hallucination|false statement|libel|slander",
    "Contract": r"contract|terms of service|license|breach",
}

df["claim_types"] = df["combined_text"].apply(
    lambda x: detect_labels(x, claim_patterns)
)

df["has_claim_signal"] = df["claim_types"].apply(lambda x: len(x) > 0)

# Relevence score

def relevance_score(row):
    score = 0

    if row["has_target_company"]:
        score += 3

    if "Copyright" in row["claim_types"]:
        score += 3

    if "Training Data" in row["claim_types"]:
        score += 2

    if re.search(r"generative AI|ChatGPT|large language model|LLM|Claude|Gemini", row["combined_text"], re.I):
        score += 2

    return score

df["relevance_score"] = df.apply(relevance_score, axis=1)

df_candidates = df[df["relevance_score"] >= 3].copy()

df_candidates.to_excel(f"{OUT_DIR}/03_candidate_records.xlsx", index=False)

print("Candidate records:", df_candidates.shape)

# Deduplicated to case-level dataset

def normalize_case_name(x):
    x = str(x).lower()
    x = re.sub(r"\b20\d{2}\b", "", x)
    x = re.sub(r"u\.s\. dist\. lexis \d+", "", x)
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()

if "name" in df_candidates.columns:
    df_candidates["case_name_norm"] = df_candidates["name"].apply(normalize_case_name)
else:
    df_candidates["case_name_norm"] = df_candidates["title"].apply(normalize_case_name)

dedupe_cols = ["case_name_norm"]

if "number" in df_candidates.columns:
    dedupe_cols.append("number")

if "court" in df_candidates.columns:
    dedupe_cols.append("court")

df_cases = (
    df_candidates
    .sort_values("relevance_score", ascending=False)
    .drop_duplicates(subset=dedupe_cols)
    .copy()
)

df_cases.to_excel(f"{OUT_DIR}/04_unique_cases.xlsx", index=False)

print("Unique cases:", df_cases.shape)