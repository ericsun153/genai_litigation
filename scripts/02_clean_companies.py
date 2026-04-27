import pandas as pd
import ast
import re
import os

IN_PATH = "data_clean/04_unique_cases.xlsx"
OUT_PATH = "data_clean/05_unique_cases_company_cleaned.xlsx"

os.makedirs("data_clean", exist_ok=True)

df = pd.read_excel(IN_PATH)

def safe_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return []

df["target_companies"] = df["target_companies"].apply(safe_list)

# Standardize noisy company names
STANDARD_MAP = {
    "For OpenAI Inc": "OpenAI",
    "For OpenAI GP LLC": "OpenAI",
    "For OpenAI LLC": "OpenAI",
    "For OpenAI Global LLC": "OpenAI",
    "For OpenAI OpCo LLC": "OpenAI",
    "OpenAI Global LLC": "OpenAI",
    "OpenAI OpCo LLC": "OpenAI",
    "OAI Corporation": "OpenAI",
    "For OAI Corporation LLC": "OpenAI",
    "Open AI": "OpenAI",

    "For Microsoft Corporation": "Microsoft",
    "Microsoft Corp": "Microsoft",
    "Microsoft Corporation": "Microsoft",

    "Apple Inc": "Apple",
    "Stability AI Ltd": "Stability AI",
}

# Obvious false positives
DROP_COMPANIES = {
    "Professional Corporation",
    "Bell Atl. Corp",
    "Spark Innovations Corp",
    "Sony Corp",
}

# Optional: for this copyright-focused AI dataset,
# Amazon is often a false hit unless the row also has AWS/Bedrock or defendant context.
SUSPICIOUS_COMPANIES = {"Amazon"}

def clean_company_list(companies, text):
    cleaned = []

    text = str(text)

    for c in companies:
        c = STANDARD_MAP.get(c, c)

        if c in DROP_COMPANIES:
            continue

        # Keep Amazon only if clearly AI-related
        if c == "Amazon":
            if not re.search(r"\bAWS\b|Bedrock|Amazon Web Services|Amazon AI", text, re.I):
                continue

        cleaned.append(c)

    return sorted(list(set(cleaned)))

df["target_companies_clean"] = df.apply(
    lambda row: clean_company_list(row["target_companies"], row.get("combined_text", "")),
    axis=1
)

df["has_clean_company"] = df["target_companies_clean"].apply(lambda x: len(x) > 0)

# Keep only rows with a clean AI company OR strong copyright/training data signal
def strong_signal(row):
    claims = safe_list(row.get("claim_types", []))
    text = str(row.get("combined_text", ""))

    if row["has_clean_company"]:
        return True

    if "Copyright" in claims and re.search(r"generative AI|ChatGPT|LLM|large language model|training data|scraping", text, re.I):
        return True

    return False

df["keep_after_company_cleaning"] = df.apply(strong_signal, axis=1)

df_clean = df[df["keep_after_company_cleaning"]].copy()

df_clean.to_excel(OUT_PATH, index=False)

# Save company counts after cleaning
company_rows = []
for _, row in df_clean.iterrows():
    for c in row["target_companies_clean"]:
        company_rows.append(c)

company_counts = pd.Series(company_rows).value_counts()
company_counts.to_csv("data_clean/company_counts_cleaned.csv")

print("Before cleaning:", df.shape)
print("After cleaning:", df_clean.shape)
print(company_counts.head(30))
print("Saved:", OUT_PATH)