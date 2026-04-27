import pandas as pd
import ast
import re
import os

IN_PATH = "data_clean/05_unique_cases_company_cleaned.xlsx"
OUT_PATH = "data_clean/06_auto_validation_sheet.xlsx"

os.makedirs("data_clean", exist_ok=True)

df = pd.read_excel(IN_PATH)

def safe_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return []

df["target_companies_clean"] = df["target_companies_clean"].apply(safe_list)
df["claim_types"] = df["claim_types"].apply(safe_list)

CORE_AI_COMPANIES = {
    "OpenAI", "Anthropic", "Google", "Meta", "Microsoft",
    "Stability AI", "Midjourney", "Perplexity", "Cohere",
    "Nvidia", "xAI", "Suno", "Udio", "ElevenLabs",
    "ByteDance", "Bloomberg", "Ross Intelligence", "UMG"
}

FALSE_POSITIVE_TERMS = [
    "sanction", "attorney used", "legal brief", "hallucinated citation",
    "court rule", "administrative order", "criminal appeal",
    "bankruptcy", "immigration"
]

def auto_label(row):
    text = str(row.get("combined_text", "")).lower()
    companies = set(row["target_companies_clean"])
    claims = set(row["claim_types"])

    reasons = []

    has_core_company = len(companies & CORE_AI_COMPANIES) > 0
    has_copyright = "Copyright" in claims
    has_training = "Training Data" in claims

    if has_core_company:
        reasons.append("core_ai_company")

    if has_copyright:
        reasons.append("copyright_signal")

    if has_training:
        reasons.append("training_data_signal")

    if re.search(r"copyright infringement|fair use|training data|scrap|large language model|generative ai|chatgpt|claude|gemini", text):
        reasons.append("strong_ai_copyright_text")

    if any(term in text for term in FALSE_POSITIVE_TERMS):
        reasons.append("possible_false_positive")

    score = 0
    if has_core_company:
        score += 3
    if has_copyright:
        score += 3
    if has_training:
        score += 2
    if "strong_ai_copyright_text" in reasons:
        score += 2
    if "possible_false_positive" in reasons:
        score -= 3

    if score >= 6:
        label = 1
    elif score <= 2:
        label = 0
    else:
        label = "maybe"

    return pd.Series([label, score, "; ".join(reasons)])

df[["auto_relevant", "auto_score", "auto_reason"]] = df.apply(auto_label, axis=1)

# Sort so uncertain cases appear first for human checking
sort_order = {"maybe": 0, 1: 1, 0: 2}
df["review_priority"] = df["auto_relevant"].map(sort_order)

df = df.sort_values(["review_priority", "auto_score"], ascending=[True, False])

df["manual_relevant"] = ""
df["manual_notes"] = ""

df.to_excel(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print(df["auto_relevant"].value_counts(dropna=False))