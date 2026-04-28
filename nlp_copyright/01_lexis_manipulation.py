# ==========================================================
# FILE: 01_lexis_manipulation.py
#
# Input:
#   data_clean/07_final_case_dataset.xlsx
#
# Output:
#   data_nlp/lexis_structured_candidates.csv
#
# Purpose:
#   1. Manipulate LexisNexis cleaned dataset
#   2. Extract docket numbers such as 23-cv-11195 or 1:23-cv-11195
#   3. Filter copyright + AI/training-data cases
#   4. Produce structured candidates for CourtListener/RECAP query
# ==========================================================

import os
import re
import ast
import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "data_clean/07_final_case_dataset.xlsx"
OUTPUT_PATH = "data_nlp/lexis_structured_candidates.csv"

os.makedirs("data_nlp", exist_ok=True)


# ==========================================================
# HELPERS
# ==========================================================

def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x)


def clean_text(text):
    text = safe_text(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"LexisNexis|LEXISNEXIS", " ", text)
    return text.strip()


def parse_list_like(x):
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    x = str(x).strip()

    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return [str(v) for v in val]
        return [str(val)]
    except Exception:
        return [x]


def contains_any(text, terms):
    text = safe_text(text).lower()
    return any(t.lower() in text for t in terms)


def count_terms(text, terms):
    text = safe_text(text).lower()
    return sum(text.count(t.lower()) for t in terms)


def extract_year(text):
    match = re.search(r"\b(19\d{2}|20\d{2})\b", safe_text(text))
    return int(match.group(1)) if match else np.nan


def extract_docket_number(text):
    """
    Extract federal civil docket numbers.

    Examples:
    - 23-cv-11195
    - 1:23-cv-11195
    - 1:23-cv-11195-SHS-OTW
    - No. 23 Civ. 11195
    - Civil Action No. 23-11195
    """
    text = safe_text(text)

    patterns = [
        r"\b\d{1,2}:\d{2}-cv-\d{3,6}(?:-[A-Za-z0-9]+)*\b",
        r"\b\d{2}-cv-\d{3,6}(?:-[A-Za-z0-9]+)*\b",
        r"\b\d{1,2}:\d{2}-md-\d{3,6}(?:-[A-Za-z0-9]+)*\b",
        r"\b\d{2}-md-\d{3,6}(?:-[A-Za-z0-9]+)*\b",
        r"\bNo\.\s*\d{1,2}:\d{2}-cv-\d{3,6}(?:-[A-Za-z0-9]+)*\b",
        r"\bNo\.\s*\d{2}-cv-\d{3,6}(?:-[A-Za-z0-9]+)*\b",
        r"\b\d{2}\s*Civ\.\s*\d{3,6}\b",
        r"\bCivil Action No\.\s*\d{2}-\d{3,6}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            docket = match.group(0)
            docket = re.sub(r"^No\.\s*", "", docket, flags=re.I)
            docket = re.sub(r"^Civil Action No\.\s*", "", docket, flags=re.I)
            docket = docket.strip()
            docket = docket.replace("Civ.", "cv")
            docket = re.sub(r"\s+", "-", docket)
            return docket

    return ""


def normalize_case_query(case_name):
    """
    Remove LexisNexis citation/noise from case title before CourtListener search.
    """
    q = safe_text(case_name)

    q = re.sub(r",\s*\d{4}\s+.*$", "", q)
    q = re.sub(r",\s*\d+\s+F\..*$", "", q)
    q = re.sub(r",\s*\d+\s+U\.S\..*$", "", q)
    q = re.sub(r",\s*No\..*$", "", q, flags=re.I)
    q = re.sub(r"\s+", " ", q)

    return q.strip()


# ==========================================================
# LOAD LEXISNEXIS DATA
# ==========================================================

df = pd.read_excel(INPUT_PATH)

print("Loaded shape:", df.shape)
print("Columns:", df.columns.tolist())


# ==========================================================
# 1. BUILD LEXIS TEXT
# ==========================================================

TEXT_FIELDS = [
    "title",
    "overview",
    "headnotes",
    "core_terms",
    "disposition",
    "history",
    "opinion",
    "combined_text",
    "claim_types",
    "target_companies_clean",
    "manual_notes",
]

available_text_fields = [c for c in TEXT_FIELDS if c in df.columns]

def build_lexis_text(row):
    parts = []

    for c in [
        "title",
        "overview",
        "headnotes",
        "core_terms",
        "disposition",
        "history",
        "opinion",
        "combined_text",
        "claim_types",
        "target_companies_clean",
        "manual_notes",
    ]:
        if c in row.index and pd.notna(row[c]):
            parts.append(str(row[c]))

    return clean_text(" ".join(parts))


df["lexis_case_text"] = df.apply(build_lexis_text, axis=1)


# ==========================================================
# 2. BASIC METADATA
# ==========================================================

if "title" not in df.columns:
    raise ValueError("Need a 'title' column in the LexisNexis dataset.")

df["case_name"] = df["title"].fillna("").astype(str)
df["courtlistener_query"] = df["case_name"].apply(normalize_case_query)

# Extract docket number from multiple possible fields
docket_source_cols = [
    c for c in [
        "title",
        "number",
        "citation",
        "overview",
        "history",
        "disposition",
        "combined_text",
        "opinion",
    ]
    if c in df.columns
]

df["docket_search_blob"] = df[docket_source_cols].fillna("").astype(str).agg(" ".join, axis=1)
df["docket_number_clean"] = df["docket_search_blob"].apply(extract_docket_number)

# Year
df["year"] = df["case_name"].apply(extract_year)
missing_year = df["year"].isna()
df.loc[missing_year, "year"] = df.loc[missing_year, "lexis_case_text"].apply(extract_year)

# Court
if "court" in df.columns:
    df["court_clean"] = df["court"].fillna("").astype(str)
else:
    df["court_clean"] = ""

if "publication_location" in df.columns:
    df["court_clean"] = np.where(
        df["court_clean"].str.strip().eq("") | df["court_clean"].str.lower().eq("nan"),
        df["publication_location"].fillna("").astype(str),
        df["court_clean"],
    )

if "court_code" in df.columns:
    df["court_code_clean"] = df["court_code"].fillna("").astype(str)
else:
    df["court_code_clean"] = ""

# Judge
if "judges" in df.columns:
    df["judge_clean"] = df["judges"].fillna("").astype(str)
else:
    df["judge_clean"] = ""

if "opinion_by" in df.columns:
    df["opinion_by_clean"] = df["opinion_by"].fillna("").astype(str)
else:
    df["opinion_by_clean"] = ""

# Citation
df["citation_clean"] = df["case_name"].str.extract(
    r",\s*([^,]*?(?:LEXIS|F\. Supp\.|F\.4th|F\.3d|U\.S\.|WL).*)$"
)[0]

if "citation" in df.columns:
    df["citation_clean"] = df["citation_clean"].fillna(df["citation"])

df["citation_clean"] = df["citation_clean"].fillna("").astype(str)


# ==========================================================
# 3. FILTER COPYRIGHT + AI/TRAINING CASES
# ==========================================================

COPYRIGHT_TERMS = [
    "copyright",
    "fair use",
    "dmca",
    "copyright infringement",
    "derivative work",
    "substantial similarity",
    "17 u.s.c.",
    "section 106",
    "section 107",
]

AI_TERMS = [
    "openai",
    "chatgpt",
    "anthropic",
    "claude",
    "stability ai",
    "stable diffusion",
    "midjourney",
    "meta",
    "llama",
    "microsoft",
    "copilot",
    "google",
    "gemini",
    "artificial intelligence",
    "generative ai",
    "large language model",
    "llm",
    "machine learning",
    "training data",
    "model training",
    "ross intelligence",
    "quizlet",
]

TRAINING_DATA_TERMS = [
    "training data",
    "trained on",
    "model training",
    "training corpus",
    "dataset",
    "corpus",
    "scraped",
    "scraping",
    "web scraping",
    "large language model",
    "generative ai",
]

if "claim_types" in df.columns:
    df["claim_type_list"] = df["claim_types"].apply(parse_list_like)
else:
    df["claim_type_list"] = [[] for _ in range(len(df))]

df["claim_types_clean"] = df["claim_type_list"].apply(lambda xs: "; ".join(xs))

df["has_copyright_claim"] = (
    df["claim_type_list"].apply(lambda xs: any("copyright" in x.lower() for x in xs))
    | df["lexis_case_text"].apply(lambda x: contains_any(x, COPYRIGHT_TERMS))
)

df["has_training_data_issue"] = (
    df["claim_type_list"].apply(lambda xs: any("training data" in x.lower() for x in xs))
    | df["lexis_case_text"].apply(lambda x: contains_any(x, TRAINING_DATA_TERMS))
)

df["has_ai_signal"] = (
    df["lexis_case_text"].apply(lambda x: contains_any(x, AI_TERMS))
    | df["case_name"].apply(lambda x: contains_any(x, AI_TERMS))
)

if "final_relevant" in df.columns:
    df["is_final_relevant"] = df["final_relevant"].fillna(0).astype(int).eq(1)
else:
    df["is_final_relevant"] = True

candidate_df = df[
    df["is_final_relevant"]
    & df["has_copyright_claim"]
    & (df["has_ai_signal"] | df["has_training_data_issue"])
].copy()

print("Candidate copyright + AI/training-data cases:", candidate_df.shape)


# ==========================================================
# 4. LEXIS STRUCTURED FIELDS
# ==========================================================

AI_DEFENDANT_PATTERNS = {
    "OpenAI": [r"\bOpenAI\b"],
    "Microsoft": [r"\bMicrosoft\b"],
    "Anthropic": [r"\bAnthropic\b"],
    "Google": [r"\bGoogle\b", r"\bAlphabet\b"],
    "Meta": [r"\bMeta\b", r"\bFacebook\b"],
    "Stability AI": [r"\bStability AI\b", r"\bStable Diffusion\b"],
    "Midjourney": [r"\bMidjourney\b"],
    "Ross Intelligence": [r"\bRoss Intel", r"\bRoss Intelligence\b"],
    "Quizlet": [r"\bQuizlet\b"],
    "Perlmutter / Copyright Office": [r"\bPerlmutter\b", r"\bCopyright Office\b"],
}

def extract_true_ai_defendants_from_title(case_name):
    found = []
    for company, patterns in AI_DEFENDANT_PATTERNS.items():
        if any(re.search(p, safe_text(case_name), flags=re.I) for p in patterns):
            found.append(company)
    return "; ".join(sorted(set(found)))


def extract_possible_ai_mentions(text):
    found = []
    for company, patterns in AI_DEFENDANT_PATTERNS.items():
        if any(re.search(p, safe_text(text), flags=re.I) for p in patterns):
            found.append(company)
    return "; ".join(sorted(set(found)))


candidate_df["true_ai_defendants_lexis_title"] = candidate_df["case_name"].apply(
    extract_true_ai_defendants_from_title
)

candidate_df["possible_ai_mentions_lexis_text"] = candidate_df["lexis_case_text"].apply(
    extract_possible_ai_mentions
)


PLAINTIFF_TYPE_TERMS = {
    "news_publisher": [
        "new york times",
        "daily news",
        "newspaper",
        "journalism",
        "news publisher",
    ],
    "book_author_or_author_group": [
        "authors guild",
        "author",
        "novelist",
        "writer",
        "book",
    ],
    "music_rightsholder": [
        "concord music",
        "music",
        "lyrics",
        "song",
        "composer",
        "publisher plaintiffs",
    ],
    "visual_artist_or_image_owner": [
        "artist",
        "photograph",
        "image",
        "visual",
        "illustrator",
        "stability",
    ],
    "legal_database_or_research_provider": [
        "thomson reuters",
        "westlaw",
        "legal research",
        "ross intelligence",
    ],
    "education_content_owner": [
        "quizlet",
        "textbook",
        "educational",
        "barkley",
    ],
    "copyright_registration_ai_author": [
        "thaler",
        "copyright office",
        "human authorship",
    ],
}

def classify_plaintiff_type(row):
    text = (
        safe_text(row.get("case_name", ""))
        + " "
        + safe_text(row.get("lexis_case_text", ""))
    ).lower()

    scores = {}
    for label, terms in PLAINTIFF_TYPE_TERMS.items():
        scores[label] = sum(text.count(t.lower()) for t in terms)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


candidate_df["plaintiff_type_lexis"] = candidate_df.apply(classify_plaintiff_type, axis=1)


DOCTRINE_TERMS = {
    "fair_use_issue": [
        "fair use",
        "transformative",
        "section 107",
        "17 u.s.c. 107",
        "purpose and character",
        "market effect",
    ],
    "training_data_issue": [
        "training data",
        "trained on",
        "model training",
        "dataset",
        "corpus",
    ],
    "scraping_issue": [
        "scraping",
        "scraped",
        "web scraping",
        "crawl",
        "crawled",
    ],
    "market_harm_issue": [
        "market harm",
        "licensing market",
        "lost licensing",
        "substitution",
        "potential market",
        "lost sales",
    ],
    "memorization_or_output_issue": [
        "memorization",
        "regurgitation",
        "verbatim",
        "output",
        "substantially similar output",
    ],
    "dmca_issue": [
        "dmca",
        "1202",
        "copyright management information",
        "cmi",
    ],
    "derivative_work_issue": [
        "derivative work",
        "derivative works",
        "adaptation",
    ],
    "substantial_similarity_issue": [
        "substantial similarity",
        "substantially similar",
    ],
    "injunction_issue": [
        "injunction",
        "preliminary injunction",
        "permanent injunction",
        "enjoin",
    ],
    "damages_issue": [
        "damages",
        "statutory damages",
        "actual damages",
        "profits",
        "disgorgement",
    ],
}

for label, terms in DOCTRINE_TERMS.items():
    candidate_df[label + "_lexis_count"] = candidate_df["lexis_case_text"].apply(
        lambda x: count_terms(x, terms)
    )
    candidate_df[label + "_lexis"] = candidate_df[label + "_lexis_count"] > 0


PROCEDURAL_PATTERNS = {
    "motion_to_dismiss": ["motion to dismiss", "motions to dismiss", "dismissal"],
    "summary_judgment": ["summary judgment"],
    "preliminary_injunction": ["preliminary injunction"],
    "discovery": ["discovery", "compel", "protective order", "subpoena"],
    "class_certification": ["class certification", "certify the class", "class action"],
    "transfer_or_mdl": [
        "transfer",
        "multidistrict litigation",
        "mdl",
        "judicial panel on multidistrict litigation",
    ],
    "appeal": ["appeal", "appellate", "court of appeals"],
    "copyright_registration": ["copyright registration", "register", "registration"],
}

def detect_procedural_stage(text):
    text = safe_text(text).lower()
    found = []
    for stage, terms in PROCEDURAL_PATTERNS.items():
        if any(t in text for t in terms):
            found.append(stage)
    return "; ".join(sorted(set(found))) if found else "unknown"


candidate_df["procedural_stage_lexis"] = candidate_df["lexis_case_text"].apply(
    detect_procedural_stage
)


OUTCOME_PATTERNS = {
    "granted": ["is granted", "granted in part", "motion granted", "granted"],
    "denied": ["is denied", "denied in part", "motion denied", "denied"],
    "dismissed": [
        "dismissed with prejudice",
        "dismissed without prejudice",
        "dismissed",
    ],
    "affirmed": ["affirmed"],
    "reversed": ["reversed"],
    "settled_or_stipulated": ["settlement", "settled", "stipulation of dismissal"],
    "transferred": ["transferred", "transfer granted"],
}

def detect_outcome(row):
    disp = safe_text(row.get("disposition", ""))
    text = (disp + " " + safe_text(row.get("lexis_case_text", ""))).lower()

    found = []
    for outcome, terms in OUTCOME_PATTERNS.items():
        if any(t in text for t in terms):
            found.append(outcome)

    return "; ".join(sorted(set(found))) if found else "unknown"


candidate_df["outcome_detected_lexis"] = candidate_df.apply(detect_outcome, axis=1)


FAIR_USE_FACTORS = {
    "fair_use_factor1_purpose_lexis_count": [
        "purpose and character",
        "transformative",
        "commercial use",
        "bad faith",
        "good faith",
    ],
    "fair_use_factor2_nature_lexis_count": [
        "nature of the copyrighted work",
        "creative work",
        "factual work",
        "published",
        "unpublished",
    ],
    "fair_use_factor3_amount_lexis_count": [
        "amount and substantiality",
        "entire work",
        "heart of the work",
        "quantitatively",
        "qualitatively",
    ],
    "fair_use_factor4_market_lexis_count": [
        "effect upon the potential market",
        "market effect",
        "market harm",
        "licensing market",
        "substitution",
    ],
}

for col, terms in FAIR_USE_FACTORS.items():
    candidate_df[col] = candidate_df["lexis_case_text"].apply(
        lambda x: count_terms(x, terms)
    )


def make_case_summary(row, max_chars=900):
    overview = safe_text(row.get("overview", ""))
    if len(overview.strip()) > 50:
        return clean_text(overview)[:max_chars]

    headnotes = safe_text(row.get("headnotes", ""))
    if len(headnotes.strip()) > 50:
        return clean_text(headnotes)[:max_chars]

    return safe_text(row.get("lexis_case_text", ""))[:max_chars]


candidate_df["case_summary_lexis"] = candidate_df.apply(make_case_summary, axis=1)
candidate_df["lexis_text_chars"] = candidate_df["lexis_case_text"].str.len()
candidate_df["lexis_text_words"] = candidate_df["lexis_case_text"].str.split().str.len()


# ==========================================================
# SELECT OUTPUT COLUMNS
# ==========================================================

preferred_cols = [
    "case_name",
    "courtlistener_query",
    "docket_number_clean",
    "citation_clean",
    "year",
    "court_clean",
    "court_code_clean",
    "judge_clean",
    "opinion_by_clean",
    "publication_location",
    "publication_type",
    "number",
    "claim_types_clean",
    "target_companies_clean",
    "final_relevant",
    "manual_notes",
    "true_ai_defendants_lexis_title",
    "possible_ai_mentions_lexis_text",
    "plaintiff_type_lexis",
    "has_copyright_claim",
    "has_training_data_issue",
    "has_ai_signal",
    "procedural_stage_lexis",
    "outcome_detected_lexis",
    "case_summary_lexis",
    "lexis_case_text",
    "lexis_text_chars",
    "lexis_text_words",
]

for label in DOCTRINE_TERMS:
    preferred_cols.extend([label + "_lexis", label + "_lexis_count"])

preferred_cols.extend(list(FAIR_USE_FACTORS.keys()))

final_cols = [c for c in preferred_cols if c in candidate_df.columns]

final_df = candidate_df[final_cols].copy()

final_df = final_df.drop_duplicates(
    subset=["case_name", "citation_clean", "docket_number_clean"],
    keep="first",
)

final_df = final_df.sort_values(["year", "case_name"], na_position="last").reset_index(drop=True)

final_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved Lexis structured candidates -> {OUTPUT_PATH}")
print("Final shape:", final_df.shape)

preview_cols = [
    "case_name",
    "docket_number_clean",
    "courtlistener_query",
    "year",
    "court_clean",
    "true_ai_defendants_lexis_title",
    "plaintiff_type_lexis",
    "procedural_stage_lexis",
    "outcome_detected_lexis",
]

preview_cols = [c for c in preview_cols if c in final_df.columns]
print(final_df[preview_cols].head(30).to_string(index=False))