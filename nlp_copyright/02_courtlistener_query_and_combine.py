# ==========================================================
# FILE: 02_courtlistener_query_and_combine.py
#
# Input:
#   data_nlp/lexis_structured_candidates.csv
#
# Output:
#   data_nlp/enriched_copyright_cases.csv
#
# Purpose:
#   1. Use docket number first to search CourtListener RECAP
#   2. Fall back to case name search in RECAP
#   3. Fall back to CourtListener opinion search
#   4. Combine LexisNexis + CourtListener/RECAP metadata/text
#
# Install:
#   pip install pandas requests tqdm python-dotenv
#
# Optional:
#   Create .env file:
#   COURTLISTENER_TOKEN=your_free_token_here
# ==========================================================

import os
import re
import json
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "data_nlp/lexis_structured_candidates.csv"
OUTPUT_PATH = "data_nlp/enriched_copyright_cases.csv"
CACHE_PATH = "data_nlp/courtlistener_recap_cache.json"

BASE_URL = "https://www.courtlistener.com/api/rest/v4"
COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN", "")

os.makedirs("data_nlp", exist_ok=True)


# ==========================================================
# HELPERS
# ==========================================================

def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x)


def clean_html_text(text):
    text = safe_text(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_query(q):
    q = safe_text(q)
    q = re.sub(r",\s*\d{4}\s+.*$", "", q)
    q = re.sub(r",\s*\d+\s+F\..*$", "", q)
    q = re.sub(r",\s*\d+\s+U\.S\..*$", "", q)
    q = re.sub(r",\s*No\..*$", "", q, flags=re.I)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def extract_year(text):
    match = re.search(r"\b(19\d{2}|20\d{2})\b", safe_text(text))
    return int(match.group(1)) if match else np.nan


def headers():
    h = {
        "User-Agent": "AI-Copyright-Litigation-Research/1.0"
    }

    if COURTLISTENER_TOKEN:
        h["Authorization"] = f"Token {COURTLISTENER_TOKEN}"

    return h


# ==========================================================
# CACHE
# ==========================================================

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}


def save_cache():
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def safe_get(url, params=None, sleep=0.7):
    """
    Polite GET with caching.
    """
    cache_key = json.dumps(
        {"url": url, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )

    if cache_key in cache:
        return cache[cache_key]

    time.sleep(sleep)

    try:
        r = requests.get(
            url,
            params=params,
            headers=headers(),
            timeout=45,
        )

        if r.status_code == 429:
            print("Rate limited. Sleeping 20 seconds...")
            time.sleep(20)
            r = requests.get(
                url,
                params=params,
                headers=headers(),
                timeout=45,
            )

        if r.status_code == 401:
            print("401 Unauthorized. Add a free COURTLISTENER_TOKEN if needed.")
            return None

        if r.status_code >= 400:
            print("Request error:", r.status_code, url, params)
            return None

        data = r.json()
        cache[cache_key] = data
        save_cache()

        return data

    except Exception as e:
        print("Request failed:", e)
        return None


# ==========================================================
# COURTLISTENER SEARCH FUNCTIONS
# ==========================================================

def search_api(q, result_type):
    """
    CourtListener v4 search endpoint.

    result_type:
      - r: RECAP federal cases/dockets with nested documents
      - o: opinions
    """
    if not q or len(q.strip()) == 0:
        return None

    return safe_get(
        f"{BASE_URL}/search/",
        params={
            "q": q,
            "type": result_type,
            "order_by": "score desc",
        },
    )


def get_first_result(search_json):
    if not search_json:
        return None

    results = search_json.get("results", [])

    if not results:
        return None

    return results[0]


def search_recap_by_docket(docket_number):
    """
    Search RECAP using docketNumber field first.
    """
    docket_number = safe_text(docket_number).strip()

    if not docket_number:
        return None, ""

    queries = [
        f'docketNumber:"{docket_number}"',
        f'"{docket_number}"',
        docket_number,
    ]

    # Also try without judge suffix, e.g. 23-cv-11195-SHS-OTW -> 23-cv-11195
    base = re.sub(r"(-[A-Za-z]{2,5})+$", "", docket_number)
    if base != docket_number:
        queries.extend([
            f'docketNumber:"{base}"',
            f'"{base}"',
            base,
        ])

    for q in queries:
        data = search_api(q, result_type="r")
        if data and data.get("results"):
            return data, q

    return None, queries[0]


def search_recap_by_case_name(case_name, citation=""):
    """
    Search RECAP by case name / citation.
    """
    case_name = normalize_query(case_name)
    citation = safe_text(citation)

    queries = []

    if case_name:
        queries.append(case_name)
        queries.append(f'"{case_name}"')

    if citation:
        queries.append(citation)
        queries.append(f'"{citation}"')

    for q in queries:
        data = search_api(q, result_type="r")
        if data and data.get("results"):
            return data, q

    return None, queries[0] if queries else ""


def search_opinion_by_docket_or_case(docket_number, case_name, citation=""):
    """
    Search opinions as fallback.
    """
    docket_number = safe_text(docket_number).strip()
    case_name = normalize_query(case_name)
    citation = safe_text(citation)

    queries = []

    if docket_number:
        queries.append(f'docketNumber:"{docket_number}"')
        queries.append(f'"{docket_number}"')
        queries.append(docket_number)

        base = re.sub(r"(-[A-Za-z]{2,5})+$", "", docket_number)
        if base != docket_number:
            queries.append(f'docketNumber:"{base}"')
            queries.append(base)

    if case_name:
        queries.append(case_name)
        queries.append(f'"{case_name}"')

    if citation:
        queries.append(citation)
        queries.append(f'"{citation}"')

    for q in queries:
        data = search_api(q, result_type="o")
        if data and data.get("results"):
            return data, q

    return None, queries[0] if queries else ""


def get_resource(url):
    if not url:
        return None

    if url.startswith("/"):
        url = "https://www.courtlistener.com" + url

    return safe_get(url)


def get_opinion_by_id(opinion_id):
    if not opinion_id:
        return None

    return safe_get(f"{BASE_URL}/opinions/{opinion_id}/")


def extract_text_from_opinion(opinion):
    if not opinion:
        return ""

    fields = [
        "html_with_citations",
        "plain_text",
        "html",
        "html_lawbox",
        "html_columbia",
        "html_anon_2020",
        "xml_harvard",
    ]

    texts = []

    for f in fields:
        val = opinion.get(f, "")
        if val:
            texts.append(clean_html_text(val))

    if not texts:
        return ""

    return max(texts, key=len)


def citation_to_string(citation_obj):
    if not citation_obj:
        return ""

    if isinstance(citation_obj, list):
        vals = []

        for item in citation_obj:
            if isinstance(item, dict):
                if item.get("cite"):
                    vals.append(item["cite"])
            else:
                vals.append(str(item))

        return "; ".join([v for v in vals if v])

    return str(citation_obj)


# ==========================================================
# EXTRACT RECAP / OPINION RESULT FIELDS
# ==========================================================

def parse_recap_result(result):
    out = {
        "recap_found": False,
        "recap_result_id": "",
        "recap_case_name": "",
        "recap_court": "",
        "recap_date_filed": "",
        "recap_docket_number": "",
        "recap_suit_nature": "",
        "recap_assigned_to": "",
        "recap_referred_to": "",
        "recap_absolute_url": "",
        "recap_more_docs": "",
        "recap_nested_docs_count": 0,
        "recap_doc_descriptions": "",
    }

    if not result:
        return out

    out["recap_found"] = True
    out["recap_result_id"] = safe_text(result.get("id", ""))
    out["recap_case_name"] = safe_text(
        result.get("caseName", "")
        or result.get("case_name", "")
        or result.get("caseNameFull", "")
    )
    out["recap_court"] = safe_text(result.get("court", "") or result.get("court_id", ""))
    out["recap_date_filed"] = safe_text(
        result.get("dateFiled", "")
        or result.get("date_filed", "")
        or result.get("dateCreated", "")
    )
    out["recap_docket_number"] = safe_text(
        result.get("docketNumber", "")
        or result.get("docket_number", "")
    )
    out["recap_suit_nature"] = safe_text(
        result.get("suitNature", "")
        or result.get("suit_nature", "")
        or result.get("nature_of_suit", "")
    )
    out["recap_assigned_to"] = safe_text(
        result.get("assignedTo", "")
        or result.get("assigned_to", "")
    )
    out["recap_referred_to"] = safe_text(
        result.get("referredTo", "")
        or result.get("referred_to", "")
    )
    out["recap_absolute_url"] = safe_text(result.get("absolute_url", ""))
    out["recap_more_docs"] = safe_text(result.get("more_docs", ""))

    docs = (
        result.get("documents", [])
        or result.get("recap_documents", [])
        or result.get("entries", [])
        or []
    )

    if isinstance(docs, list):
        out["recap_nested_docs_count"] = len(docs)

        descriptions = []
        for d in docs:
            if isinstance(d, dict):
                desc = (
                    d.get("description", "")
                    or d.get("short_description", "")
                    or d.get("document_type", "")
                    or d.get("plain_text", "")
                )
                if desc:
                    descriptions.append(clean_html_text(desc)[:200])

        out["recap_doc_descriptions"] = " | ".join(descriptions[:5])

    return out


def parse_opinion_result(result):
    out = {
        "opinion_found": False,
        "opinion_result_id": "",
        "opinion_case_name": "",
        "opinion_court": "",
        "opinion_date_filed": "",
        "opinion_citation": "",
        "opinion_absolute_url": "",
        "opinion_cluster_url": "",
        "opinion_docket_url": "",
        "opinion_text": "",
        "opinion_text_chars": 0,
    }

    if not result:
        return out

    out["opinion_found"] = True
    out["opinion_result_id"] = safe_text(result.get("id", ""))
    out["opinion_case_name"] = safe_text(
        result.get("caseName", "")
        or result.get("case_name", "")
        or result.get("caseNameFull", "")
    )
    out["opinion_court"] = safe_text(result.get("court", ""))
    out["opinion_date_filed"] = safe_text(
        result.get("dateFiled", "")
        or result.get("date_filed", "")
    )
    out["opinion_citation"] = citation_to_string(result.get("citation", ""))
    out["opinion_absolute_url"] = safe_text(result.get("absolute_url", ""))

    snippet_text = " ".join([
        safe_text(result.get("snippet", "")),
        safe_text(result.get("plain_text", "")),
        safe_text(result.get("html", "")),
    ])

    out["opinion_text"] = clean_html_text(snippet_text)

    opinion_id = result.get("id", "")
    opinion = get_opinion_by_id(opinion_id)

    if opinion:
        full_text = extract_text_from_opinion(opinion)

        if len(full_text) > len(out["opinion_text"]):
            out["opinion_text"] = full_text

        out["opinion_cluster_url"] = safe_text(opinion.get("cluster", ""))

        cluster = get_resource(out["opinion_cluster_url"])
        if cluster:
            out["opinion_case_name"] = safe_text(
                cluster.get("case_name", out["opinion_case_name"])
            )
            out["opinion_date_filed"] = safe_text(
                cluster.get("date_filed", out["opinion_date_filed"])
            )

            citations = cluster.get("citations", [])
            if citations:
                out["opinion_citation"] = citation_to_string(citations)

            out["opinion_docket_url"] = safe_text(cluster.get("docket", ""))

    out["opinion_text_chars"] = len(out["opinion_text"])

    return out


# ==========================================================
# CASE-LEVEL ENRICHMENT
# ==========================================================

def enrich_one_case(row):
    docket_number = safe_text(row.get("docket_number_clean", ""))
    case_name = safe_text(row.get("courtlistener_query", "")) or safe_text(row.get("case_name", ""))
    citation = safe_text(row.get("citation_clean", ""))

    output = {
        "recap_query_used": "",
        "opinion_query_used": "",
    }

    # 1. RECAP search by docket number
    recap_json, recap_query = search_recap_by_docket(docket_number)
    recap_result = get_first_result(recap_json)

    # 2. RECAP fallback by case name
    if recap_result is None:
        recap_json, recap_query = search_recap_by_case_name(case_name, citation)
        recap_result = get_first_result(recap_json)

    # 3. Opinion search fallback / supplement
    opinion_json, opinion_query = search_opinion_by_docket_or_case(
        docket_number=docket_number,
        case_name=case_name,
        citation=citation,
    )
    opinion_result = get_first_result(opinion_json)

    output["recap_query_used"] = recap_query
    output["opinion_query_used"] = opinion_query

    output.update(parse_recap_result(recap_result))
    output.update(parse_opinion_result(opinion_result))

    return output


# ==========================================================
# FINAL COMBINE HELPERS
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

def extract_true_ai_defendants(case_name):
    found = []

    for company, patterns in AI_DEFENDANT_PATTERNS.items():
        if any(re.search(p, safe_text(case_name), flags=re.I) for p in patterns):
            found.append(company)

    return "; ".join(sorted(set(found)))


def choose_case_name(row):
    recap_name = safe_text(row.get("recap_case_name", ""))
    opinion_name = safe_text(row.get("opinion_case_name", ""))
    lexis_name = safe_text(row.get("case_name", ""))

    if len(recap_name) > 3:
        return recap_name

    if len(opinion_name) > 3:
        return opinion_name

    return lexis_name


def choose_year(row):
    for col in ["recap_date_filed", "opinion_date_filed"]:
        dt = pd.to_datetime(row.get(col, ""), errors="coerce")
        if not pd.isna(dt):
            return int(dt.year)

    year = row.get("year", np.nan)
    try:
        if not pd.isna(year):
            return int(year)
    except Exception:
        pass

    return extract_year(row.get("case_name_final", ""))


def choose_court(row):
    recap_court = safe_text(row.get("recap_court", ""))
    opinion_court = safe_text(row.get("opinion_court", ""))
    lexis_court = safe_text(row.get("court_clean", ""))

    if recap_court:
        return recap_court

    if opinion_court:
        return opinion_court

    return lexis_court


def choose_judge(row):
    recap_judge = safe_text(row.get("recap_assigned_to", ""))
    lexis_judge = safe_text(row.get("judge_clean", ""))

    if recap_judge:
        return recap_judge

    return lexis_judge


def choose_docket_number(row):
    recap_docket = safe_text(row.get("recap_docket_number", ""))
    lexis_docket = safe_text(row.get("docket_number_clean", ""))

    if recap_docket:
        return recap_docket

    return lexis_docket


def choose_case_text(row):
    """
    Use opinion text if substantial.
    RECAP usually gives docket metadata, not clean full opinion text.
    Otherwise keep Lexis text.
    """
    opinion_text = safe_text(row.get("opinion_text", ""))
    lexis_text = safe_text(row.get("lexis_case_text", ""))

    if len(opinion_text) >= 1200:
        return opinion_text

    return lexis_text


def choose_text_source(row):
    if safe_text(row.get("opinion_text", "")) and len(safe_text(row.get("opinion_text", ""))) >= 1200:
        return "CourtListener Opinion"

    if bool(row.get("recap_found", False)):
        return "LexisNexis Text + RECAP Metadata"

    return "LexisNexis Only"


def count_terms(text, terms):
    text = safe_text(text).lower()
    return sum(text.count(t.lower()) for t in terms)


def detect_procedural_stage(text):
    text = safe_text(text).lower()

    stages = []

    if "motion to dismiss" in text:
        stages.append("motion_to_dismiss")
    if "summary judgment" in text:
        stages.append("summary_judgment")
    if "preliminary injunction" in text:
        stages.append("preliminary_injunction")
    if "discovery" in text or "subpoena" in text or "compel" in text:
        stages.append("discovery")
    if "class certification" in text or "class action" in text:
        stages.append("class_action")
    if "multidistrict litigation" in text or "mdl" in text or "transfer" in text:
        stages.append("transfer_or_mdl")
    if "appeal" in text or "court of appeals" in text:
        stages.append("appeal")
    if "copyright registration" in text or "human authorship" in text:
        stages.append("copyright_registration")

    return "; ".join(sorted(set(stages))) if stages else "unknown"


def detect_outcome(text):
    text = safe_text(text).lower()

    outcomes = []

    if "granted in part and denied in part" in text:
        outcomes.append("granted_in_part_denied_in_part")
    else:
        if "granted" in text:
            outcomes.append("granted")
        if "denied" in text:
            outcomes.append("denied")

    if "dismissed with prejudice" in text:
        outcomes.append("dismissed_with_prejudice")
    elif "dismissed without prejudice" in text:
        outcomes.append("dismissed_without_prejudice")
    elif "dismissed" in text:
        outcomes.append("dismissed")

    if "affirmed" in text:
        outcomes.append("affirmed")
    if "reversed" in text:
        outcomes.append("reversed")
    if "settlement" in text or "settled" in text or "stipulation of dismissal" in text:
        outcomes.append("settled_or_stipulated")
    if "transferred" in text:
        outcomes.append("transferred")

    return "; ".join(sorted(set(outcomes))) if outcomes else "unknown"


def detect_motion_outcome(text):
    text = safe_text(text).lower()

    if "motion to dismiss" in text:
        if "granted in part and denied in part" in text:
            return "motion_to_dismiss_granted_in_part_denied_in_part"
        if "motion to dismiss is granted" in text or "motion to dismiss granted" in text:
            return "motion_to_dismiss_granted"
        if "motion to dismiss is denied" in text or "motion to dismiss denied" in text:
            return "motion_to_dismiss_denied"

    if "summary judgment" in text:
        if "summary judgment is granted" in text or "summary judgment granted" in text:
            return "summary_judgment_granted"
        if "summary judgment is denied" in text or "summary judgment denied" in text:
            return "summary_judgment_denied"

    if "preliminary injunction" in text:
        if "preliminary injunction is granted" in text:
            return "preliminary_injunction_granted"
        if "preliminary injunction is denied" in text:
            return "preliminary_injunction_denied"

    return "unknown"


# ==========================================================
# RUN
# ==========================================================

df = pd.read_csv(INPUT_PATH)

print("Loaded candidates:", df.shape)

enriched_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    enriched_rows.append(enrich_one_case(row))

api_df = pd.DataFrame(enriched_rows)

combined = pd.concat(
    [
        df.reset_index(drop=True),
        api_df.reset_index(drop=True),
    ],
    axis=1,
)

# -----------------------------
# Final combined fields
# -----------------------------
combined["case_name_final"] = combined.apply(choose_case_name, axis=1)
combined["year_final"] = combined.apply(choose_year, axis=1)
combined["court_final"] = combined.apply(choose_court, axis=1)
combined["judge_final"] = combined.apply(choose_judge, axis=1)
combined["docket_number_final"] = combined.apply(choose_docket_number, axis=1)
combined["case_detail_text"] = combined.apply(choose_case_text, axis=1)
combined["data_source_for_text"] = combined.apply(choose_text_source, axis=1)

combined["true_ai_defendants_final"] = combined["case_name_final"].apply(
    extract_true_ai_defendants
)

# If final defendant is empty, keep Lexis title extraction
combined["true_ai_defendants_final"] = np.where(
    combined["true_ai_defendants_final"].str.len().gt(0),
    combined["true_ai_defendants_final"],
    combined.get("true_ai_defendants_lexis_title", "").fillna("").astype(str),
)

combined["case_detail_text_chars"] = combined["case_detail_text"].str.len()
combined["case_detail_text_words"] = combined["case_detail_text"].str.split().str.len()

combined["procedural_stage_final"] = combined["case_detail_text"].apply(
    detect_procedural_stage
)
combined["outcome_detected_final"] = combined["case_detail_text"].apply(
    detect_outcome
)
combined["motion_outcome_specific_final"] = combined["case_detail_text"].apply(
    detect_motion_outcome
)


# -----------------------------
# Final doctrine scores
# -----------------------------
DOCTRINE_TERMS = {
    "fair_use": [
        "fair use",
        "transformative",
        "section 107",
        "17 u.s.c. 107",
        "purpose and character",
        "market effect",
    ],
    "training_data": [
        "training data",
        "trained on",
        "model training",
        "dataset",
        "corpus",
    ],
    "scraping": [
        "scraping",
        "scraped",
        "web scraping",
        "crawl",
        "crawled",
    ],
    "market_harm": [
        "market harm",
        "licensing market",
        "lost licensing",
        "substitution",
        "potential market",
        "lost sales",
    ],
    "memorization_or_output": [
        "memorization",
        "regurgitation",
        "verbatim",
        "output",
        "substantially similar output",
    ],
    "dmca": [
        "dmca",
        "1202",
        "copyright management information",
        "cmi",
    ],
    "derivative_work": [
        "derivative work",
        "derivative works",
        "adaptation",
    ],
    "substantial_similarity": [
        "substantial similarity",
        "substantially similar",
    ],
    "injunction": [
        "injunction",
        "preliminary injunction",
        "permanent injunction",
        "enjoin",
    ],
    "damages": [
        "damages",
        "statutory damages",
        "actual damages",
        "profits",
        "disgorgement",
    ],
}

for label, terms in DOCTRINE_TERMS.items():
    combined[label + "_count_final"] = combined["case_detail_text"].apply(
        lambda x: count_terms(x, terms)
    )
    combined[label + "_flag_final"] = combined[label + "_count_final"] > 0


FAIR_USE_FACTORS = {
    "fair_use_factor1_purpose_final_count": [
        "purpose and character",
        "transformative",
        "commercial use",
        "bad faith",
        "good faith",
    ],
    "fair_use_factor2_nature_final_count": [
        "nature of the copyrighted work",
        "creative work",
        "factual work",
        "published",
        "unpublished",
    ],
    "fair_use_factor3_amount_final_count": [
        "amount and substantiality",
        "entire work",
        "heart of the work",
        "quantitatively",
        "qualitatively",
    ],
    "fair_use_factor4_market_final_count": [
        "effect upon the potential market",
        "market effect",
        "market harm",
        "licensing market",
        "substitution",
    ],
}

for col, terms in FAIR_USE_FACTORS.items():
    combined[col] = combined["case_detail_text"].apply(
        lambda x: count_terms(x, terms)
    )


def make_case_summary(row, max_chars=1000):
    recap_docs = safe_text(row.get("recap_doc_descriptions", ""))
    lexis_summary = safe_text(row.get("case_summary_lexis", ""))
    opinion_text = safe_text(row.get("opinion_text", ""))
    case_text = safe_text(row.get("case_detail_text", ""))

    if len(lexis_summary) > 50:
        return lexis_summary[:max_chars]

    if len(recap_docs) > 50:
        return recap_docs[:max_chars]

    if len(opinion_text) > 1200:
        return opinion_text[:max_chars]

    return case_text[:max_chars]


combined["case_summary_final"] = combined.apply(make_case_summary, axis=1)

# -----------------------------
# Select output columns
# -----------------------------
preferred_cols = [
    # final identifiers
    "case_name_final",
    "case_name",
    "year_final",
    "court_final",
    "judge_final",
    "docket_number_final",
    "docket_number_clean",
    "citation_clean",

    # CourtListener / RECAP
    "recap_found",
    "recap_query_used",
    "recap_result_id",
    "recap_case_name",
    "recap_court",
    "recap_date_filed",
    "recap_docket_number",
    "recap_suit_nature",
    "recap_assigned_to",
    "recap_referred_to",
    "recap_absolute_url",
    "recap_more_docs",
    "recap_nested_docs_count",
    "recap_doc_descriptions",

    # CourtListener opinions
    "opinion_found",
    "opinion_query_used",
    "opinion_result_id",
    "opinion_case_name",
    "opinion_court",
    "opinion_date_filed",
    "opinion_citation",
    "opinion_absolute_url",
    "opinion_text_chars",

    # Lexis metadata
    "publication_location",
    "publication_type",
    "number",
    "claim_types_clean",
    "target_companies_clean",
    "final_relevant",
    "manual_notes",

    # final case structure
    "true_ai_defendants_final",
    "true_ai_defendants_lexis_title",
    "possible_ai_mentions_lexis_text",
    "plaintiff_type_lexis",
    "procedural_stage_final",
    "outcome_detected_final",
    "motion_outcome_specific_final",
    "data_source_for_text",

    # text diagnostics
    "lexis_text_chars",
    "case_detail_text_chars",
    "case_detail_text_words",

    # summary and full text
    "case_summary_final",
    "case_detail_text",
    "lexis_case_text",
    "opinion_text",
]

for label in DOCTRINE_TERMS:
    preferred_cols.extend([
        label + "_flag_final",
        label + "_count_final",
    ])

preferred_cols.extend(list(FAIR_USE_FACTORS.keys()))

final_cols = [c for c in preferred_cols if c in combined.columns]

final_df = combined[final_cols].copy()

final_df = final_df.drop_duplicates(
    subset=["case_name_final", "docket_number_final", "citation_clean"],
    keep="first",
)

final_df = final_df.sort_values(
    ["year_final", "case_name_final"],
    na_position="last",
).reset_index(drop=True)

final_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved combined enriched dataset ->", OUTPUT_PATH)
print("Final shape:", final_df.shape)

print("\nRECAP match:")
print(final_df["recap_found"].value_counts(dropna=False))

print("\nOpinion match:")
print(final_df["opinion_found"].value_counts(dropna=False))

print("\nText source:")
print(final_df["data_source_for_text"].value_counts(dropna=False))

preview_cols = [
    "case_name_final",
    "docket_number_final",
    "year_final",
    "court_final",
    "recap_found",
    "opinion_found",
    "data_source_for_text",
    "true_ai_defendants_final",
    "procedural_stage_final",
    "outcome_detected_final",
]

preview_cols = [c for c in preview_cols if c in final_df.columns]

print("\nPreview:")
print(final_df[preview_cols].head(30).to_string(index=False))