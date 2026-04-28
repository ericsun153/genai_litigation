# ==========================================================
# FILE: 03_nlp_analysis.py
#
# Input:
#   data_nlp/enriched_copyright_cases.csv
#
# Outputs:
#   data_nlp/final_ai_copyright_nlp_output.csv
#   data_nlp/ai_copyright_case_summary_for_paper.csv
#   figures/*.png
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


INPUT_PATH = "data_nlp/enriched_copyright_cases.csv"
OUTPUT_PATH = "data_nlp/final_ai_copyright_nlp_output.csv"
SUMMARY_OUTPUT_PATH = "data_nlp/ai_copyright_case_summary_for_paper.csv"
FIG_DIR = "figures"

os.makedirs("data_nlp", exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x)


def contains_any(text, terms):
    text = safe_text(text).lower()
    return any(t.lower() in text for t in terms)


def count_terms(text, terms):
    text = safe_text(text).lower()
    return sum(text.count(t.lower()) for t in terms)


def normalize_score(series):
    series = series.fillna(0)
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())


def safe_bar_plot(series, title, xlabel, ylabel, path, figsize=(11, 5), rotation=30):
    if series is None or len(series) == 0:
        print(f"Skip empty plot: {title}")
        return

    plt.figure(figsize=figsize)
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(INPUT_PATH)

df["case_detail_text"] = df["case_detail_text"].fillna("")
df["case_name_final"] = df["case_name_final"].fillna("")
df["true_ai_defendants_final"] = df["true_ai_defendants_final"].fillna("")

print("Loaded enriched dataset:", df.shape)


# ==========================================================
# 1. FINAL CORE FILTER
# ==========================================================

CORE_AI_TERMS = [
    "openai", "chatgpt",
    "anthropic", "claude",
    "stability ai", "stable diffusion",
    "midjourney",
    "meta", "llama",
    "microsoft", "copilot",
    "google", "gemini",
    "ross intelligence",
    "quizlet",
    "artificial intelligence",
    "generative ai",
    "large language model",
    "llm",
    "machine learning",
    "training data",
    "model training",
]

COPYRIGHT_TERMS = [
    "copyright",
    "copyright infringement",
    "fair use",
    "dmca",
    "derivative work",
    "substantial similarity",
    "training data",
    "scraping",
    "licensing market",
    "market harm",
]

df["nlp_core_ai_signal"] = (
    df["case_name_final"].apply(lambda x: contains_any(x, CORE_AI_TERMS))
    | df["case_detail_text"].apply(lambda x: contains_any(x, CORE_AI_TERMS))
    | df["true_ai_defendants_final"].str.len().gt(0)
)

df["nlp_core_copyright_signal"] = (
    df["case_name_final"].apply(lambda x: contains_any(x, COPYRIGHT_TERMS))
    | df["case_detail_text"].apply(lambda x: contains_any(x, COPYRIGHT_TERMS))
)

core_df = df[df["nlp_core_ai_signal"] & df["nlp_core_copyright_signal"]].copy()

print("Core cases for NLP:", core_df.shape)

if len(core_df) == 0:
    raise ValueError("No cases left after NLP filter.")


# ==========================================================
# 2. DOCTRINE SCORING
# ==========================================================

DOCTRINE_TERMS = {
    "fair_use": [
        "fair use", "transformative", "section 107", "17 u.s.c. 107",
        "purpose and character", "market effect"
    ],
    "training_data": [
        "training data", "trained on", "model training", "training corpus",
        "dataset", "corpus"
    ],
    "scraping": [
        "scraping", "scraped", "web scraping", "crawl", "crawled"
    ],
    "market_harm": [
        "market harm", "market effect", "licensing market",
        "lost licensing", "substitution", "potential market", "lost sales"
    ],
    "memorization_output": [
        "memorization", "regurgitation", "verbatim",
        "output", "outputs", "substantially similar output"
    ],
    "dmca_cmi": [
        "dmca", "1202", "copyright management information", "cmi"
    ],
    "derivative_work": [
        "derivative work", "derivative works", "adaptation"
    ],
    "substantial_similarity": [
        "substantial similarity", "substantially similar"
    ],
    "injunction": [
        "injunction", "preliminary injunction", "permanent injunction", "enjoin"
    ],
    "damages": [
        "damages", "statutory damages", "actual damages", "profits", "disgorgement"
    ],
}

for label, terms in DOCTRINE_TERMS.items():
    core_df[label + "_nlp_score"] = core_df["case_detail_text"].apply(
        lambda x: count_terms(x, terms)
    )
    core_df[label + "_nlp_flag"] = core_df[label + "_nlp_score"] > 0


# ==========================================================
# 3. FAIR USE FOUR FACTORS
# ==========================================================

FAIR_USE_FACTORS = {
    "factor1_purpose_transformative": [
        "purpose and character", "transformative",
        "commercial use", "nonprofit", "good faith", "bad faith"
    ],
    "factor2_nature_work": [
        "nature of the copyrighted work",
        "creative work", "factual work", "published", "unpublished"
    ],
    "factor3_amount_taken": [
        "amount and substantiality", "entire work",
        "heart of the work", "quantitatively", "qualitatively"
    ],
    "factor4_market_effect": [
        "effect upon the potential market",
        "market effect", "market harm", "licensing market", "substitution"
    ],
}

for label, terms in FAIR_USE_FACTORS.items():
    core_df[label + "_nlp_score"] = core_df["case_detail_text"].apply(
        lambda x: count_terms(x, terms)
    )


# ==========================================================
# 4. INDEX SCORES
# ==========================================================

core_df["training_data_conflict_index"] = (
    normalize_score(core_df["training_data_nlp_score"])
    + normalize_score(core_df["scraping_nlp_score"])
    + normalize_score(core_df["market_harm_nlp_score"])
)

core_df["fair_use_salience_index"] = (
    normalize_score(core_df["fair_use_nlp_score"])
    + normalize_score(core_df["factor1_purpose_transformative_nlp_score"])
    + normalize_score(core_df["factor4_market_effect_nlp_score"])
)

core_df["output_risk_index"] = (
    normalize_score(core_df["memorization_output_nlp_score"])
    + normalize_score(core_df["substantial_similarity_nlp_score"])
)

core_df["remedy_pressure_index"] = (
    normalize_score(core_df["injunction_nlp_score"])
    + normalize_score(core_df["damages_nlp_score"])
)


# ==========================================================
# 5. CASE THEME CLASSIFICATION
# ==========================================================

def classify_case_theme(row):
    text = safe_text(row["case_detail_text"]).lower()
    name = safe_text(row["case_name_final"]).lower()
    combined = name + " " + text

    if any(t in combined for t in ["training data", "model training", "trained on", "training corpus", "corpus"]):
        return "Training Data / Model Training"

    if any(t in combined for t in ["memorization", "regurgitation", "verbatim", "substantially similar output"]):
        return "Output Similarity / Memorization"

    if any(t in combined for t in ["dmca", "1202", "copyright management information"]):
        return "DMCA / CMI"

    if any(t in combined for t in ["fair use", "transformative"]):
        return "Fair Use / Transformative Use"

    if any(t in combined for t in ["registration", "human authorship", "copyright office"]):
        return "AI Authorship / Registration"

    return "Other AI Copyright"


core_df["case_theme"] = core_df.apply(classify_case_theme, axis=1)


# ==========================================================
# 6. TF-IDF + NMF TOPIC MODELING
# ==========================================================

min_df_value = 2 if len(core_df) >= 10 else 1

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=min_df_value,
    max_df=0.85,
)

tfidf = vectorizer.fit_transform(core_df["case_detail_text"])
feature_names = vectorizer.get_feature_names_out()

N_TOPICS = min(6, max(2, len(core_df)))

nmf = NMF(
    n_components=N_TOPICS,
    random_state=42,
    init="nndsvda",
    max_iter=800,
)

topic_matrix = nmf.fit_transform(tfidf)

raw_topic_labels = {}

print("\n========== NMF TOPICS ==========")
for topic_idx, topic in enumerate(nmf.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-15:][::-1]]
    raw_topic_labels[topic_idx] = ", ".join(top_words[:8])
    print(f"Topic {topic_idx}: {', '.join(top_words)}")


def name_topic_from_keywords(keyword_string):
    text = keyword_string.lower()

    if any(w in text for w in ["fair use", "transformative", "section 107", "market effect"]):
        return "Fair Use / Transformative Use"

    if any(w in text for w in ["training", "dataset", "corpus", "scraping", "scraped"]):
        return "Training Data / Scraping"

    if any(w in text for w in ["dmca", "1202", "cmi", "copyright management"]):
        return "DMCA / Copyright Management Info"

    if any(w in text for w in ["output", "memorization", "verbatim", "similarity"]):
        return "Output Similarity / Memorization"

    if any(w in text for w in ["registration", "human authorship", "copyright office"]):
        return "AI Authorship / Copyright Registration"

    if any(w in text for w in ["injunction", "damages", "dismiss", "motion"]):
        return "Procedure / Remedies"

    return "Other Copyright Litigation"


topic_named_labels = {
    topic_id: name_topic_from_keywords(keywords)
    for topic_id, keywords in raw_topic_labels.items()
}

core_df["dominant_topic"] = topic_matrix.argmax(axis=1)
core_df["dominant_topic_score"] = topic_matrix.max(axis=1)
core_df["topic_keywords"] = core_df["dominant_topic"].map(raw_topic_labels)
core_df["topic_name"] = core_df["dominant_topic"].map(topic_named_labels)


# ==========================================================
# 7. PER-CASE KEYWORDS
# ==========================================================

def top_keywords_for_doc(row_index, top_n=12):
    row = tfidf[row_index].toarray().flatten()
    top_idx = row.argsort()[-top_n:][::-1]
    return "; ".join(feature_names[top_idx])


core_df["top_tfidf_keywords"] = [
    top_keywords_for_doc(i) for i in range(tfidf.shape[0])
]


# ==========================================================
# 8. CLUSTERING
# ==========================================================

N_CLUSTERS = min(5, max(2, len(core_df)))

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=20,
)

core_df["text_cluster"] = kmeans.fit_predict(tfidf)

cluster_labels = {}

print("\n========== TEXT CLUSTERS ==========")
for cluster_id in range(N_CLUSTERS):
    centroid = kmeans.cluster_centers_[cluster_id]
    top_words = [feature_names[i] for i in centroid.argsort()[-12:][::-1]]
    cluster_labels[cluster_id] = ", ".join(top_words[:8])
    print(f"Cluster {cluster_id}: {', '.join(top_words)}")

core_df["cluster_keywords"] = core_df["text_cluster"].map(cluster_labels)


# ==========================================================
# 9. CASE SIMILARITY
# ==========================================================

similarity = cosine_similarity(tfidf)

nearest_cases = []
case_names = core_df["case_name_final"].tolist()

for i in range(len(core_df)):
    sims = similarity[i].copy()
    sims[i] = -1
    nearest_idx = int(np.argmax(sims))

    nearest_cases.append({
        "nearest_case_name": case_names[nearest_idx],
        "nearest_case_similarity": float(sims[nearest_idx]),
    })

nearest_df = pd.DataFrame(nearest_cases).reset_index(drop=True)
core_df = pd.concat([core_df.reset_index(drop=True), nearest_df], axis=1)


# ==========================================================
# 10. FIGURES
# ==========================================================

if "year_final" in core_df.columns:
    years = pd.to_numeric(core_df["year_final"], errors="coerce")
    year_counts = years.dropna().astype(int).value_counts().sort_index()

    safe_bar_plot(
        year_counts,
        "AI Copyright / Training Data Cases Over Time",
        "Year",
        "Number of Cases",
        f"{FIG_DIR}/01_cases_over_time.png",
        rotation=0,
    )

defendant_counts = (
    core_df["true_ai_defendants_final"]
    .replace("", np.nan)
    .dropna()
    .str.split("; ")
    .explode()
    .value_counts()
)

safe_bar_plot(
    defendant_counts,
    "AI Defendants in Copyright Cases",
    "Defendant",
    "Number of Cases",
    f"{FIG_DIR}/02_ai_defendants.png",
)

if "plaintiff_type_lexis" in core_df.columns:
    safe_bar_plot(
        core_df["plaintiff_type_lexis"].value_counts(),
        "Plaintiff Types in AI Copyright Cases",
        "Plaintiff Type",
        "Number of Cases",
        f"{FIG_DIR}/03_plaintiff_types.png",
    )

doctrine_score_cols = [
    c for c in core_df.columns
    if c.endswith("_nlp_score") and not c.startswith("factor")
]

doctrine_totals = core_df[doctrine_score_cols].sum().sort_values(ascending=False)

safe_bar_plot(
    doctrine_totals,
    "Copyright Doctrine / Issue Mentions",
    "Doctrine / Issue",
    "Total Mentions",
    f"{FIG_DIR}/04_doctrine_mentions.png",
    figsize=(13, 5),
)

factor_cols = [
    c for c in core_df.columns
    if c.startswith("factor") and c.endswith("_nlp_score")
]

factor_totals = core_df[factor_cols].sum().sort_values(ascending=False)

safe_bar_plot(
    factor_totals,
    "Fair Use Four-Factor Emphasis",
    "Fair Use Factor",
    "Total Mentions",
    f"{FIG_DIR}/05_fair_use_factor_emphasis.png",
    figsize=(12, 5),
)

safe_bar_plot(
    core_df["case_theme"].value_counts(),
    "Case Theme Distribution",
    "Case Theme",
    "Number of Cases",
    f"{FIG_DIR}/06_case_theme_distribution.png",
)

safe_bar_plot(
    core_df["topic_name"].value_counts(),
    "NLP Topic Distribution",
    "Topic Theme",
    "Number of Cases",
    f"{FIG_DIR}/07_topic_distribution_named.png",
    figsize=(12, 5),
)

if "data_source_for_text" in core_df.columns:
    safe_bar_plot(
        core_df["data_source_for_text"].value_counts(),
        "Text Source Distribution",
        "Text Source",
        "Number of Cases",
        f"{FIG_DIR}/08_text_source_distribution.png",
    )

index_cols = [
    "training_data_conflict_index",
    "fair_use_salience_index",
    "output_risk_index",
    "remedy_pressure_index",
]

index_totals = core_df[index_cols].sum().sort_values(ascending=False)

safe_bar_plot(
    index_totals,
    "Aggregate NLP Research Index Scores",
    "Index",
    "Aggregate Score",
    f"{FIG_DIR}/09_research_index_scores.png",
    figsize=(11, 5),
)


# ==========================================================
# 11. EXPORT
# ==========================================================

core_df.to_csv(OUTPUT_PATH, index=False)

summary_cols = [
    "case_name_final",
    "year_final",
    "court_final",
    "judge_final",
    "docket_number_final",
    "recap_found",
    "opinion_found",
    "data_source_for_text",
    "true_ai_defendants_final",
    "plaintiff_type_lexis",
    "case_theme",
    "procedural_stage_final",
    "outcome_detected_final",
    "motion_outcome_specific_final",
    "dominant_topic",
    "topic_name",
    "topic_keywords",
    "text_cluster",
    "cluster_keywords",
    "top_tfidf_keywords",
    "nearest_case_name",
    "nearest_case_similarity",
    "fair_use_nlp_score",
    "training_data_nlp_score",
    "scraping_nlp_score",
    "market_harm_nlp_score",
    "memorization_output_nlp_score",
    "dmca_cmi_nlp_score",
    "training_data_conflict_index",
    "fair_use_salience_index",
    "output_risk_index",
    "remedy_pressure_index",
    "case_summary_final",
]

summary_cols = [c for c in summary_cols if c in core_df.columns]

core_df[summary_cols].to_csv(SUMMARY_OUTPUT_PATH, index=False)

# ==========================================================
# 12. SAVE TOPICS AND CLUSTERS AS OUTPUT FILES
# ==========================================================

# ----- NMF topic table -----
topic_output_rows = []

for topic_idx, topic in enumerate(nmf.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-20:][::-1]]

    topic_output_rows.append({
        "topic_id": topic_idx,
        "topic_name": topic_named_labels.get(topic_idx, ""),
        "top_keywords": ", ".join(top_words),
        "case_count": int((core_df["dominant_topic"] == topic_idx).sum())
    })

topic_output_df = pd.DataFrame(topic_output_rows)
topic_output_path = "data_nlp/nmf_topic_keywords.csv"
topic_output_df.to_csv(topic_output_path, index=False)


# ----- Cluster table -----
cluster_output_rows = []

for cluster_id in range(N_CLUSTERS):
    centroid = kmeans.cluster_centers_[cluster_id]
    top_words = [feature_names[i] for i in centroid.argsort()[-20:][::-1]]

    cluster_output_rows.append({
        "cluster_id": cluster_id,
        "cluster_keywords": ", ".join(top_words),
        "case_count": int((core_df["text_cluster"] == cluster_id).sum())
    })

cluster_output_df = pd.DataFrame(cluster_output_rows)
cluster_output_path = "data_nlp/text_cluster_keywords.csv"
cluster_output_df.to_csv(cluster_output_path, index=False)


# ----- TXT report -----
txt_output_path = "data_nlp/nlp_topics_and_clusters_report.txt"

with open(txt_output_path, "w", encoding="utf-8") as f:
    f.write("========== NMF TOPICS ==========\n")
    for _, row in topic_output_df.iterrows():
        f.write(
            f"Topic {row['topic_id']} | {row['topic_name']} | "
            f"Cases: {row['case_count']}\n"
        )
        f.write(f"{row['top_keywords']}\n\n")

    f.write("\n========== TEXT CLUSTERS ==========\n")
    for _, row in cluster_output_df.iterrows():
        f.write(
            f"Cluster {row['cluster_id']} | "
            f"Cases: {row['case_count']}\n"
        )
        f.write(f"{row['cluster_keywords']}\n\n")


# ----- Save table images -----
def save_dataframe_as_image(table_df, title, output_path, figsize=(14, 6), font_size=9):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.5)

    plt.title(title, fontsize=15, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


topic_img_df = topic_output_df.copy()
topic_img_df["top_keywords"] = topic_img_df["top_keywords"].str.slice(0, 120) + "..."

save_dataframe_as_image(
    topic_img_df,
    "NMF Topic Keywords",
    "figures/13_nmf_topic_keywords_table.png",
    figsize=(16, 6),
    font_size=8
)

cluster_img_df = cluster_output_df.copy()
cluster_img_df["cluster_keywords"] = cluster_img_df["cluster_keywords"].str.slice(0, 130) + "..."

save_dataframe_as_image(
    cluster_img_df,
    "Text Cluster Keywords",
    "figures/14_text_cluster_keywords_table.png",
    figsize=(16, 5),
    font_size=8
)

print("\nSaved topic/cluster outputs:")
print(topic_output_path)
print(cluster_output_path)
print(txt_output_path)
print("figures/13_nmf_topic_keywords_table.png")
print("figures/14_text_cluster_keywords_table.png")

print("\nSaved:")
print(OUTPUT_PATH)
print(SUMMARY_OUTPUT_PATH)
print(FIG_DIR)

print("\nPreview:")
print(core_df[summary_cols].head(12).to_string(index=False))