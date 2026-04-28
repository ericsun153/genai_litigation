# ==========================================================
# FILE: 04_summary_table.py
#
# Input:
#   data_nlp/final_ai_copyright_nlp_output.csv
#
# Outputs:
#   figures/10_quantitative_summary_table.png
#   figures/11_topic_distribution_named_table.png
#   figures/12_topic_distribution_named_bar.png
#   data_nlp/quantitative_summary_table.csv
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


INPUT_PATH = "data_nlp/final_ai_copyright_nlp_output.csv"
SUMMARY_CSV_PATH = "data_nlp/quantitative_summary_table.csv"
FIG_DIR = "figures"

os.makedirs("data_nlp", exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def safe_pct(series):
    if series is None or len(series) == 0:
        return 0
    return round(series.fillna(False).astype(bool).mean() * 100, 1)


def safe_mean(df, col):
    if col not in df.columns:
        return np.nan
    return round(pd.to_numeric(df[col], errors="coerce").mean(), 2)


def top_value(df, col):
    if col not in df.columns:
        return "N/A"

    s = df[col].fillna("").replace("", np.nan).dropna()

    if len(s) == 0:
        return "N/A"

    return s.value_counts().idxmax()


def top_split_value(df, col, sep="; "):
    if col not in df.columns:
        return "N/A"

    s = (
        df[col]
        .fillna("")
        .replace("", np.nan)
        .dropna()
        .astype(str)
        .str.split(sep)
        .explode()
        .str.strip()
    )

    s = s[s != ""]

    if len(s) == 0:
        return "N/A"

    return s.value_counts().idxmax()


def make_table_image(table_df, title, output_path, figsize=(13, 8), font_size=10):
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
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.5)

    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(INPUT_PATH)

print("Loaded:", df.shape)


# ==========================================================
# QUANTITATIVE SUMMARY TABLE
# ==========================================================

n_cases = len(df)

years = pd.to_numeric(df.get("year_final", pd.Series(dtype=float)), errors="coerce")
year_min = int(years.min()) if years.notna().any() else "N/A"
year_max = int(years.max()) if years.notna().any() else "N/A"

n_defendants = (
    df.get("true_ai_defendants_final", pd.Series(dtype=str))
    .fillna("")
    .replace("", np.nan)
    .dropna()
    .astype(str)
    .str.split("; ")
    .explode()
    .str.strip()
    .replace("", np.nan)
    .dropna()
    .nunique()
)

n_plaintiff_types = (
    df.get("plaintiff_type_lexis", pd.Series(dtype=str))
    .fillna("")
    .replace("", np.nan)
    .dropna()
    .nunique()
)

summary = pd.DataFrame({
    "Metric": [
        "Total Cases",
        "Years Covered",
        "Unique AI Defendants",
        "Plaintiff Types",
        "RECAP Match Rate (%)",
        "Opinion Match Rate (%)",
        "Avg Fair Use Score",
        "Avg Training Data Score",
        "Avg Scraping Score",
        "Avg Market Harm Score",
        "Avg Memorization / Output Score",
        "Avg DMCA / CMI Score",
        "Avg Training Data Conflict Index",
        "Avg Fair Use Salience Index",
        "Avg Output Risk Index",
        "Avg Remedy Pressure Index",
        "Top Defendant",
        "Top Plaintiff Type",
        "Most Common Theme",
        "Most Common Topic"
    ],
    "Value": [
        n_cases,
        f"{year_min} - {year_max}",
        n_defendants,
        n_plaintiff_types,
        safe_pct(df["recap_found"]) if "recap_found" in df.columns else "N/A",
        safe_pct(df["opinion_found"]) if "opinion_found" in df.columns else "N/A",
        safe_mean(df, "fair_use_nlp_score"),
        safe_mean(df, "training_data_nlp_score"),
        safe_mean(df, "scraping_nlp_score"),
        safe_mean(df, "market_harm_nlp_score"),
        safe_mean(df, "memorization_output_nlp_score"),
        safe_mean(df, "dmca_cmi_nlp_score"),
        safe_mean(df, "training_data_conflict_index"),
        safe_mean(df, "fair_use_salience_index"),
        safe_mean(df, "output_risk_index"),
        safe_mean(df, "remedy_pressure_index"),
        top_split_value(df, "true_ai_defendants_final"),
        top_value(df, "plaintiff_type_lexis"),
        top_value(df, "case_theme"),
        top_value(df, "topic_name"),
    ]
})

summary.to_csv(SUMMARY_CSV_PATH, index=False)

make_table_image(
    summary,
    "Quantitative Summary of AI Copyright Litigation Dataset",
    f"{FIG_DIR}/10_quantitative_summary_table.png",
    figsize=(13, 8),
    font_size=10,
)


# ==========================================================
# TOPIC DISTRIBUTION TABLE IMAGE
# ==========================================================

topic_table = (
    df.groupby("topic_name")
    .agg(
        cases=("case_name_final", "count"),
        avg_fair_use_score=("fair_use_nlp_score", "mean"),
        avg_training_data_score=("training_data_nlp_score", "mean"),
        avg_market_harm_score=("market_harm_nlp_score", "mean"),
        avg_output_risk_index=("output_risk_index", "mean"),
    )
    .reset_index()
)

topic_table = topic_table.sort_values("cases", ascending=False)

for col in [
    "avg_fair_use_score",
    "avg_training_data_score",
    "avg_market_harm_score",
    "avg_output_risk_index",
]:
    topic_table[col] = topic_table[col].round(2)

topic_table_for_img = topic_table.rename(columns={
    "topic_name": "Topic Theme",
    "cases": "Cases",
    "avg_fair_use_score": "Avg Fair Use",
    "avg_training_data_score": "Avg Training Data",
    "avg_market_harm_score": "Avg Market Harm",
    "avg_output_risk_index": "Avg Output Risk",
})

make_table_image(
    topic_table_for_img,
    "NLP Topic Distribution and Quantitative Scores",
    f"{FIG_DIR}/11_topic_distribution_named_table.png",
    figsize=(14, 5.5),
    font_size=9,
)


# ==========================================================
# NAMED TOPIC BAR CHART
# ==========================================================

topic_counts = df["topic_name"].value_counts()

plt.figure(figsize=(12, 6))
topic_counts.plot(kind="bar")
plt.title("NLP Topic Distribution by Named Topic")
plt.xlabel("Topic Theme")
plt.ylabel("Number of Cases")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/12_topic_distribution_named_bar.png", dpi=300)
plt.close()


print("Saved:")
print(SUMMARY_CSV_PATH)
print(f"{FIG_DIR}/10_quantitative_summary_table.png")
print(f"{FIG_DIR}/11_topic_distribution_named_table.png")
print(f"{FIG_DIR}/12_topic_distribution_named_bar.png")