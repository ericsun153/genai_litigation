import pandas as pd
import os

IN_PATH = "data_clean/06_auto_validation_sheet.xlsx"
OUT_PATH = "data_clean/07_final_case_dataset.xlsx"

os.makedirs("data_clean", exist_ok=True)

df = pd.read_excel(IN_PATH)

# ---------------------------------------------------
# Use manual label if provided; otherwise fallback auto
# ---------------------------------------------------

def parse_label(x):
    x = str(x).strip().lower()

    if x in ["1", "yes", "true"]:
        return 1
    if x in ["0", "no", "false"]:
        return 0
    if x in ["maybe", "uncertain", ""]:
        return None

    return None


def final_label(row):
    manual = parse_label(row.get("manual_relevant", ""))
    auto = parse_label(row.get("auto_relevant", ""))

    if manual is not None:
        return manual

    if auto is not None:
        return auto

    return 0


df["final_relevant"] = df.apply(final_label, axis=1)

# ---------------------------------------------------
# Keep final relevant rows only
# ---------------------------------------------------

df_final = df[df["final_relevant"] == 1].copy()

# ---------------------------------------------------
# Optional: deduplicate again
# ---------------------------------------------------

if "manual_case_group" in df_final.columns:
    df_final["manual_case_group"] = (
        df_final["manual_case_group"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    grouped = df_final["manual_case_group"] != ""

    df_grouped = (
        df_final[grouped]
        .sort_values("relevance_score", ascending=False)
        .drop_duplicates(subset=["manual_case_group"], keep="first")
    )

    df_ungrouped = df_final[~grouped]

    df_final = pd.concat([df_grouped, df_ungrouped], ignore_index=True)

elif "case_name_norm" in df_final.columns:
    df_final = (
        df_final
        .sort_values("relevance_score", ascending=False)
        .drop_duplicates(subset=["case_name_norm"], keep="first")
    )

# ---------------------------------------------------
# Save outputs
# ---------------------------------------------------

df_final.to_excel(OUT_PATH, index=False)

summary = pd.DataFrame({
    "metric": [
        "total_input_rows",
        "final_relevant_rows"
    ],
    "value": [
        len(df),
        len(df_final)
    ]
})

summary.to_csv("data_clean/final_dataset_summary.csv", index=False)

print("Input rows:", len(df))
print("Final relevant cases:", len(df_final))
print("Saved:", OUT_PATH)
print("Saved: data_clean/final_dataset_summary.csv")