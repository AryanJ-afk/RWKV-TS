import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================
TURBINE_ID = "T06"
WINDOW_HOURS = 48

SIGNALS_FILE = Path("data/raw/Wind-Turbine-SCADA-signals-2016.xlsx")
FAILURES_FILE = Path("data/raw/Failure_2016.xlsx")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_OUTPUT = OUTPUT_DIR / "t06_dataset_with_labels.csv"
IMPUTED_OUTPUT = OUTPUT_DIR / "t06_signals_2016_imputed.csv"  # optional debug artifact

# =========================
# 1. Load and extract turbine
# =========================
print(f"Loading signals from: {SIGNALS_FILE}")
df = pd.read_excel(SIGNALS_FILE)

df = df[df["Turbine_ID"] == TURBINE_ID].copy()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

print(f"Rows for {TURBINE_ID}: {len(df)}")
print(f"Start: {df['Timestamp'].min()}")
print(f"End:   {df['Timestamp'].max()}")

# =========================
# 2. Check and fill missing timestamps
# =========================
start = df["Timestamp"].min()
end = df["Timestamp"].max()

full_range = pd.date_range(start=start, end=end, freq="10min", tz=start.tz)
full_df = pd.DataFrame({"Timestamp": full_range})

df_full = full_df.merge(df, on="Timestamp", how="left")

non_timestamp_cols = [c for c in df_full.columns if c != "Timestamp"]
inserted_rows = df_full[non_timestamp_cols].isna().all(axis=1).sum()

print(f"Expected timestamps: {len(full_range)}")
print(f"Existing timestamps: {len(df)}")
print(f"Inserted missing timestamp rows: {inserted_rows}")

if inserted_rows > 0:
    print("\nFirst 10 inserted timestamps:")
    print(df_full.loc[df_full[non_timestamp_cols].isna().all(axis=1), "Timestamp"].head(10))

# =========================
# 3. Keep Avg + Std features
# =========================
feature_cols = [
    c for c in df_full.columns
    if c.endswith("_Avg") or c.endswith("_Std")
]

df_features = df_full[["Timestamp"] + feature_cols].copy()

print(f"\nFeature count (Avg + Std): {len(feature_cols)}")
print(f"Dataset shape before imputation: {df_features.shape}")
print(f"Missing cells before imputation: {df_features.drop(columns=['Timestamp']).isna().sum().sum()}")
print(f"Rows with missing before imputation: {df_features.drop(columns=['Timestamp']).isna().any(axis=1).sum()}")

# =========================
# 4. Impute by time interpolation
# =========================
df_imputed = df_features.copy()
df_imputed = df_imputed.set_index("Timestamp")
df_imputed = df_imputed.interpolate(method="time")
df_imputed = df_imputed.ffill().bfill()
df_imputed = df_imputed.reset_index()

print(f"\nMissing cells after imputation: {df_imputed.drop(columns=['Timestamp']).isna().sum().sum()}")
print(f"Rows with missing after imputation: {df_imputed.drop(columns=['Timestamp']).isna().any(axis=1).sum()}")

# optional debug save
df_imputed.to_csv(IMPUTED_OUTPUT, index=False)
print(f"Saved imputed dataset: {IMPUTED_OUTPUT}")

# =========================
# 5. Add failure labels using pre-failure windows
# =========================
print(f"\nLoading failures from: {FAILURES_FILE}")
failures = pd.read_excel(FAILURES_FILE)
failures["Timestamp"] = pd.to_datetime(failures["Timestamp"])
failures_t06 = failures[failures["Turbine_ID"] == TURBINE_ID].copy()

df_final = df_imputed.copy()
df_final["failure"] = 0

print(f"{TURBINE_ID} failure records found: {len(failures_t06)}")

for ts in failures_t06["Timestamp"]:
    start = ts - pd.Timedelta(hours=WINDOW_HOURS)
    mask = (df_final["Timestamp"] >= start) & (df_final["Timestamp"] <= ts)
    count = mask.sum()
    df_final.loc[mask, "failure"] = 1
    print(f"Failure window applied: {start} -> {ts} | rows marked: {count}")

print(f"\nTotal anomaly rows: {df_final['failure'].sum()}")

# =========================
# 6. Save final dataset
# =========================
df_final.to_csv(FINAL_OUTPUT, index=False)
print(f"\nSaved final dataset: {FINAL_OUTPUT}")
print(f"Final shape: {df_final.shape}")