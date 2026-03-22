import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--feature_list_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    with open(args.feature_list_file, "r") as f:
        selected_features = [line.strip() for line in f if line.strip()]

    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input file: {missing}")

    final_df = df[["Timestamp"] + selected_features + ["failure"]].copy()
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_file, index=False)

    print(f"Saved: {args.output_file}")
    print(f"Feature count: {len(selected_features)}")

if __name__ == "__main__":
    main()