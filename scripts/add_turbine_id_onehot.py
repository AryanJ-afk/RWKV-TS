import argparse
from pathlib import Path
import pandas as pd

ALL_TURBINES = ["T01", "T06", "T07", "T11"]

def main():
    parser = argparse.ArgumentParser(description="Add one-hot turbine ID columns to a dataset")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--turbine_id", type=str, required=True)
    args = parser.parse_args()

    if args.turbine_id not in ALL_TURBINES:
        raise ValueError(f"turbine_id must be one of {ALL_TURBINES}")

    df = pd.read_csv(args.input_file)

    # add one-hot columns
    for tid in ALL_TURBINES:
        df[f"is_{tid}"] = 1 if tid == args.turbine_id else 0

    # keep Timestamp first, failure last if present
    cols = df.columns.tolist()

    if "failure" in cols:
        feature_cols = [c for c in cols if c not in ["Timestamp", "failure"] and not c.startswith("is_")]
        id_cols = [f"is_{tid}" for tid in ALL_TURBINES]
        df = df[["Timestamp"] + feature_cols + id_cols + ["failure"]]
    else:
        feature_cols = [c for c in cols if c != "Timestamp" and not c.startswith("is_")]
        id_cols = [f"is_{tid}" for tid in ALL_TURBINES]
        df = df[["Timestamp"] + feature_cols + id_cols]

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False)

    print(f"Saved: {args.output_file}")
    print("Added columns:", [f"is_{tid}" for tid in ALL_TURBINES])
    print("Final shape:", df.shape)

if __name__ == "__main__":
    main()