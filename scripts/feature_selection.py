import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Basic feature selection for wind turbine anomaly detection")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/processed/t06_dataset_with_labels.csv",
        help="Input CSV with Timestamp, features, failure"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/t06_dataset_with_labels_selected.csv",
        help="Output CSV after feature selection"
    )
    parser.add_argument(
        "--feature_list_file",
        type=str,
        default="data/processed/selected_features.txt",
        help="Output txt file containing selected features"
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="data/processed/feature_selection_summary.txt",
        help="Summary log file"
    )
    parser.add_argument(
        "--std_threshold",
        type=float,
        default=0.01,
        help="Drop features with std strictly below this threshold"
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.95,
        help="Drop one feature from pairs with abs(corr) strictly above this threshold"
    )
    parser.add_argument(
        "--min_features",
        type=int,
        default=10,
        help="Minimum number of features to keep after filtering"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    feature_list_path = Path(args.feature_list_file)
    summary_path = Path(args.summary_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {"Timestamp", "failure"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    feature_cols = [c for c in df.columns if c not in ["Timestamp", "failure"]]

    print(f"Initial feature count: {len(feature_cols)}")

    # -------------------------
    # 1. Low-std filtering
    # -------------------------
    feature_stds = df[feature_cols].std(numeric_only=True)
    kept_after_std = feature_stds[feature_stds >= args.std_threshold].index.tolist()
    dropped_low_std = sorted(set(feature_cols) - set(kept_after_std))

    print(f"Features kept after std filter: {len(kept_after_std)}")
    print(f"Features dropped by std filter: {len(dropped_low_std)}")

    if len(kept_after_std) < args.min_features:
        raise ValueError(
            f"Only {len(kept_after_std)} features remain after std filtering, "
            f"which is below min_features={args.min_features}. "
            f"Lower std_threshold or min_features."
        )

    # -------------------------
    # 2. Correlation filtering
    # -------------------------
    corr_df = df[kept_after_std].corr().abs()

    to_drop_corr = set()
    kept_order = []

    # Greedy keep-first strategy
    for col in kept_after_std:
        if col in to_drop_corr:
            continue
        kept_order.append(col)
        highly_corr = corr_df.index[(corr_df[col] > args.corr_threshold) & (corr_df.index != col)].tolist()
        for other in highly_corr:
            if other not in kept_order:
                to_drop_corr.add(other)

    final_features = [c for c in kept_order if c not in to_drop_corr]
    dropped_corr = [c for c in kept_after_std if c not in final_features]

    print(f"Features kept after corr filter: {len(final_features)}")
    print(f"Features dropped by corr filter: {len(dropped_corr)}")

    if len(final_features) < args.min_features:
        raise ValueError(
            f"Only {len(final_features)} features remain after correlation filtering, "
            f"which is below min_features={args.min_features}. "
            f"Raise corr_threshold or lower min_features."
        )

    # -------------------------
    # 3. Save outputs
    # -------------------------
    final_df = df[["Timestamp"] + final_features + ["failure"]].copy()
    final_df.to_csv(output_path, index=False)

    with open(feature_list_path, "w") as f:
        for col in final_features:
            f.write(col + "\n")

    with open(summary_path, "w") as f:
        f.write(f"Input file: {input_path}\n")
        f.write(f"Output file: {output_path}\n")
        f.write(f"Std threshold: {args.std_threshold}\n")
        f.write(f"Correlation threshold: {args.corr_threshold}\n")
        f.write(f"Min features: {args.min_features}\n\n")

        f.write(f"Initial feature count: {len(feature_cols)}\n")
        f.write(f"After std filter: {len(kept_after_std)}\n")
        f.write(f"After corr filter: {len(final_features)}\n\n")

        f.write("Dropped by std filter:\n")
        for col in dropped_low_std:
            f.write(f"{col}\n")

        f.write("\nDropped by correlation filter:\n")
        for col in dropped_corr:
            f.write(f"{col}\n")

        f.write("\nFinal selected features:\n")
        for col in final_features:
            f.write(f"{col}\n")

    print(f"\nSaved filtered dataset: {output_path}")
    print(f"Saved feature list: {feature_list_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Final feature count: {len(final_features)}")


if __name__ == "__main__":
    main()