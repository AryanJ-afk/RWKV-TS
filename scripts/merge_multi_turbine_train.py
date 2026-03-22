import argparse
from pathlib import Path
import shutil
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge multiple turbine train.npy files into one multi-turbine train set")
    parser.add_argument(
        "--train_files",
        type=str,
        nargs="+",
        required=True,
        help="List of train.npy files to merge"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Test npy file to copy into output folder (e.g. T06 test.npy)"
    )
    parser.add_argument(
        "--test_labels_file",
        type=str,
        required=True,
        help="Test labels npy file to copy into output folder (e.g. T06 test_labels.npy)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output folder where merged train.npy, test.npy, test_labels.npy will be saved"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_arrays = []
    expected_features = None

    print("Loading training files...")
    for train_file in args.train_files:
        train_path = Path(train_file)
        arr = np.load(train_path)

        if arr.ndim != 2:
            raise ValueError(f"{train_path} must be 2D, got shape {arr.shape}")

        print(f"{train_path}: shape={arr.shape}")

        if expected_features is None:
            expected_features = arr.shape[1]
        elif arr.shape[1] != expected_features:
            raise ValueError(
                f"Feature count mismatch: {train_path} has {arr.shape[1]} features, "
                f"expected {expected_features}"
            )

        train_arrays.append(arr)

    merged_train = np.concatenate(train_arrays, axis=0)
    print(f"Merged train shape: {merged_train.shape}")

    train_out = output_dir / "train.npy"
    np.save(train_out, merged_train)
    print(f"Saved merged train: {train_out}")

    # Copy test + labels from target turbine
    test_src = Path(args.test_file)
    labels_src = Path(args.test_labels_file)

    test_out = output_dir / "test.npy"
    labels_out = output_dir / "test_labels.npy"

    shutil.copy2(test_src, test_out)
    shutil.copy2(labels_src, labels_out)

    print(f"Copied test file: {test_src} -> {test_out}")
    print(f"Copied labels file: {labels_src} -> {labels_out}")

    # quick sanity check
    test_arr = np.load(test_out)
    labels_arr = np.load(labels_out)

    print(f"Test shape: {test_arr.shape}")
    print(f"Test labels shape: {labels_arr.shape}")

    if test_arr.ndim != 2:
        raise ValueError(f"test.npy must be 2D, got shape {test_arr.shape}")
    if labels_arr.ndim != 1:
        raise ValueError(f"test_labels.npy must be 1D, got shape {labels_arr.shape}")

    if test_arr.shape[1] != expected_features:
        raise ValueError(
            f"Test feature count mismatch: test.npy has {test_arr.shape[1]} features, "
            f"but merged train has {expected_features}"
        )

    if test_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            f"Row count mismatch: test.npy has {test_arr.shape[0]} rows, "
            f"but test_labels.npy has {labels_arr.shape[0]} labels"
        )

    print("\nAll checks passed.")
    print(f"Final dataset ready at: {output_dir}")


if __name__ == "__main__":
    main()