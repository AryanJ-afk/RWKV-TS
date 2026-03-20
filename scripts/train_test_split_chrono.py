import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="data/processed/t06_dataset_with_labels.csv")
parser.add_argument("--padding_hours", type=int, default=24,
                    help="Normal padding before first anomaly to include in test")
parser.add_argument("--train_out", type=str, default="data/processed/t06_train_chrono.csv")
parser.add_argument("--test_out", type=str, default="data/processed/t06_test_chrono.csv")
args = parser.parse_args()

df = pd.read_csv(args.input_file)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

anomaly_rows = df[df["failure"] == 1].copy()

if anomaly_rows.empty:
    raise ValueError("No anomaly rows found in dataset.")

first_anomaly_time = anomaly_rows["Timestamp"].min()
split_time = first_anomaly_time - pd.Timedelta(hours=args.padding_hours)

train_df = df[df["Timestamp"] < split_time].copy()
test_df = df[df["Timestamp"] >= split_time].copy()

print("First anomaly time:", first_anomaly_time)
print("Split time:", split_time)
print("Padding hours:", args.padding_hours)

print("\nTrain shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Anomalies in train:", train_df["failure"].sum())
print("Anomalies in test:", test_df["failure"].sum())

print("\nTrain start:", train_df["Timestamp"].min())
print("Train end:", train_df["Timestamp"].max())
print("Test start:", test_df["Timestamp"].min())
print("Test end:", test_df["Timestamp"].max())

train_df.to_csv(args.train_out, index=False)
test_df.to_csv(args.test_out, index=False)

print("\nSaved:")
print(args.train_out)
print(args.test_out)