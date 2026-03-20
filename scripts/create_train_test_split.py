import pandas as pd

df = pd.read_csv("data/processed/t06_dataset_with_labels.csv")

# TRAIN: only normal data
train_df = df[df["failure"] == 0].copy()

# TEST: full dataset
test_df = df.copy()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Anomalies in test:", test_df["failure"].sum())

train_df.to_csv("data/processed/t06_train.csv", index=False)
test_df.to_csv("data/processed/t06_test.csv", index=False)

print("Saved train/test splits")