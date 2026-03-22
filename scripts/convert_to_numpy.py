import pandas as pd
import numpy as np

train_df = pd.read_csv("data/processed/T11_train_id.csv")
test_df = pd.read_csv("data/processed/T11_test_id.csv")

# drop timestamp
train_data = train_df.drop(columns=["Timestamp", "failure"]).values
test_data = test_df.drop(columns=["Timestamp", "failure"]).values

# labels only for test
test_labels = test_df["failure"].values

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
print("Labels shape:", test_labels.shape)

np.save("data/processed/T11/train.npy", train_data)
np.save("data/processed/T11/test.npy", test_data)
np.save("data/processed/T11/test_labels.npy", test_labels)

print("Saved numpy files")