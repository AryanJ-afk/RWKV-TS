import pandas as pd
import numpy as np

train_df = pd.read_csv("data/processed/t06_train_chrono.csv")
test_df = pd.read_csv("data/processed/t06_test_chrono.csv")

# drop timestamp
train_data = train_df.drop(columns=["Timestamp", "failure"]).values
test_data = test_df.drop(columns=["Timestamp", "failure"]).values

# labels only for test
test_labels = test_df["failure"].values

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
print("Labels shape:", test_labels.shape)

np.save("data/processed/train.npy", train_data)
np.save("data/processed/test.npy", test_data)
np.save("data/processed/test_labels.npy", test_labels)

print("Saved numpy files")