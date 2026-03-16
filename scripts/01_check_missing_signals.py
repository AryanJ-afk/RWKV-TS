import pandas as pd

file_path = "data/raw/Wind-Turbine-SCADA-signals-2016.xlsx"

df = pd.read_excel(file_path)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

missing = df.isna().sum()
missing_only = missing[missing > 0]

print("\nColumns with missing values:")
if len(missing_only) == 0:
    print("No missing values found.")
else:
    print(missing_only)

print("\nPercentage missing per column:")
if len(missing_only) == 0:
    print("No missing values found.")
else:
    print((missing_only / len(df)) * 100)