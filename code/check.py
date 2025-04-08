import pandas as pd

# Load data (Corrected file assignments)
train_df = pd.read_parquet("data/processed/train_processed_02.parquet")
test_df = pd.read_parquet("data/processed/test_processed_02.parquet")

# Show label distributions
print("📊 Training Label Distribution:\n", train_df["Label"].value_counts(), "\n")
print("📊 Testing Label Distribution:\n", test_df["Label"].value_counts())
