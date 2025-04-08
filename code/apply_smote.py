import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os

# Paths
train_path = "data/processed/train_processed_02.parquet"
output_path = "data/processed/train_smote.parquet"

# Load training data
df = pd.read_parquet(train_path)

# Separate features and label
X = df.drop(columns=["Label", "_source_file"], errors="ignore")

y = df["Label"]

# Apply SMOTE
print("📈 Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine into a new DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["Label"] = y_resampled

# Save the resampled dataset
df_resampled.to_parquet(output_path, index=False)
print(f"✅ SMOTE-applied dataset saved to: {output_path}")
