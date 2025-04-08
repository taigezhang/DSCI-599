import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the original training dataset
df = pd.read_parquet("artifacts/train_raw.parquet")

# Fit label encoder
le = LabelEncoder()
le.fit(df["Label"])

# Print label → class mappings
print("🔍 Label Number to Attack Name Mapping:\n")
for idx, name in enumerate(le.classes_):
    print(f"{idx}: {name}")
