import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load SMOTE-applied data
df = pd.read_parquet("data/processed/train_smote.parquet")

# Display label counts
label_counts = df["Label"].value_counts().sort_index()
print("📊 Label distribution after SMOTE:\n")
print(label_counts)

# Optional: Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, palette="viridis")
plt.xlabel("Label")
plt.ylabel("Number of Samples")
plt.title("Label Distribution After SMOTE")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("label_distribution_after_smote.png")
plt.show()
