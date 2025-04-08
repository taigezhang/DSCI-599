import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import os

# === Load Data ===
data_path = "artifacts/train_raw.parquet"
train_df = pd.read_parquet(data_path)

# === Drop non-numeric or irrelevant columns ===
columns_to_drop = ['Label', 'Timestamp', 'Source IP', 'Destination IP', '_source_file']
X_raw = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], errors='ignore')

# === Encode label ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df["Label"])

# === Impute missing values ===
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

# === Train-test split ===
X_train, _, y_train, _ = train_test_split(X_imputed, y, stratify=y, test_size=0.2, random_state=42)

# === Train Random Forest ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
importance_series = pd.Series(importances, index=X_imputed.columns).sort_values(ascending=False)

# === Filter by importance threshold ===
importance_threshold = 0.005
important_features = importance_series[importance_series > importance_threshold].index.tolist()
print(f"🔎 Found {len(important_features)} features with importance > {importance_threshold}")

# === Compute correlation matrix ===
corr_matrix = X_imputed[important_features].corr().abs()

# === Select features with low correlation ===
selected_features = []
corr_threshold = 0.9

for col in important_features:
    if len(selected_features) == 0:
        selected_features.append(col)
    else:
        corr_values = corr_matrix.loc[col, selected_features]
        if all(corr_values < corr_threshold):
            selected_features.append(col)

print(f"✅ Selected {len(selected_features)} features after correlation filtering.")

# === Plot Final Importance ===
plt.figure(figsize=(10, 8))
sns.barplot(x=importance_series[selected_features], y=selected_features)
plt.title("Top Feature Importances (Filtered by Correlation & >0.005)")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.tight_layout()
os.makedirs("artifacts", exist_ok=True)
plt.savefig("artifacts/feature_selection_filtered_0.005.png")
plt.show()

# === Save correlation heatmap ===
plt.figure(figsize=(12, 10))
sns.heatmap(X_imputed[selected_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Selected Features")
plt.tight_layout()
plt.savefig("artifacts/feature_correlation_heatmap_0.005.png")
plt.show()

# === Save selected features list ===
os.makedirs("data", exist_ok=True)
pd.Series(selected_features).to_csv("data/selected_features_0.005.csv", index=False)
print("📎 Saved to 'data/selected_features_0.005.csv'")
