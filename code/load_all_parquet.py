import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 🔧 Directory paths
base_path = "data"
output_path = "artifacts"
os.makedirs(output_path, exist_ok=True)

# ✅ Label merging dictionary
label_mapping = {
    "DrDoS_NetBIOS": "NetBIOS",
    "DrDoS_MSSQL": "MSSQL",
    "DrDoS_UDP": "UDP",
    "DrDoS_DNS": "DNS",
    "DrDoS_SNMP": "SNMP",
    "DrDoS_LDAP": "LDAP",
    "UDP-lag": "UDPLag"
}

def load_all_parquet(data_dir):
    dfs = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Directory does not exist: {data_dir}")

    for file in os.listdir(data_dir):
        if file.endswith(".parquet") and ("training" in file or "testing" in file):
            df = pd.read_parquet(os.path.join(data_dir, file))
            df["_source_file"] = file
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

# 📦 Load and combine data
print("📦 Loading raw data from all .parquet files...")
df = load_all_parquet(base_path)
print(f"🔍 Total records loaded: {len(df):,}")

# ✅ Drop rows without labels
df.dropna(subset=["Label"], inplace=True)

# ✅ Apply label merging
df["Label"] = df["Label"].replace(label_mapping)

# ✅ Stratified split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Label"],
    random_state=42
)

# 💾 Save the outputs
train_df.to_parquet(os.path.join(output_path, "train_raw.parquet"), index=False)
test_df.to_parquet(os.path.join(output_path, "test_raw.parquet"), index=False)

train_df.to_csv(os.path.join(output_path, "train_raw.csv"), index=False)
test_df.to_csv(os.path.join(output_path, "test_raw.csv"), index=False)

# 📊 Report
print("\n✅ Finished splitting and saving:")
print(f"  Training set: {len(train_df):,} rows")
print(f"  Testing set:  {len(test_df):,} rows")
print("  Saved to:", output_path)
