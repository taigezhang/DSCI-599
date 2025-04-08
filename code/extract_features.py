import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# === Paths ===
feature_list_path = "./data/selected_features_0.005.csv"
train_path = "./artifacts/train_raw.parquet"
test_path = "./artifacts/test_raw.parquet"
output_dir = "./data/processed"
os.makedirs(output_dir, exist_ok=True)

# === Custom label mapping to merge 18 -> 13 categories ===
label_mapping = {
    'Benign': 'Benign',
    'LDAP': 'LDAP',
    'MSSQL': 'MSSQL',
    'NetBIOS': 'NetBIOS',
    'Portmap': 'Portmap',
    'Syn': 'Syn',
    'UDP': 'UDP',
    'UDPLag': 'UDPLag',
    'UDP-lag': 'UDPLag',
    'DrDoS_NTP': 'DrDoS_NTP',
    'DrDoS_DNS': 'DNS',
    'DrDoS_LDAP': 'LDAP',
    'DrDoS_MSSQL': 'MSSQL',
    'DrDoS_NetBIOS': 'NetBIOS',
    'DrDoS_SNMP': 'SNMP',
    'DrDoS_UDP': 'UDP',
    'TFTP': 'TFTP',
    'WebDDoS': 'WebDDoS'
}

# === Final encoding: 13 unified classes -> integer labels ===
final_label_encoding = {
    'Benign': 0,
    'Syn': 1,
    'UDP': 2,
    'UDPLag': 3,
    'MSSQL': 4,
    'LDAP': 5,
    'DNS': 6,
    'NetBIOS': 7,
    'SNMP': 8,
    'Portmap': 9,
    'DrDoS_NTP': 10,
    'TFTP': 11,
    'WebDDoS': 12
}

# === Load features ===
selected_features = pd.read_csv(feature_list_path, header=None).squeeze("columns").tolist()
# Drop accidentally included index string '0'
selected_features = [f for f in selected_features if f != "0"]

# === Load and preprocess datasets ===
def load_and_prepare(path):
    df = pd.read_parquet(path)
    df = df[df["Label"].isin(label_mapping.keys())].copy()
    df["Label"] = df["Label"].map(label_mapping)            # Merge raw → 13 labels
    df["Label"] = df["Label"].map(final_label_encoding)     # Encode 13 labels → integers
    df.dropna(subset=selected_features + ["Label"], inplace=True)

    X = df[selected_features]
    y = df["Label"]

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    df[selected_features] = X_scaled
    df["Label"] = y
    return df

train_df = load_and_prepare(train_path)
test_df = load_and_prepare(test_path)

# === Save processed data ===
train_df.to_parquet(f"{output_dir}/train_processed_02.parquet", index=False)
test_df.to_parquet(f"{output_dir}/test_processed_02.parquet", index=False)

print("✅ Saved processed train and test datasets with encoded labels.")
