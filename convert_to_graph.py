import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ✅ Ensure GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 📂 File paths
file_path = "cicddos2019_cleaned.csv"
chunk_size = 50000  # Process in chunks

# 🏗 Initialize directed graph
G = nx.DiGraph()
ip_to_id = {}
node_features = {}
labels = {}

# 🔹 Define attack type mapping
attack_types = {
    "DrDoS_SNMP": 0,
    "DrDoS_DNS": 1,
    "DrDoS_MSSQL": 2,
    "DrDoS_NetBIOS": 3,
    "DrDoS_UDP": 4,
    "DrDoS_SSDP": 5,
    "DrDoS_LDAP": 6,
    "Syn": 7,
    "DrDoS_NTP": 8,
    "UDPLag": 9
}

attack_counts = {key: 0 for key in attack_types}  # Track attack counts

# ✅ Verify column existence
df_sample = pd.read_csv(file_path, nrows=5)
expected_columns = [
    "Source IP", "Destination IP", "Protocol", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Fwd Packet Length Min", "Bwd Packet Length Max",
    "Fwd IAT Mean", "Flow Packets/s", "SYN Flag Count", "Flow Bytes/s",
    "Min Packet Length", "Max Packet Length", "Attack_Type"
]

missing_cols = [col for col in expected_columns if col not in df_sample.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns in CSV: {missing_cols}")

print("🔹 Building Graph in Chunks...")

for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), desc="Processing CSV"):
    for _, row in chunk.iterrows():
        src, dst = row["Source IP"], row["Destination IP"]
        attack_type = row["Attack_Type"].strip()

        # ✅ Assign unique ID to each node
        if src not in ip_to_id:
            ip_to_id[src] = len(ip_to_id)
            node_features[ip_to_id[src]] = [
                row["Protocol"], row["Flow Duration"], row["Total Fwd Packets"], row["Total Backward Packets"],
                row["Fwd Packet Length Min"], row["Bwd Packet Length Max"], row["Fwd IAT Mean"],
                row["Flow Packets/s"], row["SYN Flag Count"], row["Flow Bytes/s"],
                row["Min Packet Length"], row["Max Packet Length"]
            ]

        if dst not in ip_to_id:
            ip_to_id[dst] = len(ip_to_id)
            node_features[ip_to_id[dst]] = [
                row["Protocol"], row["Flow Duration"], row["Total Fwd Packets"], row["Total Backward Packets"],
                row["Fwd Packet Length Min"], row["Bwd Packet Length Max"], row["Fwd IAT Mean"],
                row["Flow Packets/s"], row["SYN Flag Count"], row["Flow Bytes/s"],
                row["Min Packet Length"], row["Max Packet Length"]
            ]

        # ✅ Convert attack type to numeric labels
        label = attack_types.get(attack_type, -1)
        if label != -1:
            labels[ip_to_id[src]] = label
            attack_counts[attack_type] += 1  # Track count of attacks

        # ✅ Improved Edge Connectivity (Flow-based)
        if not G.has_edge(ip_to_id[src], ip_to_id[dst]):
            G.add_edge(ip_to_id[src], ip_to_id[dst])

# 🔍 Print attack type distribution
print("\n✅ Attack Type Distribution:")
for attack, count in attack_counts.items():
    print(f"{attack}: {count}")

print("✅ Graph Built. Now Converting to PyTorch Geometric Format...")

# 🔹 Convert to PyTorch Geometric format
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous().to(device)
node_features_tensor = torch.tensor(list(node_features.values()), dtype=torch.float).to(device)

# ✅ Ensure labels match node count
labels_tensor = torch.zeros(len(ip_to_id), dtype=torch.long).to(device)
for node_id, label in labels.items():
    labels_tensor[node_id] = label

# 📦 Create PyG Data object with labels
data = Data(x=node_features_tensor, edge_index=edge_index, y=labels_tensor).to(device)

# 💾 Save graph for training
torch.save(data, "ddos_graph.pt")
print("✅ Graph conversion completed. Saved as 'ddos_graph.pt' with labels!")
