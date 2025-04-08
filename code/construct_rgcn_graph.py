import pandas as pd
import torch
import dgl
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# Load preprocessed training data
train_path = "./data/processed/train_processed.parquet"
df = pd.read_parquet(train_path)

# Encode labels
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])

# Simulate node IDs based on data index for source and destination
df["src"] = df.index % 255
df["dst"] = (df.index + 1) % 255

# Create edges and determine number of nodes
edges_src = torch.tensor(df["src"].values, dtype=torch.int64)
edges_dst = torch.tensor(df["dst"].values, dtype=torch.int64)
num_nodes = max(edges_src.max().item(), edges_dst.max().item()) + 1

# Normalize features
exclude_cols = ["Label", "src", "dst"]
feature_columns = df.columns.difference(exclude_cols)
scaler = MinMaxScaler()
features = torch.tensor(scaler.fit_transform(df[feature_columns]), dtype=torch.float32)

# Aggregate features per node by averaging
node_features = torch.zeros((num_nodes, features.shape[1]))
node_counts = torch.zeros(num_nodes, dtype=torch.int64)

for i in range(len(df)):
    node_features[df["src"][i]] += features[i]
    node_counts[df["src"][i]] += 1

node_counts[node_counts == 0] = 1
node_features /= node_counts.unsqueeze(1)

# Construct graph
graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)
graph.ndata["feat"] = node_features

# Assign label to src nodes (as node classification task)
node_labels = torch.full((num_nodes,), -1, dtype=torch.long)
for i in range(len(df)):
    node_labels[df["src"][i]] = df["Label"][i]
graph.ndata["label"] = node_labels

# Save graph
os.makedirs("data/graph", exist_ok=True)
graph_path = "data/graph/rgcn_graph.bin"
dgl.save_graphs(graph_path, [graph])

print(f"✅ R-GCN graph saved to {graph_path}")
