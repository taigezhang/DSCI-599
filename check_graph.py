import torch
from torch_geometric.data import Data

# ✅ Fix: Load the graph with `weights_only=False`
data = torch.load("ddos_graph.pt", weights_only=False)

# 🔹 Print graph information
print("\n✅ Graph Loaded Successfully!")
print(f"📌 Number of Nodes: {data.x.shape[0]}")
print(f"📌 Number of Edges: {data.edge_index.shape[1]}")
print(f"📌 Node Features Shape: {data.x.shape}")
print(f"📌 Edge Index Shape: {data.edge_index.shape}")
