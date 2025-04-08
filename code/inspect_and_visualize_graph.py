import dgl
import torch
import matplotlib.pyplot as plt

# 📥 Load the graph
graph = dgl.load_graphs("data/graph/rgcn_graph.bin")[0][0]

# 🧠 Node features and labels
features = graph.ndata["feat"]
labels = graph.ndata["label"]

# 📊 Basic Info
print("✅ R-GCN Graph Loaded!")
print("📏 Number of Nodes:", graph.num_nodes())
print("🔗 Number of Edges:", graph.num_edges())
print("🧠 Node Feature Shape:", features.shape)

# 🎯 Label distribution
print("🎯 Label Distribution:", torch.bincount(labels))

# 📈 Visualize distribution of first 3 features
plt.figure(figsize=(10, 6))
for i in range(min(3, features.shape[1])):
    plt.hist(features[:, i].cpu().numpy(), bins=30, alpha=0.6, label=f"Feature {i}")
plt.title("Distribution of First Few Node Features")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
