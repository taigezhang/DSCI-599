import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors

# === Paths ===
train_path = "data/processed/train_smote.parquet"  # ✅ Corrected
test_path = "data/processed/test_processed_02.parquet"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "best_graphsage_model.pth")

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# === Load Data ===
print("🔹 Loading data...")
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# === Encode Labels ===
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df["Label"])
test_labels = label_encoder.transform(test_df["Label"])

# === Normalize Features ===
non_feature_cols = ["Label", "_source_file"]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df.drop(columns=non_feature_cols, errors="ignore"))
X_test = scaler.transform(test_df.drop(columns=non_feature_cols, errors="ignore"))

X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([train_labels, test_labels])

# === Build KNN Graph ===
k = 15
print(f"🔎 K={k} - Constructing graph...")
nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
_, indices = nbrs.kneighbors(X_all)

src, dst = [], []
for i in range(indices.shape[0]):
    for j in indices[i][1:]:  # Skip self-loop
        src.append(i)
        dst.append(j)

g = dgl.graph((src, dst), num_nodes=X_all.shape[0])
g = dgl.to_bidirected(g)
g.ndata["feat"] = torch.tensor(X_all, dtype=torch.float32)
g.ndata["label"] = torch.tensor(y_all, dtype=torch.long)

print(f"✅ Graph created: {g.num_nodes()} nodes, {g.num_edges()} edges")

# === GraphSAGE Model ===
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout=0.2):
        super().__init__()
        self.sage1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.sage2 = SAGEConv(h_feats, h_feats, aggregator_type="mean")
        self.sage3 = SAGEConv(h_feats, h_feats, aggregator_type="mean")
        self.out = SAGEConv(h_feats, num_classes, aggregator_type="mean")
        self.dropout = dropout

    def forward(self, g, x):
        x = F.relu(self.sage1(g, x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.sage2(g, x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.sage3(g, x))
        return self.out(g, x)

# === Masks and Loss Weights ===
train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
train_mask[:len(train_df)] = True
test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
test_mask[len(train_df):] = True

labels_tensor = torch.tensor(train_labels)
class_counts = torch.bincount(labels_tensor)
weights = 1.0 / class_counts.float()
weights = weights / weights.sum()

# === Initialize Model ===
model = GraphSAGE(
    in_feats=X_all.shape[1],
    h_feats=128,
    num_classes=len(label_encoder.classes_)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# === Training with Early Stopping (if needed) ===
early_stop_patience = 25
best_loss = float("inf")
wait = 0

if not os.path.exists(model_path):
    print("🚀 Training GraphSAGE...")
    for epoch in range(1, 1001):
        model.train()
        logits = model(g.to(device), g.ndata["feat"].to(device))
        loss = criterion(logits[train_mask], g.ndata["label"][train_mask].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                pred = logits[test_mask].argmax(dim=1)
                acc = (pred == g.ndata["label"][test_mask].to(device)).float().mean().item()
                print(f"📦 Epoch {epoch} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), model_path)
                wait = 0
            else:
                wait += 1
                if wait >= early_stop_patience:
                    print("⏹️ Early stopping triggered.")
                    break
else:
    print("✅ Found existing model — skipping training.")

# === Load and Evaluate ===
print("\n📊 Evaluating best saved model...")
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    preds = model(g.to(device), g.ndata["feat"].to(device))[test_mask].argmax(dim=1).cpu()
    true = g.ndata["label"][test_mask].cpu()
    present_labels = np.unique(true.numpy())
    target_names = [str(label_encoder.classes_[i]) for i in present_labels]
    print(classification_report(true.numpy(), preds.numpy(), labels=present_labels, target_names=target_names))
