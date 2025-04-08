import os
import pandas as pd
import numpy as np
import torch
import dgl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score
)
from sklearn.neighbors import NearestNeighbors
from model import GraphSAGE

# === Final 13-class label mapping ===
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
index_to_label = {v: k for k, v in final_label_encoding.items()}

# === Paths ===
train_path = "data/processed/train_smote.parquet"
test_path = "data/processed/test_processed_02.parquet"
model_path = "models/best_graphsage_model.pth"
result_dir = "knn_gcn_model_result"
os.makedirs(result_dir, exist_ok=True)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧪 Using device: {device}")

# === Load Data ===
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# === Labels ===
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df["Label"])
test_labels = label_encoder.transform(test_df["Label"])

# === Features ===
non_feature_cols = ["Label", "_source_file"]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df.drop(columns=non_feature_cols, errors="ignore"))
X_test = scaler.transform(test_df.drop(columns=non_feature_cols, errors="ignore"))
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([train_labels, test_labels])

# === Build Graph ===
k = 15
nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
_, indices = nbrs.kneighbors(X_all)
src, dst = [], []
for i in range(indices.shape[0]):
    for j in indices[i][1:]:  # skip self
        src.append(i)
        dst.append(j)
g = dgl.graph((src, dst), num_nodes=X_all.shape[0])
g = dgl.to_bidirected(g)
g.ndata["feat"] = torch.tensor(X_all, dtype=torch.float32)
g.ndata["label"] = torch.tensor(y_all, dtype=torch.long)

# === Load Model ===
model = GraphSAGE(
    in_feats=X_all.shape[1],
    h_feats=128,
    num_classes=len(label_encoder.classes_)
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# === Evaluate ===
train_len = len(train_df)
test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
test_mask[train_len:] = True

with torch.no_grad():
    logits = model(g.to(device), g.ndata["feat"].to(device))[test_mask]
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    true = g.ndata["label"][test_mask].cpu().numpy()

# === Classification Report (with class names)
print("\n📊 Evaluation Report:")
present_labels = np.unique(true)
target_names = [index_to_label[i] for i in present_labels]
print(classification_report(true, preds, labels=present_labels, target_names=target_names))

# === Confusion Matrix ===
cm = confusion_matrix(true, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=target_names,
            yticklabels=target_names,
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
plt.show()

# === ROC AUC Curves ===
n_classes = len(label_encoder.classes_)
true_binarized = label_binarize(true, classes=np.arange(n_classes))
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    if np.sum(true_binarized[:, i]) == 0:
        continue  # skip class if not present
    fpr[i], tpr[i], _ = roc_curve(true_binarized[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 8))
for i in roc_auc:
    plt.plot(fpr[i], tpr[i], label=f"{index_to_label[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "roc_auc_curves.png"))
plt.show()

# === F1 Score Summary ===
f1_macro = f1_score(true, preds, average="macro")
f1_micro = f1_score(true, preds, average="micro")
print(f"\n📈 F1 Macro: {f1_macro:.4f} | F1 Micro: {f1_micro:.4f}")

# === Label Index Mapping
print("\n📚 Label Mapping (index → class name):")
for i in sorted(index_to_label):
    print(f"{i}: {index_to_label[i]}")
