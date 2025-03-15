import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ✅ Load the graph
data = torch.load("ddos_graph.pt", weights_only=False)

# ✅ Ensure data is on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

# ✅ Compute Class Weights to Handle Imbalance
attack_counts = torch.bincount(data.y)
class_weights = 1.0 / attack_counts.float()
class_weights /= class_weights.max()  # Normalize by max count
class_weights = class_weights.to(device)

# ✅ Split data into training & test sets
num_nodes = data.x.shape[0]
train_mask, test_mask = train_test_split(torch.arange(num_nodes), test_size=0.2, random_state=42)
train_mask = torch.tensor(train_mask, dtype=torch.long).to(device)
test_mask = torch.tensor(test_mask, dtype=torch.long).to(device)

# ✅ Define a deeper GCN model with BatchNorm & Dropout
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# ✅ Initialize the model
model = GCN(num_features=data.x.shape[1], hidden_dim=64, num_classes=len(attack_counts)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)  # Reduce LR every 1000 epochs

# ✅ Use Focal Loss Instead of CrossEntropyLoss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=class_weights, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss()

# ✅ Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Prevent NaNs
    optimizer.step()
    return loss.item()

# ✅ Evaluation function with Precision, Recall, F1-score
def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.size(0)

        # Print detailed classification report
        print("\n📊 Classification Report:")
        print(classification_report(data.y[test_mask].cpu(), pred[test_mask].cpu(), digits=4))
        
    return acc

# 🔥 Train for 5000 epochs
print("🚀 Training GCN Model...")
for epoch in range(5000):
    loss = train()
    scheduler.step()  # Adjust learning rate
    if epoch % 500 == 0:  # Print every 500 epochs
        acc = evaluate()
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Test Accuracy = {acc:.4f}")

print("✅ Training Complete!")

# ✅ Final Model Evaluation
final_acc = evaluate()
print(f"🎯 Final Test Accuracy: {final_acc:.4f}")

# ✅ Save the trained model
torch.save(model.state_dict(), "gcn_model_focal.pth")
print("✅ Model saved as 'gcn_model_focal.pth'")
