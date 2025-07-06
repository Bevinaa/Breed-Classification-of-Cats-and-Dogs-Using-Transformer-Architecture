# 1. Imports
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Model Definition
class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_prob=0.4):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10, dropout_prob=0.4):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim, dropout_prob)
        self.res_block2 = ResidualBlock(hidden_dim, dropout_prob)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.output_layer(x)
        return x

# 3. Load full dataset (features + labels)
features = np.load("combined_features.npy")  # shape: (samples, feature_dim)
labels = np.load("labels.npy")               # shape: (samples,)

# 4. Split into val set (again)
train_features_np, val_features_np, train_labels_np, val_labels_np = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels)

val_features = torch.from_numpy(val_features_np).float()
val_labels = torch.from_numpy(val_labels_np).long()

# 5. Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dim = 2560
num_classes = len(np.unique(labels))

model = ResidualMLP(input_dim=input_dim, num_classes=num_classes)
model.load_state_dict(torch.load("residual_mlp_final.pth", map_location=device))
model.to(device)
model.eval()

# 6. Predict
with torch.no_grad():
    inputs = val_features.to(device)
    labels = val_labels.to(device)
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    true = labels.cpu().numpy()

# 7. Metrics
print("📊 Classification Report:")
print(classification_report(true, preds, digits=4))

# 8. Confusion Matrix Plot
cm = confusion_matrix(true, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
