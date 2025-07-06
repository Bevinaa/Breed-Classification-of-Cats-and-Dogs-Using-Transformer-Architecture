import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 10
DATA_DIR = "data"

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

num_classes = len(dataset.classes)

# Model Wrappers

class ResidualMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        out = self.relu(self.fc1(x))
        res = out.clone()
        out = self.relu(self.fc2(out))
        out = out + res
        out = self.fc3(out)
        return out

def modify_model(base_model, num_classes):
    for param in base_model.parameters():
        param.requires_grad = False

    if hasattr(base_model, 'fc'):  # ResNet, etc.
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, num_classes)

    elif hasattr(base_model, 'classifier'):
        classifier = base_model.classifier
        if isinstance(classifier, nn.Sequential):  # VGG19
            in_features = classifier[-1].in_features
            classifier[-1] = nn.Linear(in_features, num_classes)
        elif isinstance(classifier, nn.Linear):  # DenseNet, MobileNet
            in_features = classifier.in_features
            base_model.classifier = nn.Linear(in_features, num_classes)

    return base_model


class RCNNStyle(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.rnn = nn.LSTM(input_size=64 * 56, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), x.size(1), -1)  # (B, C, H*W)
        x = x.permute(0, 2, 1)  # (B, H*W, C)
        _, (h_n, _) = self.rnn(x)
        return self.fc(h_n[-1])

MODELS = {
    "Residual MLP": ResidualMLP(num_classes),
    "VGG19": modify_model(models.vgg19(pretrained=True), num_classes),
    "ResNet101": modify_model(models.resnet101(pretrained=True), num_classes),
    "DenseNet201": modify_model(models.densenet201(pretrained=True), num_classes),
    "MobileNetV3": modify_model(models.mobilenet_v3_large(pretrained=True), num_classes),
    "RCNN": RCNNStyle(num_classes)
}

def train(model, loader):
    model.to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


def evaluate(model, loader, class_names, model_name="Model"):
    model.to(DEVICE)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            preds = model(xb).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"{model_name} Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.show()

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return acc


if __name__ == "__main__":
    for name, model in MODELS.items():
        print(f"\nTraining {name}...")
        train(model, train_loader)
        class_names = dataset.classes
        acc = evaluate(model, test_loader, class_names, model_name=name)
        print(f"{name} Accuracy: {acc:.4f}")
