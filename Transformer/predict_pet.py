import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import json


with open('class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)

IMAGE_PATH = "test_images/my_pet7.jpg"
MODEL_PATH = "residual_mlp_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
MODEL_NAMES = [
    'vit_base_patch16_224',
    'swin_base_patch4_window7_224',
    'deit_base_patch16_224'
]

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

def load_transformer(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.reset_classifier(0)
    model.eval()
    return model.to(DEVICE)

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Load transformer models
    transformers = [load_transformer(name) for name in MODEL_NAMES]

    # Extract features
    with torch.no_grad():
        features = [model(image_tensor).cpu() for model in transformers]
        combined_features = torch.cat(features, dim=1).to(DEVICE)

    input_dim = combined_features.shape[1]
    output_dim = 35

    # Load Residual MLP
    model = ResidualMLP(input_dim=input_dim, num_classes=output_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Predict
    with torch.no_grad():
        logits = model(combined_features)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[0][pred_idx].item()

    print(f"\nImage: {image_path}")
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {confidence:.2f}")

    visualize_prediction(image_path, pred_class, confidence, probs, CLASS_NAMES)

def visualize_prediction(image_path, pred_class, confidence, class_probs, class_names):
    image = Image.open(image_path).convert("RGB")

    # Plot image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Prediction: {pred_class}\nConfidence: {confidence:.2%}", fontsize=14, color='green')
    plt.show()

    # Plot top-5 probabilities
    topk = 5
    probs_np = class_probs.cpu().numpy().flatten()
    top_indices = probs_np.argsort()[-topk:][::-1]
    top_probs = probs_np[top_indices]
    top_labels = [class_names[i] for i in top_indices]

    plt.figure(figsize=(10, 4))
    bars = plt.barh(range(topk), top_probs[::-1], color='skyblue')
    plt.yticks(range(topk), top_labels[::-1])
    plt.xlabel("Probability")
    plt.title("Top-5 Predicted Classes")
    for i, bar in enumerate(bars):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2%}", va='center')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_image(IMAGE_PATH)