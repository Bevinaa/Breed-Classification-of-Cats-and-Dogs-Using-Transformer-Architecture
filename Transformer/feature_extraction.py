import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm
import numpy as np
from tqdm import tqdm

IMAGE_SIZE = 224
BATCH_SIZE = 16
DATA_DIR = 'data'
DEVICE = torch.device('cpu')

MODEL_NAMES = [
    'vit_base_patch16_224',
    'swin_base_patch4_window7_224',
    'deit_base_patch16_224'
]

def load_feature_extractor(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.reset_classifier(0)  # Remove classification head
    model.eval()
    return model.to(DEVICE)

def main():
    print(f"Device in use: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Loaded {len(dataset)} images across {len(dataset.classes)} classes.")

    print("Loading transformer models...")
    models = [load_feature_extractor(name) for name in MODEL_NAMES]
    print(f"Loaded models: {MODEL_NAMES}")

    # Extract features
    all_features = []
    all_labels = []

    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(DEVICE)
            features = []

            for model in models:
                out = model(images)  # shape: [BATCH_SIZE, feature_dim]
                features.append(out.cpu())

            combined = torch.cat(features, dim=1)  # shape: [BATCH_SIZE, combined_dim]
            all_features.append(combined)
            all_labels.append(labels)

    features_np = torch.cat(all_features).numpy()
    labels_np = torch.cat(all_labels).numpy()

    np.save('combined_features.npy', features_np)
    np.save('labels.npy', labels_np)

    print("Feature extraction complete.")
    print(f"Saved combined_features.npy with shape {features_np.shape}")
    print(f"Saved labels.npy with shape {labels_np.shape}")

if __name__ == "__main__":
    main()