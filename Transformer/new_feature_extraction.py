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
DEVICE = torch.device('cpu')  # Use 'cuda' if available and desired

# New transformer models to add
NEW_MODEL_NAMES = [
    'beit_base_patch16_224',
    'convnext_base'
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

    print("Loading new transformer models...")
    models = [load_feature_extractor(name) for name in NEW_MODEL_NAMES]
    print(f"Loaded models: {NEW_MODEL_NAMES}")

    # Extract features
    all_new_features = []

    print("Extracting features from new models...")
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(DEVICE)
            features = []

            for model in models:
                out = model(images)  # shape: [BATCH_SIZE, feature_dim]
                features.append(out.cpu())

            combined = torch.cat(features, dim=1)  # [BATCH_SIZE, new_combined_dim]
            all_new_features.append(combined)

    new_features_np = torch.cat(all_new_features).numpy()

    np.save('new_combined_features.npy', new_features_np)
    print("New feature extraction complete.")
    print(f"Saved new_combined_features.npy with shape {new_features_np.shape}")

if __name__ == "__main__":
    main()
