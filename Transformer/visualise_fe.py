# import torch
# from torchvision import models, transforms
# from PIL import Image
# import matplotlib.pyplot as plt

# # Load image
# img = Image.open("test_images/my_pet.jpg")
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
# img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# # Load model and hook a layer
# model = models.resnet18(pretrained=True)
# model.eval()

# # Hook to capture feature maps
# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

# model.layer1[0].conv1.register_forward_hook(get_activation('conv1'))

# # Forward pass
# _ = model(img_tensor)

# # Visualize feature maps
# act = activation['conv1'].squeeze()  # remove batch dim
# fig, axarr = plt.subplots(1, 5, figsize=(15, 5))
# for idx in range(5):  # first 5 feature maps
#     axarr[idx].imshow(act[idx].cpu(), cmap='viridis')
#     axarr[idx].axis('off')
# plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load image
image_path = 'test_images/my_pet.jpg'
original = cv2.imread(image_path)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Step 1: Edge Detection
edges = cv2.Canny(original, threshold1=100, threshold2=200)

# Step 2: Feature Vector (basic example)
edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
area = np.sum(binary > 0)

height, width = edges.shape
shape_ratio = width / height

# Step 3: Plot
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(original_rgb)
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(edges, cmap='gray')
axs[1].set_title("Edge Detection")
axs[1].axis('off')

axs[2].text(0.1, 0.8, f"Edge density: {edge_density:.2f}", fontsize=12)
axs[2].text(0.1, 0.6, f"Shape ratio: {shape_ratio:.2f}", fontsize=12)
axs[2].text(0.1, 0.4, f"Area: {area} px", fontsize=12)
axs[2].set_title("Feature Vector")
axs[2].axis('off')

plt.tight_layout()
plt.show()
