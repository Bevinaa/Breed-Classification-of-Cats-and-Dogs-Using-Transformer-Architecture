import os
import torch
import timm
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# ==== SETUP ====
image_path = 'test_images/my_pet.jpg'  # Replace with your image path
save_dir = 'heatmaps'
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = [
    'vit_base_patch16_224',
    'swin_base_patch4_window7_224',
    'deit_base_patch16_224',
    'beit_base_patch16_224',
    'convnext_base'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# ==== LOAD IMAGE ====
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)
original_img = np.array(img.resize((224, 224)))

# ==== HOOK AND VISUALIZATION FUNCTION ====
def generate_attention_map(model_name):
    print(f"Processing: {model_name}")
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()

    attention_data = []

    def hook_fn(module, input, output):
        attention_data.append(output)

    # Register appropriate hooks
    try:
        if 'vit' in model_name or 'deit' in model_name or 'beit' in model_name:
            model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)
        elif 'swin' in model_name:
            model.layers[-1].blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)
        elif 'convnext' in model_name:
            model.stages[-1][-1].drop_path.register_forward_hook(hook_fn)
        else:
            print("Unsupported model.")
            return

        with torch.no_grad():
            _ = model(input_tensor)

        attn = attention_data[0].squeeze(0).mean(0)
        if model_name != 'convnext_base':
            attn = attn[1:].mean(0) if attn.size(0) > 1 else attn.mean(0)
        attn = attn.reshape(14, 14).cpu().numpy()
        attn = cv2.resize(attn, (224, 224))
        attn = (attn - attn.min()) / (attn.max() - attn.min())
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        return

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap, 0.5, 0)
    save_path = os.path.join(save_dir, f"{model_name}_heatmap.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved: {save_path}")

# ==== RUN FOR ALL MODELS ====
for model in model_names:
    generate_attention_map(model)
