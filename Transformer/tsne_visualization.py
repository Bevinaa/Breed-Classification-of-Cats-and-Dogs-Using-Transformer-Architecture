import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Step 1: Load the features and labels
features = np.load('combined_features.npy', allow_pickle=True)
labels = np.load('labels.npy')

# ✅ Step 2: Check the number of unique classes
unique_classes = np.unique(labels)
print(f"Total unique classes in labels: {len(unique_classes)}")
print(f"Classes present: {unique_classes}")

# ✅ Step 3: Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)

# ✅ Step 4: Plot with a large color palette for 35 classes
plt.figure(figsize=(14, 10))
sns.scatterplot(
    x=features_2d[:, 0],
    y=features_2d[:, 1],
    hue=labels,
    palette='hsv',         # Better palette for more colors (e.g., 'Set3', 'tab20', 'Spectral', etc.)
    legend='full',
    s=50,
    edgecolor='k'
)

# ✅ Step 5: Annotate and adjust plot
plt.title("t-SNE Visualization of Extracted Features from 35 Classes")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')
plt.tight_layout()
plt.grid(True)
plt.show()
