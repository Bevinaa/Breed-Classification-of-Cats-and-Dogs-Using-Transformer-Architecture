import numpy as np

f1 = np.load('combined_features.npy')         # From old code
f2 = np.load('new_combined_features.npy')     # From new models

final_combined = np.concatenate((f1, f2), axis=1)
np.save('combined_features.npy', final_combined)

print("Final combined feature shape:", final_combined.shape)
