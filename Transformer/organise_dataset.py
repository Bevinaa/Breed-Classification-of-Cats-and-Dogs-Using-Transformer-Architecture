import os

data_dir = "data"

for folder in os.listdir(data_dir):
    old_path = os.path.join(data_dir, folder)
    if os.path.isdir(old_path):
        new_name = folder.capitalize()
        new_path = os.path.join(data_dir, new_name)
        os.rename(old_path, new_path)

print("Folder names standardized to Title Case.")
