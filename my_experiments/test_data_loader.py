import sys
import os
import torch
import numpy as np

# 1. Add the upar_dataset path so Python can find the UPAR class
sys.path.append('/content/upar_dataset/UPAR')
from UPAR import UPAR, get_transform

# 2. Define Paths
ROOT_PATH = "/content/drive/MyDrive/UPAR_Project/datasets"
DATA_PATH = "/content/upar_dataset/UPAR/dataset_all.pkl"

# 3. Get the transforms
train_transform, valid_transform = get_transform(256, 128)

# 4. Initialize the Dataset
print("Loading dataset...")
try:
    dataset = UPAR(split='train', partition=0, root=ROOT_PATH, data_path=DATA_PATH, transform=train_transform)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit()

print(f"Total images in this split: {len(dataset)}")
print(f"Number of attributes: {dataset.attr_num}")
print(f"Attribute names: {dataset.attr_id}")

# 5. Try to load the very first image
print("\nTrying to load the first image...")
try:
    img, gt_label, did, imgname = dataset[0]
    print("✅ Image loaded successfully!")
    print(f"Image shape: {img.shape} (Should be [3, 256, 128])")
    print(f"Label shape: {gt_label.shape} (Should be [40])")
    print(f"Dataset ID: {did}")
    print(f"Image Name: {imgname}")
    print(f"Labels (first 10): {gt_label[:10]}")
except Exception as e:
    print(f"❌ Error loading image: {e}")
    print("\nThis usually means the image path in the pickle file doesn't match your Google Drive folder structure.")