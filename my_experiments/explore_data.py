import pickle
import numpy as np

# Path to the annotations (we will update this path when running in Colab)
# For local testing, point it to your local upar_dataset/UPAR/dataset_all.pkl
dataset_path = "path/to/dataset_all.pkl"

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

print("Keys in dataset:", data.keys())
# Try to print the length of the labels and the names of the 40 attributes
# print("Number of images:", len(data['labels']))
# print("Attribute names:", data['attributes'])