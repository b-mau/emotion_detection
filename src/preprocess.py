# src/preprocess.py
import os
import glob
import pickle
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Paths and labels
classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
data_dir = "data/sample_images"  # Put a small sample of images here

# Load images
images, labels = [], []

for idx, emotion in enumerate(classes):
    path = os.path.join(data_dir, emotion, "*.jpg")
    for filename in glob.glob(path):
        im = Image.open(filename).convert("L")  # grayscale
        images.append(np.array(im).flatten())
        labels.append(idx)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Save to .pkl
os.makedirs("data", exist_ok=True)
with open("data/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("data/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("data/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("data/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("Preprocessing done. Data saved to 'data/' folder.")
