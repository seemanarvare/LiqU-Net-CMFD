import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Paths
images_dir = r"D:\datasets\CoMoFoD\images"
masks_dir = r"D:\datasets\CoMoFoD\masks"
output_dir = r"D:\datasets\CoMoFoD\processed"

img_size = (224, 224)

os.makedirs(output_dir, exist_ok=True)

def load_images_and_masks(images_dir, masks_dir, target_size):
    images, masks = [], []
    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(images_dir, filename)
        mask_name = os.path.splitext(filename)[0] + "_gt.png"  # adjust if mask naming different
        mask_path = os.path.join(masks_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {filename}, skipping...")
            continue

        # Load and preprocess image
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img) / 255.0

        # Load and preprocess mask
        mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0
        mask = (mask > 0.5).astype(np.float32)  # binarize

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load dataset
print("Loading dataset...")
X, y = load_images_and_masks(images_dir, masks_dir, img_size)
print("Dataset loaded:", X.shape, y.shape)

# Train-val-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save preprocessed data
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

print("Preprocessing completed. Files saved in:", output_dir)
