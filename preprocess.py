import tensorflow as tf
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

# -------------------------
# Configuration
# -------------------------
IMG_SIZE = (224, 224)

# Example folders
train_img_dir = './data/train/images'
train_mask_dir = './data/train/masks'

val_img_dir = './data/val/images'
val_mask_dir = './data/val/masks'

test_img_dir = './data/test/images'
test_mask_dir = './data/test/masks'

# -------------------------
# Dataset Loader
# -------------------------
def load_images_and_masks(image_dir, mask_dir):
    images = []
    masks = []
    img_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(img_files, mask_files):
        # Load image
        img_path = os.path.join(image_dir, img_file)
        img = imread(img_path)
        img = resize(img, IMG_SIZE, anti_aliasing=True)
        if img.shape[-1] == 4:  # Remove alpha channel if present
            img = img[..., :3]
        images.append(img)

        # Load mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = imread(mask_path, as_gray=True)
        mask = resize(mask, IMG_SIZE, anti_aliasing=False)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask / 255.0  # Normalize mask
        masks.append(mask)

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

# -------------------------
# Load all datasets
# -------------------------
train_images, train_masks = load_images_and_masks(train_img_dir, train_mask_dir)
val_images, val_masks = load_images_and_masks(val_img_dir, val_mask_dir)
test_images, test_masks = load_images_and_masks(test_img_dir, test_mask_dir)

print("Train:", train_images.shape, train_masks.shape)
print("Val:", val_images.shape, val_masks.shape)
print("Test:", test_images.shape, test_masks.shape)

# Convert to TensorFlow datasets
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
