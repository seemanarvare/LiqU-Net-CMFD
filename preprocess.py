import tensorflow as tf
import pandas as pd
import os

# Configuration
dataset_paths = {
    'CoMoFoD': {
        'train': './data/CoMoFoD/train_dataset.csv',
        'val': './data/CoMoFoD/val_dataset.csv',
        'test': './data/CoMoFoD/test_dataset.csv'
    },
    'IMD': {
        'train': './data/IMD/train_dataset.csv',
        'val': './data/IMD/val_dataset.csv',
        'test': './data/IMD/test_dataset.csv'
    },
    'ARD': {
        'train': './data/ARD/train_dataset.csv',
        'val': './data/ARD/val_dataset.csv',
        'test': './data/ARD/test_dataset.csv'
    },
    'Covrage': {
        'train': './data/Covrage/train_dataset.csv',
        'val': './data/Covrage/val_dataset.csv',
        'test': './data/Covrage/test_dataset.csv'
    },
    'GRip': {
        'train': './data/GRip/train_dataset.csv',
        'val': './data/GRip/val_dataset.csv',
        'test': './data/GRip/test_dataset.csv'
    }
}
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Loader 
def create_dataset(csv_path, batch_size):
    """Load and preprocess images and masks from CSV file."""
    df = pd.read_csv(csv_path)
    dataset = tf.data.Dataset.from_tensor_slices((df['Image_Path'], df['Mask_Path']))

    def load_and_preprocess(img_path, mask_path):
        # Load and preprocess image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method='bilinear')  # Bilinear interpolation
        img = tf.keras.applications.resnet50.preprocess_input(img)  # ImageNet normalization

        # Load and preprocess mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, IMG_SIZE, method='nearest')  # Nearest-neighbor for masks
        mask = mask / 255.0  # Normalize to [0, 1]
        return img, mask

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_all_datasets(dataset_paths, split, batch_size):
    """Combine datasets for training, validation, or testing."""
    datasets = [create_dataset(paths[split], batch_size) for paths in dataset_paths.values()]
    return tf.data.Dataset.concatenate(*datasets)

# Save datasets
def save_datasets():
    """Load and save datasets for training, validation, and testing."""
    os.makedirs('./output/datasets', exist_ok=True)
    print("Loading and saving datasets...")
    
    # Training dataset
    train_dataset = load_all_datasets(dataset_paths, 'train', BATCH_SIZE)
    train_images, train_masks = [], []
    for img, mask in train_dataset.unbatch():
        train_images.append(img.numpy())
        train_masks.append(mask.numpy())
    train_images = np.array(train_images)
    train_masks = np.array(train_masks)
    np.save('./output/datasets/train_images.npy', train_images)
    np.save('./output/datasets/train_masks.npy', train_masks)
    
    # Validation dataset
    val_dataset = load_all_datasets(dataset_paths, 'val', BATCH_SIZE)
    val_images, val_masks = [], []
    for img, mask in val_dataset.unbatch():
        val_images.append(img.numpy())
        val_masks.append(mask.numpy())
    val_images = np.array(val_images)
    val_masks = np.array(val_masks)
    np.save('./output/datasets/val_images.npy', val_images)
    np.save('./output/datasets/val_masks.npy', val_masks)
    
    # Test dataset
    test_dataset = load_all_datasets(dataset_paths, 'test', BATCH_SIZE)
    test_images, test_masks = [], []
    for img, mask in test_dataset.unbatch():
        test_images.append(img.numpy())
        test_masks.append(mask.numpy())
    test_images = np.array(test_images)
    test_masks = np.array(test_masks)
    np.save('./output/datasets/test_images.npy', test_images)
    np.save('./output/datasets/test_masks.npy', test_masks)
    
    print("Datasets saved to './output/datasets/'")

if __name__ == "__main__":
    save_datasets()
