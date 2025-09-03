import tensorflow as tf
import os
import numpy as np

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
SUPPORTED_EXTENSIONS = {'.jpg', '.png', '.bmp', '.tif', '.tiff', '.jpeg'}

# Dataset paths
DATASET_PATHS = {
    'CoMoFoD': {
        'train': {'images': './data/CoMoFoD/train/images', 'masks': './data/CoMoFoD/train/masks'},
        'val': {'images': './data/CoMoFoD/val/images', 'masks': './data/CoMoFoD/val/masks'},
        'test': {'images': './data/CoMoFoD/test/images', 'masks': './data/CoMoFoD/test/masks'}
    },
    'IMD': {
        'train': {'images': './data/IMD/train/images', 'masks': './data/IMD/train/masks'},
        'val': {'images': './data/IMD/val/images', 'masks': './data/IMD/val/masks'},
        'test': {'images': './data/IMD/test/images', 'masks': './data/IMD/test/masks'}
    },
    'ARD': {
        'train': {'images': './data/ARD/train/images', 'masks': './data/ARD/train/masks'},
        'val': {'images': './data/ARD/val/images', 'masks': './data/ARD/val/masks'},
        'test': {'images': './data/ARD/test/images', 'masks': './data/ARD/test/masks'}
    },
    'Covrage': {
        'train': {'images': './data/Covrage/train/images', 'masks': './data/Covrage/train/masks'},
        'val': {'images': './data/Covrage/val/images', 'masks': './data/Covrage/val/masks'},
        'test': {'images': './data/Covrage/test/images', 'masks': './data/Covrage/test/masks'}
    },
    'GRip': {
        'train': {'images': './data/GRip/train/images', 'masks': './data/GRip/train/masks'},
        'val': {'images': './data/GRip/val/images', 'masks': './data/GRip/val/masks'},
        'test': {'images': './data/GRip/test/images', 'masks': './data/GRip/test/masks'}
    }
}

def preprocess_image(img_path):
    """Load and preprocess an image with supported extensions."""
    img = tf.io.read_file(img_path)
    ext = tf.strings.lower(tf.strings.split(img_path, '.')[-1])
    if ext in ['jpg', 'jpeg']:
        img = tf.image.decode_jpeg(img, channels=3)
    elif ext == 'png':
        img = tf.image.decode_png(img, channels=3)
    elif ext in ['bmp', 'tif', 'tiff']:
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
    else:
        raise ValueError(f"Unsupported image extension: {ext}")
    
    img = tf.image.resize(img, IMG_SIZE, method='bilinear')
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

def preprocess_mask(mask_path):
    """Load and preprocess a mask with supported extensions."""
    mask = tf.io.read_file(mask_path)
    ext = tf.strings.lower(tf.strings.split(mask_path, '.')[-1])
    if ext == 'png':
        mask = tf.image.decode_png(mask, channels=1)
    elif ext in ['jpg', 'jpeg', 'bmp', 'tif', 'tiff']:
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    else:
        raise ValueError(f"Unsupported mask extension: {ext}")
    
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = mask / 255.0
    return mask

def preprocess_image_mask_pair(img_path, mask_path):
    """Preprocess an image-mask pair."""
    tf.random.set_seed(SEED)
    img = preprocess_image(img_path)
    mask = preprocess_mask(mask_path)
    return img, mask

def get_image_mask_pairs(images_dir, masks_dir):
    """Get paired image and mask paths, supporting _gt/_mask suffixes."""
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(
            [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(ext)]
        )
    
    image_paths = sorted(image_paths)
    
    mask_paths = []
    for img_path in image_paths:
        filename_base = os.path.basename(img_path).rsplit('.', 1)[0]
        mask_found = False
        
        for suffix in ['_gt', '_mask']:
            for ext in SUPPORTED_EXTENSIONS:
                mask_path = os.path.join(masks_dir, f"{filename_base}{suffix}{ext}")
                if os.path.exists(mask_path):
                    mask_paths.append(mask_path)
                    mask_found = True
                    break
            if mask_found:
                break
        else:
            raise FileNotFoundError(f"No matching mask found for image: {img_path}")

    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Mismatch between number of images and masks")

    return image_paths, mask_paths

def create_dataset(images_dir, masks_dir):
    """Create a dataset from image and mask directories."""
    image_paths, mask_paths = get_image_mask_pairs(images_dir, masks_dir)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(preprocess_image_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000, seed=SEED)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def load_datasets(split):
    """Load datasets for each dataset name."""
    datasets = {}
    for dataset_name, paths in DATASET_PATHS.items():
        images_dir = paths[split]['images']
        masks_dir = paths[split]['masks']
        if not (os.path.exists(images_dir) and os.path.exists(masks_dir)):
            print(f"Warning: Skipping {dataset_name} {split} (directory not found)")
            continue
        datasets[dataset_name] = create_dataset(images_dir, masks_dir)
    
    if not datasets:
        raise ValueError(f"No valid datasets found for {split} split")
    
    return datasets
