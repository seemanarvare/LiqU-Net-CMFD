import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import dilation, disk
from preprocessing import load_datasets, BATCH_SIZE, IMG_SIZE, OUTPUT_FOLDER
from training import unet_model_with_liquid, LiquidLayer

def load_trained_model(model_path):
    """Load the trained model with custom LiquidLayer."""
    return tf.keras.models.load_model(model_path, custom_objects={'LiquidLayer': LiquidLayer})

def adaptive_binarize(mask):
    """Apply global and local thresholding with morphological dilation."""
    mask = mask.squeeze()
    
    # Global thresholding (Otsu)
    global_threshold = threshold_otsu(mask)
    global_binary = (mask > global_threshold).astype(np.uint8)
    
    # Local thresholding
    local_threshold = threshold_local(mask, block_size=35, offset=10)
    local_binary = (mask > local_threshold).astype(np.uint8)
    
    # Combine global and local with logical OR
    binary_mask = np.logical_or(global_binary, local_binary).astype(np.uint8)
    
    # Morphological dilation with disk (radius=3)
    selem = disk(3)
    binary_mask = dilation(binary_mask, selem)
    
    return binary_mask

def normalize_image(image):
    """Normalize image for visualization."""
    min_val = np.min(image)
    max_val = np.max(image)
    return np.clip((image - min_val) / (max_val - min_val), 0, 1)

def compute_metrics(y_true, y_pred):
    """Compute metrics including IoU."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'Accuracy': round(accuracy, 4),
        'IoU': round(iou, 4)
    }

def visualize_comparison_with_overlay(original_image, true_mask, pred_mask, filename, dataset_name):
    """Visualize original image, ground truth, predicted mask, and overlay."""
    output_dir = os.path.join(OUTPUT_FOLDER, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(normalize_image(original_image))
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(true_mask.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask.squeeze(), cmap='viridis')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title('Original + Predicted Mask')
    plt.imshow(normalize_image(original_image))
    plt.imshow(pred_mask.squeeze(), cmap='viridis', alpha=0.5)
    plt.axis('off')
    
    output_file_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_comparison_overlay.png')
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def evaluate_model():
    """Evaluate the model on all test datasets."""
    test_datasets = load_datasets('test')
    model = load_trained_model(os.path.join(OUTPUT_FOLDER, 'liquid_U_Net_50_epoch.keras'))
    
    results = {}
    for dataset_name, test_dataset in test_datasets.items():
        print(f"Evaluating on dataset: {dataset_name}")
        
        # Extract images and masks from dataset
        images, masks, filenames = [], [], []
        for img, mask, fname in test_dataset:
            images.append(img.numpy())
            masks.append(mask.numpy())
            filenames.append(os.path.basename(fname.numpy().decode('utf-8')))
        
        images = np.array(images)
        masks = np.array(masks)
        
        if len(images) == 0:
            print(f"Warning: No images found for dataset {dataset_name}, skipping.")
            continue
        
        # Predict in batches
        predictions_bin = []
        for i in range(0, len(images), BATCH_SIZE):
            batch_images = images[i:i + BATCH_SIZE]
            predictions = model.predict(batch_images, batch_size=BATCH_SIZE)
            predictions_bin.extend([adaptive_binarize(pred) for pred in predictions])
        predictions_bin = np.array(predictions_bin)
        
        # Binarize ground truth masks
        resized_test_masks_bin = np.array([adaptive_binarize(mask) for mask in masks])
        
        # Compute metrics
        metrics_data = []
        sum_metrics = {key: 0 for key in compute_metrics(resized_test_masks_bin[0], predictions_bin[0]).keys()}
        num_images = len(filenames)
        
        for i, filename in enumerate(filenames):
            mask = resized_test_masks_bin[i]
            pred_mask = predictions_bin[i]
            
            metrics = compute_metrics(mask, pred_mask)
            metrics['Filename'] = filename
            metrics_data.append(metrics)
            
            for key in sum_metrics.keys():
                sum_metrics[key] += metrics.get(key, 0)
            
            # Visualize
            original_image = images[i]
            visualize_comparison_with_overlay(original_image, mask, pred_mask, filename, dataset_name)
        
        # Save per-image metrics
        output_dir = os.path.join(OUTPUT_FOLDER, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(os.path.join(output_dir, f'{dataset_name}_image_metrics.xlsx'), index=False)
        
        # Save average metrics
        average_metrics = {key: (value / num_images) if key != 'Filename' else 'Average' for key, value in sum_metrics.items()}
        average_metrics_df = pd.DataFrame([average_metrics])
        average_metrics_df.to_excel(os.path.join(output_dir, f'{dataset_name}_average_metrics.xlsx'), index=False)
        
        results[dataset_name] = {
            'image_metrics': metrics_data,
            'average_metrics': average_metrics
        }
        
        print(f"Dataset: {dataset_name}")
        for key, value in average_metrics.items():
            if key != 'Filename':
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        print(f"Metrics saved to: {output_dir}")
    
    return results

def evaluate_new_dataset(images_dir, masks_dir, dataset_name, model):
    """Evaluate the model on a new dataset."""
    image_paths = []
    for ext in ['.jpg', '.png', '.bmp', '.tif', '.tiff', '.jpeg']:
        image_paths.extend(
            [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(ext)]
        )
    image_paths = sorted(image_paths)
    
    images, masks, filenames = [], [], []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing image: {filename}")
        
        filename_base = os.path.splitext(filename)[0]
        mask_found = False
        
        for suffix in ['_gt', '_mask']:
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                mask_path = os.path.join(masks_dir, f"{filename_base}{suffix}{ext}")
                if os.path.exists(mask_path):
                    img = image.load_img(img_path, target_size=IMG_SIZE)
                    img = image.img_to_array(img)
                    img = preprocess_input(img)
                    
                    mask = image.load_img(mask_path, target_size=IMG_SIZE, color_mode="grayscale")
                    mask = image.img_to_array(mask) / 255.0
                    
                    images.append(img)
                    masks.append(mask)
                    filenames.append(filename)
                    mask_found = True
                    print(f"Loaded mask for {filename}: {mask_path}")
                    break
            if mask_found:
                break
        
        if not mask_found:
            raise FileNotFoundError(f"No mask found for image: {img_path}")
    
    images = np.array(images)
    masks = np.array(masks)
    
    if len(images) == 0:
        print(f"Warning: No images found for dataset {dataset_name}, skipping.")
        return None
    
    # Predict in batches
    predictions_bin = []
    for i in range(0, len(images), BATCH_SIZE):
        batch_images = images[i:i + BATCH_SIZE]
        predictions = model.predict(batch_images, batch_size=BATCH_SIZE)
        predictions_bin.extend([adaptive_binarize(pred) for pred in predictions])
    predictions_bin = np.array(predictions_bin)
    
    # Binarize ground truth masks
    resized_test_masks_bin = np.array([adaptive_binarize(mask) for mask in masks])
    
    # Compute metrics
    metrics_data = []
    sum_metrics = {key: 0 for key in compute_metrics(resized_test_masks_bin[0], predictions_bin[0]).keys()}
    num_images = len(filenames)
    
    for i, filename in enumerate(filenames):
        mask = resized_test_masks_bin[i]
        pred_mask = predictions_bin[i]
        
        metrics = compute_metrics(mask, pred_mask)
        metrics['Filename'] = filename
        metrics_data.append(metrics)
        
        for key in sum_metrics.keys():
            sum_metrics[key] += metrics.get(key, 0)
        
        # Visualize
        original_image = images[i]
        visualize_comparison_with_overlay(original_image, mask, pred_mask, filename, dataset_name)
    
    # Save per-image metrics
    output_dir = os.path.join(OUTPUT_FOLDER, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_excel(os.path.join(output_dir, f'{dataset_name}_image_metrics.xlsx'), index=False)
    
    # Save average metrics
    average_metrics = {key: (value / num_images) if key != 'Filename' else 'Average' for key, value in sum_metrics.items()}
    average_metrics_df = pd.DataFrame([average_metrics])
    average_metrics_df.to_excel(os.path.join(output_dir, f'{dataset_name}_average_metrics.xlsx'), index=False)
    
    print(f"Dataset: {dataset_name}")
    for key, value in average_metrics.items():
        if key != 'Filename':
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Metrics saved to: {output_dir}")
    
    return {'image_metrics': metrics_data, 'average_metrics': average_metrics}

def plot_accuracy_and_loss(history):
    """Plot training and validation accuracy and loss."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'training_curves.png'))
    plt.close()

if __name__ == "__main__":
    from training import train_model
    history, model = train_model()
    plot_accuracy_and_loss(history)
    results = evaluate_model()
    # Example for new dataset (uncomment to use):
    # new_results = evaluate_new_dataset('path/to/new/images', 'path/to/new/masks', 'NewDataset', model)
