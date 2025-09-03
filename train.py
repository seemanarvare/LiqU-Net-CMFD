import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Clear session and set random seed for reproducibility
tf.keras.backend.clear_session()
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# -------------------------
# Configuration
# -------------------------
dataset_paths = {
    'CoMoFoD': {
        'train': './data/CoMoFoD/train_dataset.csv',
        'val': './data/CoMoFoD/val_dataset.csv'
    },
    'IMD': {
        'train': './data/IMD/train_dataset.csv',
        'val': './data/IMD/val_dataset.csv'
    },
    'ARD': {
        'train': './data/ARD/train_dataset.csv',
        'val': './data/ARD/val_dataset.csv'
    },
    'Covrage': {
        'train': './data/Covrage/train_dataset.csv',
        'val': './data/Covrage/val_dataset.csv'
    },
    'GRip': {
        'train': './data/GRip/train_dataset.csv',
        'val': './data/GRip/val_dataset.csv'
    }
}
output_folder = './output'
os.makedirs(output_folder, exist_ok=True)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# -------------------------
# Data Loader
# -------------------------
def create_dataset(csv_path, batch_size):
    """Load and preprocess images and masks from CSV file."""
    df = pd.read_csv(csv_path)
    dataset = tf.data.Dataset.from_tensor_slices((df['Image_Path'], df['Mask_Path']))

    def load_and_preprocess(img_path, mask_path):
        # Load and preprocess image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method='bilinear')  
        img = tf.keras.applications.resnet50.preprocess_input(img)  

        # Load and preprocess mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, IMG_SIZE, method='nearest') 
        mask = mask / 255.0  # Normalize to [0, 1]
        return img, mask

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_all_datasets(dataset_paths, split, batch_size):
    """Combine datasets for training or validation."""
    datasets = [create_dataset(paths[split], batch_size) for paths in dataset_paths.values()]
    return tf.data.Dataset.concatenate(*datasets)

# -------------------------
# Custom Loss and IoU Metric 
# -------------------------
def custom_loss(y_true, y_pred):
    """Binary Cross-Entropy loss for binary segmentation."""
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def iou_metric(y_true, y_pred):
    """Intersection over Union metric."""
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

# -------------------------
# Liquid Neural Network Layer 
# -------------------------
class LiquidLayer(layers.Layer):
    def __init__(self, units, time_steps=6, tau=0.1, **kwargs):
        super(LiquidLayer, self).__init__(**kwargs)
        self.units = int(units)
        self.time_steps = int(time_steps)
        self.tau = float(tau)
        self.dt = 1.0 / float(self.time_steps)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Expected input shape [B, H, W, C], got {input_shape}")
        self.input_dim = int(input_shape[-1])
        self.w = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
            trainable=True,
            name=self.name + '_w'
        )
        self.u = self.add_weight(
            shape=(self.units, self.units),
            initializer=tf.keras.initializers.Orthogonal(seed=SEED),
            trainable=True,
            name=self.name + '_u'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name=self.name + '_b'
        )
        super(LiquidLayer, self).build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        HW = H * W
        inputs_reshaped = tf.reshape(inputs, [B, HW, C])  # [B, HW, C_in]
        state = tf.zeros([B, HW, self.units], dtype=inputs.dtype)

        for _ in range(self.time_steps):
            input_term = tf.tensordot(inputs_reshaped, self.w, axes=[[2], [0]])
            recurrent_term = tf.tensordot(state, self.u, axes=[[2], [0]])
            dxdt = -state / self.tau + tf.nn.relu(input_term + recurrent_term + self.b)
            state = state + self.dt * dxdt

        out = tf.reshape(state, [B, H, W, self.units])
        return out

    def get_config(self):
        cfg = super(LiquidLayer, self).get_config()
        cfg.update({'units': self.units, 'time_steps': self.time_steps, 'tau': self.tau})
        return cfg

# -------------------------
# Build LiqU-Net Model 
# -------------------------
def unet_model_with_liquid(input_size=(224, 224, 3)):
    """Build LiqU-Net with ResNet-50 encoder and LNN decoder."""
    inputs = Input(input_size)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Freeze ResNet-50 weights
    for layer in base_model.layers:
        layer.trainable = False

    # Encoder feature maps 
    c1 = base_model.get_layer("conv1_relu").output  # 112x112x64
    c2 = base_model.get_layer("conv2_block3_out").output  # 56x56x256
    c3 = base_model.get_layer("conv3_block4_out").output  # 28x28x512
    c4 = base_model.get_layer("conv4_block6_out").output  # 14x14x1024
    c5 = base_model.get_layer("conv5_block3_out").output  # 7x7x2048

    # Decoder: Upsampling blocks with LNN 
    # U6
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    u6 = BatchNormalization()(u6)
    u6 = LiquidLayer(512, time_steps=6, tau=0.1)(u6)
    u6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    u6 = BatchNormalization()(u6)

    # U7
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = concatenate([u7, c3])
    u7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    u7 = BatchNormalization()(u7)
    u7 = LiquidLayer(256, time_steps=6, tau=0.1)(u7)
    u7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    u7 = BatchNormalization()(u7)

    # U8
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = concatenate([u8, c2])
    u8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    u8 = BatchNormalization()(u8)
    u8 = LiquidLayer(128, time_steps=6, tau=0.1)(u8)
    u8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    u8 = BatchNormalization()(u8)

    # U9
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = concatenate([u9, c1])
    u9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    u9 = BatchNormalization()(u9)
    u9 = LiquidLayer(64, time_steps=6, tau=0.1)(u9)
    u9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    u9 = BatchNormalization()(u9)

    # U10
    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u9)
    u10 = Conv2D(32, (3, 3), activation='relu', padding='same')(u10)
    u10 = BatchNormalization()(u10)
    u10 = LiquidLayer(32, time_steps=6, tau=0.1)(u10)
    u10 = Conv2D(32, (3, 3), activation='relu', padding='same')(u10)
    u10 = BatchNormalization()(u10)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(u10)

    model = Model(inputs=[inputs], outputs=[outputs], name='LiqU-Net')
    return model

# -------------------------
# Load Datasets
# -------------------------
print("Loading datasets...")
train_dataset = load_all_datasets(dataset_paths, 'train', BATCH_SIZE)
val_dataset = load_all_datasets(dataset_paths, 'val', BATCH_SIZE)
print("Datasets loaded.")

# -------------------------
# Build, Compile, and Train 
# -------------------------
model = unet_model_with_liquid(input_size=(IMG_SIZE[0], IMG_SIZE[1], 3))
optimizer = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy', iou_metric])
model.summary()

# Callback: Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[reduce_lr]
)

# -------------------------
# Plot Training Curves
# -------------------------
def plot_accuracy_and_loss(history):
    """Plot training and validation accuracy/loss curves."""
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
    plt.savefig(os.path.join(output_folder, 'training_curves.png'))
    plt.show()

plot_accuracy_and_loss(history)

# Save the trained model
model.save(os.path.join(output_folder, 'liquid_U_Net_trained.keras'))
