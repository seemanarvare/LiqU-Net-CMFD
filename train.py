import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from preprocessing import load_datasets, IMG_SIZE, BATCH_SIZE, SEED, OUTPUT_FOLDER

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
        inputs_reshaped = tf.reshape(inputs, [B, HW, C])
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

def unet_model_with_liquid(input_size=(224, 224, 3)):
    inputs = Input(input_size)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    for layer in base_model.layers:
        layer.trainable = False

    c1 = base_model.get_layer("conv1_relu").output
    c2 = base_model.get_layer("conv2_block3_out").output
    c3 = base_model.get_layer("conv3_block4_out").output
    c4 = base_model.get_layer("conv4_block6_out").output
    c5 = base_model.get_layer("conv5_block3_out").output

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    u6 = BatchNormalization()(u6)
    u6 = LiquidLayer(512, time_steps=6, tau=0.1)(u6)
    u6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    u6 = BatchNormalization()(u6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = concatenate([u7, c3])
    u7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    u7 = BatchNormalization()(u7)
    u7 = LiquidLayer(256, time_steps=6, tau=0.1)(u7)
    u7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    u7 = BatchNormalization()(u7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = concatenate([u8, c2])
    u8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    u8 = BatchNormalization()(u8)
    u8 = LiquidLayer(128, time_steps=6, tau=0.1)(u8)
    u8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    u8 = BatchNormalization()(u8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = concatenate([u9, c1])
    u9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    u9 = BatchNormalization()(u9)
    u9 = LiquidLayer(64, time_steps=6, tau=0.1)(u9)
    u9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    u9 = BatchNormalization()(u9)

    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u9)
    u10 = Conv2D(32, (3, 3), activation='relu', padding='same')(u10)
    u10 = BatchNormalization()(u10)
    u10 = LiquidLayer(32, time_steps=6, tau=0.1)(u10)
    u10 = Conv2D(32, (3, 3), activation='relu', padding='same')(u10)
    u10 = BatchNormalization()(u10)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(u10)
    model = Model(inputs=[inputs], outputs=[outputs], name='LiqU-Net')
    return model

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def iou_metric(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

def train_model():
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("Loading datasets...")
    train_datasets = load_datasets('train')
    val_datasets = load_datasets('val')
    print("Datasets loaded.")

    # Combine all training datasets
    combined_train_dataset = None
    for dataset_name, dataset in train_datasets.items():
        if combined_train_dataset is None:
            combined_train_dataset = dataset
        else:
            combined_train_dataset = combined_train_dataset.concatenate(dataset)
    
    # Combine all validation datasets
    combined_val_dataset = None
    for dataset_name, dataset in val_datasets.items():
        if combined_val_dataset is None:
            combined_val_dataset = dataset
        else:
            combined_val_dataset = combined_val_dataset.concatenate(dataset)

    if combined_train_dataset is None or combined_val_dataset is None:
        raise ValueError("No valid datasets found for training or validation")

    print("Training on combined dataset...")
    model = unet_model_with_liquid(input_size=(IMG_SIZE[0], IMG_SIZE[1], 3))
    optimizer = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy', iou_metric])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

    history = model.fit(
        combined_train_dataset,
        validation_data=combined_val_dataset,
        epochs=50,
        callbacks=[reduce_lr]
    )

    # Save final model weights
    model.save(os.path.join(OUTPUT_FOLDER, 'liquid_U_Net_50_epoch.keras'))

    return history, model

if __name__ == "__main__":
    history, model = train_model()
