# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:09:04 2023

Author: ASUS
Description: This script defines, trains, and evaluates an autoencoder model for image compression using TensorFlow and Keras.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, LayerNormalization, UpSampling2D, Reshape, Conv2DTranspose, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from tensorflow.image import psnr
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# Callback to track time taken for each epoch
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# Load datasets
X_train = np.load('X_train_ORG.npy', mmap_mode='r')
X_val = np.load('X_val_ORG.npy', mmap_mode='r')
X_test = np.load('dataset_test.npy', mmap_mode='r')

# Custom PSNR loss function
def psnr_loss(y_true, y_pred):
    return psnr(y_true, y_pred, max_val=255.0)

# Define the autoencoder model
input_layer = Input(shape=(64, 64, 64), name="input_layer")

# Encoder
x = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=l2(0.001))(input_layer)
x = LayerNormalization()(x)
x = Dropout(0.3)(x)  # Regularization

x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
x1 = Conv2D(64, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x1)

x2 = Conv2D(64, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x2 = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x2)

x3 = Conv2D(64, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x3)

x4 = Conv2D(64, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)

x = concatenate([x1, x2, x3, x4], axis=-1)
x = LayerNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.3)(x)  # Regularization
encoded_output = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)

# Decoder
x = Dense(64 * 64 * 64, activation='relu', kernel_regularizer=l2(0.001))(encoded_output)
x = Reshape((64, 64, 64))(x)

x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
x1 = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x1)

x2 = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x2 = Conv2DTranspose(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x2)

x3 = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x3 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x3)

x4 = Conv2DTranspose(64, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)

x = concatenate([x1, x2, x3, x4], axis=-1)
x = LayerNormalization()(x)
x = Dropout(0.3)(x)  # Regularization
x = Conv2DTranspose(64, (1, 1), activation='relu', kernel_regularizer=l2(0.001))(x)
decoded_output = Conv2DTranspose(64, (1, 1))(x)

autoencoder_model = Model(input_layer, decoded_output)

# Compile the model
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
autoencoder_model.compile(optimizer=optimizer, loss='mse', metrics=[psnr_loss])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir='./logs')

# Train the model
start_time = time.time()

history = autoencoder_model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, checkpoint, tensorboard]
)

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# Evaluate the model
best_model = tf.keras.models.load_model('best_model.h5', custom_objects={'psnr_loss': psnr_loss})
evaluation = best_model.evaluate(X_test, X_test)
print(f"Evaluation results: {evaluation}")

# Plotting results
acc = history.history['psnr_loss']
val_acc = history.history['val_psnr_loss']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training PSNR Loss')
plt.plot(epochs, val_acc, 'r', label='Validation PSNR Loss')
plt.title('Training and Validation PSNR Loss')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
