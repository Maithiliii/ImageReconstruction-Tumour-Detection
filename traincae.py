import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Add
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define the path to the dataset folder
dataset_dir = r"C:\Users\HP\Desktop\Image reconstruction\Dataset"
img_size = 256

# Load and preprocess data
def load_images_from_folder(folder):
    images = []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"The folder '{folder}' does not exist. Please check the path.")
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img_to_array(img))
    return images

# Load all images from the dataset directory
try:
    img_data = load_images_from_folder(dataset_dir)
except FileNotFoundError as e:
    print(e)
    exit()

# Normalize and convert data to numpy array
img_data = np.array(img_data, dtype='float32') / 255.0

# Split data into training and testing sets
x_train, x_test = train_test_split(img_data, test_size=0.2, random_state=42)

# Build the autoencoder model with batch normalization and skip connections
def build_autoencoder(input_shape=(256, 256, 3)):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x1 = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x = BatchNormalization()(x)
    x2 = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder with Skip Connections
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    # Adjust channel size to match x2 for addition
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = Add()([x, x2])

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    # Adjust channel size to match x1 for addition
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = Add()([x, x1])

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    # Change the activation to 'linear' for output
    decoded = Conv2D(3, (3, 3), activation='linear', padding='same')(x)

    # Compile Model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return autoencoder

# Build and summarize the model
autoencoder = build_autoencoder()
autoencoder.summary()

# Train the model
autoencoder.fit(x_train, x_train, epochs=100, batch_size=32, shuffle=True, validation_data=(x_test, x_test))

# Save the model
autoencoder.save("cae_model.h5")
print("Model saved as 'cae_model.h5'")
