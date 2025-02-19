import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define the path to the single image
image_path = r"C:\Users\HP\Desktop\Image reconstruction\Dataset\4 no.jpg"
img_size = 256

# Load and preprocess the single image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0  # Normalize the image to [0, 1]
        return np.expand_dims(img, axis=0)  # Add batch dimension
    else:
        raise FileNotFoundError(f"Image not found at {image_path}")

# Load the test image
try:
    test_image = load_and_preprocess_image(image_path)
except FileNotFoundError as e:
    print(e)
    exit()

# Load the trained model
autoencoder = load_model("cae_model.h5")
print("Model loaded successfully.")

# Predict the reconstructed image
predicted_image = autoencoder.predict(test_image)

# Display original and reconstructed images
def visualize_results(test_image, reconstructed_image):
    plt.figure(figsize=(10, 4))

    # Original
    ax = plt.subplot(1, 2, 1)
    plt.imshow(test_image[0].reshape(img_size, img_size, 3))
    plt.axis("off")
    ax.set_title("Original")

    # Reconstructed
    ax = plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image[0].reshape(img_size, img_size, 3))
    plt.axis("off")
    ax.set_title("Reconstructed")

    plt.show()

# Visualize the result
visualize_results(test_image, predicted_image)
