from flask import Flask, render_template, request, send_from_directory
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("cae_model.h5")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

img_size = 256

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Serve CSS & JS from templates folder
@app.route('/<path:filename>')
def serve_files(filename):
    return send_from_directory('templates', filename)

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Process the image with CAE
            test_image = preprocess_image(img_path)
            predicted_image = model.predict(test_image)[0]

            # Convert predicted image back to BGR for OpenCV display
            reconstructed_img = (predicted_image * 255).astype(np.uint8)
            reconstructed_path = os.path.join(UPLOAD_FOLDER, "reconstructed.png")
            cv2.imwrite(reconstructed_path, cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR))

            return render_template("first.html", original=img_path, reconstructed=reconstructed_path)

    return render_template("first.html", original=None, reconstructed=None)

if __name__ == "__main__":
    app.run(debug=True)
