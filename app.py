from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Custom loss ===
def combined_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.5 * ssim_loss + 0.5 * l1_loss

# === Load models ===
brain_model = load_model('models/autoencoder8_brain_model.h5', custom_objects={'combined_loss': combined_loss})
kidney_model = load_model('models/bestkidney_model.h5', custom_objects={'combined_loss': combined_loss})

brain_classifier = load_model('models/best_model1.h5')
kidney_classifier = load_model('models/kidney_classifier_model.h5')

brain_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
kidney_labels = ['no tumor', 'tumor']

# === Preprocessing for reconstruction ===
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 127.5 - 1
    return np.expand_dims(img, axis=0)

# === Preprocessing for classification ===
def preprocess_for_classification(img_path, organ):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if organ == 'brain':
        img = cv2.resize(img, (240, 240))  # trained on 240x240
    elif organ == 'kidney':
        img = cv2.resize(img, (256, 256))  # trained on 256x256

    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


# === Enhance & Sharpen ===
def enhance_grayscale_sharpen(image):
    image = np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bright = clahe.apply(gray)
    bright = np.clip(bright * 1.15, 0, 255).astype(np.uint8)
    sharpened = cv2.filter2D(bright, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/<organ>')
def upload_page(organ):
    return render_template('upload.html', organ=organ)

@app.route('/process/<organ>', methods=['POST'])
def process_image(organ):
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    original_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(original_path)

    # Reconstruction
    if organ == 'brain':
        model = brain_model
    elif organ == 'kidney':
        model = kidney_model
    else:
        return jsonify({'error': 'Unsupported organ'}), 400

    img_input = preprocess_image(original_path)
    reconstructed = model.predict(img_input)
    enhanced = enhance_grayscale_sharpen(reconstructed[0])

    recon_filename = f'recon_{file.filename}'
    recon_path = os.path.join(UPLOAD_FOLDER, recon_filename)
    cv2.imwrite(recon_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

    # Classification
    input_for_cls = preprocess_for_classification(original_path, organ)
    if organ == 'brain':
        pred = brain_classifier.predict(input_for_cls)
        prediction = brain_labels[np.argmax(pred)]
    elif organ == 'kidney':
        pred = kidney_classifier.predict(input_for_cls)

        prediction = kidney_labels[int(np.round(pred[0][0]))]

    return jsonify({
        'original': url_for('static', filename=f'uploads/{file.filename}'),
        'denoised': url_for('static', filename=f'uploads/{recon_filename}'),
        'prediction': prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
