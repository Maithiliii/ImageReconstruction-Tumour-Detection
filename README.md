# 🧠 Image Reconstruction & Tumor Detection using Deep Learning

## 📌 Project Overview

This project enhances MRI scans and detects tumors using a two-phase deep learning pipeline:

- **Image Reconstruction**: Uses a convolutional autoencoder to denoise and sharpen MRI images.
- **Tumor Detection**: Applies a CNN classifier to detect tumors from reconstructed images.

By improving image quality first, the system helps in making more accurate diagnoses.

---

## 🧬 Abstract Summary

MRI scans are critical for early detection of brain tumors, but their quality is often reduced due to noise or low resolution. This project introduces a deep learning pipeline that combines:

- **Convolutional Autoencoders**: To reconstruct high-resolution MRI images by learning compact, meaningful representations of input data.
- **Convolutional Neural Networks (CNNs)**: To classify the reconstructed images as tumor or non-tumor.

This improves clarity before classification, leading to more reliable diagnostic outcomes.

---

## 📁 Directory Structure
ImageReconstruction-Tumour-Detection/  
├── app.py  
├── requirements.txt   
├── Dockerfile  
├── .dockerignore    
├── brainclassify.ipynb  
├── kidneyclassify.ipynb  
├── braintrain.ipynb  
├── kidneytrain.ipynb  
├── models/  
│ └── *.h5 (model files or download.txt with links)  
├── static/  
│ ├── app.js  
│ ├── style.css  
│ ├── images/  
│ └── uploads/ # (Create this folder manually)  
├── templates/  
│ ├── index.html  
│ └── upload.html  


---

## ⚙️ How to Run

### ✅ Option 1: Use Pretrained Models

- Download `.h5` model files from the link mentioned in `models/download.txt`.
- Place them inside the `models/` folder.

### 🧠 Option 2: Train Models Yourself

Run these notebooks to train and save models:

- `braintrain.ipynb` – Brain MRI autoencoder
- `kidneytrain.ipynb` – Kidney MRI autoencoder
- `brainclassify.ipynb` – Brain tumor CNN classifier
- `kidneyclassify.ipynb` – Kidney tumor CNN classifier

Saved models will automatically go inside the `models/` folder.

---

### 🚀 Run the Web App

1. Make sure the folder `static/uploads/` exists (create if missing).
2. In the terminal, run:
   ```bash
   python app.py

3. Go to http://localhost:5000 in your browser.  

