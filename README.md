# animal-image-classifier-ml
A machine learning model that classifies animal images using traditional ML techniques like HOG, LBP, and Color Histograms, trained on a Random Forest Classifier with PCA. All features are extracted using OpenCV and custom code
# 🐾 Animal Image Classifier using Traditional ML

This project builds a machine learning model to classify animal images using handcrafted features like HOG (shape), LBP (texture), and Color Histograms. It uses a Random Forest classifier and Principal Component Analysis (PCA) for dimensionality reduction.

## 🔍 Features Extracted
- **HOG (Histogram of Oriented Gradients)** – shape and edges
- **LBP (Local Binary Pattern)** – texture features (custom implemented)
- **Color Histograms** – color distribution across RGB channels

## 🧠 Model Used
- **Random Forest Classifier**
- **PCA** for dimensionality reduction
- **Label Encoder** for class labels

## 🧪 Evaluation
- Accuracy and classification report are generated on test data
- Confusion matrix for visualizing performance

## 🗂️ Dataset Structure
Your dataset folder should look like this:
