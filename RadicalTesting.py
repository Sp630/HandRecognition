#this file is for testing purposes
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model("Models/model5")
model.summary()

# Load and preprocess image
imgPath = "Data/Bulgarian/10/Image_1739741126.977221.jpg"
img = cv2.imread(imgPath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize image to match model input size (modify accordingly)

# Normalize and expand dimensions
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# Define class labels
classes = ["А", "И'", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Б", "У", "В", "Г", "Д", "E", "Ж", "З", "И"]

# Make prediction
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

# Output result
print(f"Predicted class: {classes[predicted_class]}")
print(f"Prediction probabilities: {prediction}")