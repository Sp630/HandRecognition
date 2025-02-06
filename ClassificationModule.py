import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np
from tensorflow.python.keras.saving.save import load_model


class Classifier:
    def __init__(self, modelPath):
        self.modelPath = modelPath
    def getPrediction(self, img):
        #img = cv2.imread(self.imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255
        img = np.expand_dims(img, axis=0)
        classes = ["A", "B", "C"]
        model = load_model(self.modelPath)
        prediction = model.predict(img)
        return prediction