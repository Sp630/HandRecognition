import tflite_runtime as tflite
from tflite_runtime.interpreter import Interpreter
from tensorflow.keras import models
import cv2
import numpy as np
from tensorflow.python.keras.saving.save import load_model
import threading

model = tflite



tflite_model = Interpreter(model_path="Models/Android/")


exit()

class Classifier:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.result = None
        self.lock = threading.Lock()
        self.model = load_model(self.modelPath)

    def multithreadPredict(self, img):
        t1 = threading.Thread(target=self.getPrediction, args=(img, ))
        t1.start()
        #print(self.result)
    def getPrediction(self, img):
        #img = cv2.imread(self.imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ##img = img / 255
        img = np.expand_dims(img, axis=0)
        classes = ["A", "B", "C"]
        prediction = self.model.predict(img)
        self.result = 14
        with self.lock:
            self.result = prediction