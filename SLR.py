import cv2
import numpy as np
import tensorflow
from tensorboard import summary
from tensorflow.keras.models import *

model = load_model("Models/model7")
model.summary()
imgPath = "Data/Bulgarian/19/Image_1739921920.4461687.jpg"
img = cv2.imread(imgPath)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#img = img / 255
img = np.expand_dims(img, axis=0)
classes = ["А", "И'", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Б", "У", "В", "Г", "Д", "E", "Ж", "З", "И"]
#classes = ["А", "И'", "Б", "В", "Г", "Д", "E", "Ж", "З", "И"]

prediction = model.predict(img)

print(classes[np.argmax(prediction)])
print(prediction)
