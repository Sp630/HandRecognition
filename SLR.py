#a file which tests the model on already collected images
#intended to be used before real time testing in case of wierd results
#it was used to debug differences in the way data was processed before training and then before testing
import cv2
import numpy as np
import tensorflow
from tensorboard import summary
from tensorflow.keras.models import *

model = load_model("Models/model8")
model.summary()
imgPath = "Data/Testing captures/Image_1740244592.80069.jpg"
img = cv2.imread(imgPath)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#img = img / 255
img = np.expand_dims(img, axis=0)
classes = ["А", "И'", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Б", "У", "В", "Г", "Д", "E", "Ж", "З", "И"]
#classes = ["А", "И'", "Б", "В", "Г", "Д", "E", "Ж", "З", "И"]

prediction = model.predict(img)

print(classes[np.argmax(prediction)])
print(prediction)
