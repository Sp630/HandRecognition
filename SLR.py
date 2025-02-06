import cv2
import numpy as np
import tensorflow
from tensorboard import summary
from tensorflow.keras.models import *

model = load_model("model1")
model.summary()
imgPath = "Data/BaseData/C/Image_1738721586.9495163.jpg"
img = cv2.imread(imgPath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img / 255
img = np.expand_dims(img, axis=0)
classes = ["A", "B", "C"]

prediction = model.predict(img)

print(classes[np.argmax(prediction)])
print(prediction)
