import time
import cv2
import numpy as np
#from cvzone import HandTrackingModule
from HandTrackingModule import handDetector
import math
import tensorflow as tf
from tensorboard import summary
from tensorflow.keras.models import *
import ClassificationModule
import os
import gc
import tensorflow.keras.backend as K
import threading


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#passtest
cap = cv2.VideoCapture(0)
detector = handDetector(maxHands=1)
counter = 0
#Load the model; done at the beginning to prevent slow-downs in the loop
classifier = ClassificationModule.Classifier("Models/model11")

globalImage = None
classes = ["A", "B", "C"]
pred = None

def ShowVideo():
    while True:
        success, img = cap.read()
        data, img = detector.findHands(img)
        globalImage = img
        #cv2.putText(text)
        cv2.putText((img), str(classes[np.argmax(pred)]), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        


t1 = threading.Thread(target=ShowVideo)
t1.start()

while True:
    success, img = cap.read()
    data = None
    data, img = detector.findHands(img)
    bboffset = 20
    imgSize = 300
    if classifier.result is not None:
        prediction = classifier.result
        classes = ["А", "И'", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Б", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ю", "Я", "В", "Г", "Д", "E", "Ж", "З", "И"]
        print(np.argmax(prediction))
        print(classes[np.argmax(prediction)])
        print(prediction)
        pred = prediction

    if data:
        bbxmax, bbxmin, bbymax, bbymin = data["bbox"]
        w, h = bbxmax - bbxmin, bbymax - bbymin
        cropImg = img[bbymin - bboffset: bbymax + bboffset, bbxmin - bboffset: bbxmax + bboffset]
        if cropImg is not None and cropImg.size != 0:
          cv2.imshow("CropedImage", cropImg)

        imgCropShape = cropImg.shape
        imgWhite = np.ones([imgSize, imgSize, 3], np.uint8) * 255
        # print(imgCropShape)

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(cropImg, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil(((300 - wCal) / 2))
            imgWhite[:, wGap:wCal + wGap] = imgResize
            classifier.getPrediction(imgWhite)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(cropImg, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil(((300 - hCal) / 2))
            imgWhite[hGap:hCal + hGap, :] = imgResize
            classifier.getPrediction(imgWhite)


        cv2.imshow("WhiteImage", imgWhite)
        #cv2.imshow("Image", img)
        key = cv2.waitKey(1)
