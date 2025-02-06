import time
import cv2
import numpy as np
#from cvzone import HandTrackingModule
from HandTrackingModule import handDetector
import math
import tensorflow
from tensorboard import summary
from tensorflow.keras.models import *
import ClassificationModule

cap = cv2.VideoCapture(0)
detector = handDetector(maxHands=1)

counter = 0
classifier = ClassificationModule.Classifier("model1")
while True:
    success, img = cap.read()
    data, img = detector.findHands(img)
    bboffset = 20
    imgSize = 300


    if data:
        bbxmax, bbxmin, bbymax, bbymin = data["bbox"]
        w, h = bbxmax - bbxmin, bbymax - bbymin
        cropImg = img[bbymin - bboffset : bbymax + bboffset, bbxmin - bboffset : bbxmax + bboffset]
        if cropImg is not None and cropImg.size != 0:
          cv2.imshow("CropedImage", cropImg)

        imgCropShape = cropImg.shape
        imgWhite  = np.ones([imgSize, imgSize, 3], np.uint8)*255
        #print(imgCropShape)

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(cropImg, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil(((300-wCal) / 2))
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction = classifier.getPrediction(imgWhite)
            print(prediction)
        else:
            k=imgSize/w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(cropImg, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil(((300 - hCal) / 2))
            imgWhite[hGap:hCal+hGap, :] = imgResize


        cv2.imshow("WhiteImage", imgWhite)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

