import sys
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
import tkinter as tk
from PIL import Image, ImageTk



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#threading synchronization
stop_event = threading.Event()
sharedData = None
dataLock = threading.Lock()


#videoCapture
cap = cv2.VideoCapture(0)
detector = handDetector(maxHands=1)
counter = 0

#GUI
#Tkinter
def CVtoTK(videoLabel, root, text):
    success, img = cap.read()
    if success:
        data, img = detector.findHands(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wholeImg = Image.fromarray(img)
        tkimage = ImageTk.PhotoImage(image = wholeImg)
        videoLabel.tkimage = tkimage
        videoLabel.config(image=tkimage)


        with dataLock:
            global sharedData
            print(sharedData)
            text.config(text= sharedData)

    root.after(10, lambda: CVtoTK(videoLabel, root, text))

def StartTkinter():
    root = tk.Tk()
    root.title("BGSLR")
    root.geometry("800x800")
    videoLabel = tk.Label(root)
    videoLabel.pack()
    text = tk.Label(root, font=("Arial", 30))
    text.pack(side="top", pady=10);

    quitButton = tk.Button(root,
                           text="Излез",
                           command= lambda: Quit(root),
                           font= ("Arial", 14),
                           width= 10,
                           height= 5
                           )
    quitButton.pack(side="bottom", pady=10)

    CVtoTK(videoLabel, root, text)
    root.mainloop()
def Quit(root):
    stop_event .set()

    #Ensure resources are properly released

    root.quit()
    root.destroy()
    sys.exit()

t2 = threading.Thread(target=StartTkinter)
t2.start()

#Load the model; done at the beginning to prevent slow-downs inside the loop
classifier = ClassificationModule.Classifier("Models/model11")

globalImage = None
classes = ["A", "B", "C"]
pred = None

#use this if you don't want GUI
def ShowVideo():


        success, img = cap.read()
        data, img = detector.findHands(img)
        globalImage = img
        #cv2.putText(text)
        cv2.putText((img), str(classes[np.argmax(pred)]), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        


#t1 = threading.Thread(target=ShowVideo, daemon= True)
#t1.start()



while not stop_event.isSet():
    success, img = cap.read()
    data = None
    data, img = detector.findHands(img)
    bboffset = 20
    imgSize = 300
    if classifier.result is not None:
        prediction = classifier.result
        classes = ["А", "И'", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Б", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ю", "Я", "В", "Г", "Д", "E", "Ж", "З", "И"]
        #print(np.argmax(prediction))
        #print(classes[np.argmax(prediction)])
        #print(prediction)
        sharedData = classes[np.argmax(prediction)]
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
cap.release()
