import cv2
import mediapipe
from cvzone import ClassificationModule

cap = cv2.VideoCapture(0)
classifier = ClassificationModule.Classifier

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])

    cv2.imshow("Image", img)
    cv2.waitKey(1)