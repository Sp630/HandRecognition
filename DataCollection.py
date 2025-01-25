import cv2
#from cvzone import HandTrackingModule
from HandTrackingModule import handDetector

cap = cv2.VideoCapture(0)
detector = handDetector(maxHands=1)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
