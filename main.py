import cv2
import mediapipe
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])

    cv2.imshow("Image", img)
    cv2.waitKey(1)