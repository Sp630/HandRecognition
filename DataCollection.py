import cv2
#from cvzone import HandTrackingModule
from HandTrackingModule import handDetector

cap = cv2.VideoCapture(0)
detector = handDetector(maxHands=1)
while True:
    success, img = cap.read()
    data, img = detector.findHands(img)
    bboffset = 20
    if data:
        bbxmax, bbxmin, bbymax, bbymin = data["bbox"]
        cropImg = img[bbymin - bboffset : bbymax + bboffset, bbxmin - bboffset : bbxmax + bboffset]
        if cropImg is not None and cropImg.size != 0:
          cv2.imshow("CropedImage", cropImg)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
