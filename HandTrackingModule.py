import mediapipe as mp
import cv2
import time

from tensorflow.python.keras.activations import relu6

#this class is responsible for finding hands and outputting information about their landmarks
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, handNo = 0, draw=True):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        lmList = []
        xList = []
        yList = []
        bboxList = []
        totalHand = {}
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.results.multi_hand_landmarks[handNo].landmark):

                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                lmList.append([cx, cy, cz])
                xList.append(cx)
                yList.append(cy)

                #bbox
                xmax, xmin = max(xList), min(xList)
                ymax, ymin = max(yList), min(yList)
                bw = xmax - xmin
                bh = ymax - ymin
                bbox = [xmax, xmin, ymax, ymin]
                totalHand["lm"] = lmList
                totalHand["bbox"] = bbox

            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    offset = 10
                    cv2.rectangle(img, [xmin - offset, ymin - offset], [xmax + offset, ymax + offset], (255, 0, 0), 3)
        #totalHand["lm"] = [0, 1, 2]
        return  totalHand, img

    def findPosition(self, img, handNo=0, draw = False):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if(draw):
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList

#testing
def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)
    while True:
        success, img = cap.read()
        data, img = detector.findHands(img)
        if "lm" in data:
            print(data["lm"][1])

        cv2.imshow("Image", img)
        cv2.waitKey(1)




        cv2.imshow("Image", img)



if __name__ == "__main__":
    main()


