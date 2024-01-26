import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.modelComp = model_complexity
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode, max_num_hands, model_complexity, min_detection_confidence, min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, points=True, connections=False):
        conn_flag = self.mpHands.HAND_CONNECTIONS if connections else None
            
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if points:
                    self.mpDraw.draw_landmarks(
                        img, handLms, conn_flag
                    )
        return img

    def findPosition(self, img, handNo=0, circles=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if circles:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img, points=True, draw_connections=True)
        lmList = detector.findPosition(img, circles=False)
        # if len(lmList) != 0:
        #     print(lmList[4])
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
