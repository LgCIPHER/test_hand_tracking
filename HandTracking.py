import cv2 as cv
import mediapipe as mp
import time

# Frame rate
pTime = 0
cTime = 0
    
# Activate webcam
cap = cv.VideoCapture(0)

# Hand module from "mp"
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
 
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 4:
                    print(id, cx, cy)
                    cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)
 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    fps_display = "FPS: " + str(int(fps))

    cv.putText(img, fps_display, (10, 50), cv.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
 
    cv.imshow("Image", img)
    cv.waitKey(1)