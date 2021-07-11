import time
import cv2
import numpy as np
import os

import Tracking as T

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#getting header files 
header_files = 'colors'
header_list = os.listdir(header_files)
overlay = []
for imgPath in header_list:
    img = cv2.imread(f'{header_files}/{imgPath}')
    overlay.append(img)

header = overlay[0]

brush_thickness = 22
eraser_thickness = 70
drawColor = (30, 255, 255)
imageCanvas = np.zeros((720, 1280, 3), np.uint8)

detector = T.handDetector(min_detection_confidence=0.85)
xp, yp = 0, 0


cT=0
pT =0

while True:
    success, image = cap.read()
    image = cv2.flip(image,1)

    #detect the hand and landmarks
    image = detector.findHands(image)
    landmarks_list = detector.findPosition(image, draw=False)

    if len(landmarks_list) != 0: #if there are landmarks detected
        #storing tips of index and middle fingers
        xi, yi = landmarks_list[8][1:]
        xm, ym = landmarks_list[12][1:]

        fingers = detector.fingersUp()
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(image, (xi, yi - 25), (xm, ym + 25), drawColor, cv2.FILLED)
            #changing header according to position
            if yi < 125:
                if 250 < xi < 450:
                    header = overlay[0]
                    drawColor = (30, 255, 255)
            
                elif 550 < xi < 750:
                    header = overlay[1]
                    drawColor = (0, 255, 0)
                    
                elif 800 < xi < 950:
                    header = overlay[2]
                    drawColor = (255, 0, 255)
                    
                elif 1050 < xi < 1200:
                    header = overlay[3]
                    drawColor = (0, 0, 0)
                    
        if fingers[1] and fingers[2] == False:
            cv2.circle(image, (xi, yi), 15, drawColor, cv2.FILLED)
            
            #for the first image
            if xp == 0 and yp == 0:
                xp, yp = xi, yi

            cv2.line(image, (xp, yp), (xi, yi), drawColor, brush_thickness)

            if drawColor == (0, 0, 0):
                cv2.line(image, (xp, yp), (xi, yi), drawColor, eraser_thickness)
                cv2.line(imageCanvas, (xp, yp), (xi, yi), drawColor, eraser_thickness)
            
            else:
                cv2.line(image, (xp, yp), (xi, yi), drawColor, brush_thickness)
                cv2.line(imageCanvas, (xp, yp), (xi, yi), drawColor, brush_thickness)

            xp, yp = xi, yi


        # when all fingers are up canvas will clear
        if all (x >= 1 for x in fingers):
            mgCanvas = np.zeros((720, 1280, 3), np.uint8)

    img_gray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
    _, imginv= cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)

    image = cv2.bitwise_and(image, imginv)
    image =cv2.bitwise_or(image, imageCanvas)
    cT = time.time()
    fps = 1/(cT- pT)
    pT = cT

    cv2.putText(image, 'FPS:' + str(int(fps)), (8,650), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(100, 255, 0), thickness=3)

    image[0:125, 0:1280] = header
    cv2.imshow('paint', image)
    cv2.waitKey(1)