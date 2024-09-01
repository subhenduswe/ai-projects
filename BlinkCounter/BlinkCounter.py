import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture('Video1.mp4')
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640,360,[20,50], invert=True)

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 0)

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    Success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 4, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)

        #cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        #cv2.line(img, leftLeft, leftRight, (0,200,0), 3)

        ratio = (lengthVer/lengthHor)*100

        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 35 and counter == 0:
            blinkCounter +=1
            color = (0,0,255)
            counter = 1
        if counter != 0:
            counter +=1
            if counter > 10:
                counter = 0
                color = (255, 0, 0)

        cvzone.putTextRect(img,f'Blink Count: {blinkCounter}',(50,100),
                           colorR=color)

        imgPlot = plotY.update(ratioAvg,color)
        img = cv2.resize(img, (640, 840))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 840))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    cv2.waitKey(25)
