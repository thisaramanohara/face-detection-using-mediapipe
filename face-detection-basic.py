import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture('smile.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow("Image", img)

    toClose = cv2.waitKey(1)
    if toClose==27:
        break

cv2.destroyAllWindows()
cap.release()

