import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
width, height = int(video_capture.get(3)), int(video_capture.get(4))
out = cv2.VideoWriter("1.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15.0, (width, height))
anterior = 0

known_distance1 = 4.3
known_width1 = 48

known_distance2 = 2.2
known_width2 = 107

focalLength = known_distance1*known_width1

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
    if len(faces) > 0:
        cv2.putText(frame, "%.2fM" % (focalLength/faces[0][2]),
            (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 0), 2)

    # Display the resulting frame
    out.write(frame)
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()