import cv2
import logging as log
import datetime as dt

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
width, height = int(video_capture.get(3)), int(video_capture.get(4))
# Use mp4v codec for proper MP4 file format
out = cv2.VideoWriter("1.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
anterior = 0

known_distance1 = 4.3
known_width1 = 48

focalLength = known_distance1 * known_width1

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles and put distance text below each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        distance = focalLength / w
        text_position = (x, y + h + 20)
        cv2.putText(frame, "%.2fM" % distance,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

    # Write frame to video file
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release resources
video_capture.release()
out.release()
cv2.destroyAllWindows()
