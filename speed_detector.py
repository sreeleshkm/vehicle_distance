import cv2
import dlib
import time
import math

# Classifier File (Haar Cascade)
carCascade = cv2.CascadeClassifier("vech.xml")

# Video file capture
video = cv2.VideoCapture("carsVideo4.mp4")

# Constants for YouTube Shorts size (vertical video)
WIDTH = 600
HEIGHT = 800

# Known width of vehicle in meters (average)
KNOWN_WIDTH = 1.8  
# Focal length of the camera in pixels (calibrate this value)
FOCAL_LENGTH = 700  

# Speed Estimation
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + 
                         math.pow(location2[1] - location1[1], 2))
    ppm = 8.8  # pixels per meter
    d_meters = d_pixels / ppm
    fps = 18  # frames per second
    speed = d_meters * fps * 3.6  # km/h
    return speed

# Distance Estimation
def estimateDistance(width_in_pixels):
    if width_in_pixels == 0:
        return 0
    return (KNOWN_WIDTH * FOCAL_LENGTH) / width_in_pixels

# IoU based overlapping check to avoid multiple trackers for same object
def is_overlapping(x, y, w, h, t_x, t_y, t_w, t_h, threshold=0.6):
    overlap_x1 = max(x, t_x)
    overlap_y1 = max(y, t_y)
    overlap_x2 = min(x + w, t_x + t_w)
    overlap_y2 = min(y + h, t_y + t_h)

    overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
    area = w * h
    t_area = t_w * t_h

    union_area = area + t_area - overlap_area
    if union_area == 0:
        return False
    iou = overlap_area / union_area
    return iou > threshold

# Tracking multiple objects
def trackMultipleObjects():
    rectangleColor = (0, 255, 255)
    frameCounter = 0
    currentCarID = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter('outTraffic.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    start_time = time.time()  # Timer for resetting trackers every 10 seconds

    while True:
        rc, image = video.read()
        if type(image) == type(None):
            break

        # Resize frame to YouTube Shorts size (vertical)
        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()
        frameCounter += 1

        # Reset all trackers and data every 10 seconds
        if time.time() - start_time > 10:
            carTracker.clear()
            carLocation1.clear()
            carLocation2.clear()
            speed = [None] * 1000
            start_time = time.time()

        carIDtoDelete = []

        # Remove Lost / Unmatched Objects
        for carID in list(carTracker.keys()):
            trackingQuality = carTracker[carID].update(image)

            pos = carTracker[carID].get_position()
            # Conditions to mark for deletion
            if (trackingQuality < 7 or
                pos.left() < 0 or pos.top() < 0 or
                pos.right() > WIDTH or pos.bottom() > HEIGHT):
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print(f"Removing carID {carID}")
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)
            speed[carID] = None

        # Detect new cars every 10 frames
        if frameCounter % 10 == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                matchCarID = None

                # Check if this detection overlaps significantly with an existing tracker
                for carID in carTracker.keys():
                    tracked_pos = carTracker[carID].get_position()
                    t_x, t_y = int(tracked_pos.left()), int(tracked_pos.top())
                    t_w, t_h = int(tracked_pos.width()), int(tracked_pos.height())

                    if is_overlapping(x, y, w, h, t_x, t_y, t_w, t_h, threshold=0.6):
                        matchCarID = carID
                        break

                # If no match found, create new tracker
                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        # Draw tracked objects & Car ID
        for carID in carTracker.keys():
            tracked_pos = carTracker[carID].get_position()
            t_x, t_y = int(tracked_pos.left()), int(tracked_pos.top())
            t_w, t_h = int(tracked_pos.width()), int(tracked_pos.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            # Draw carID at the top of the box
            cv2.putText(resultImage, f"ID {carID}", (t_x, t_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            carLocation2[carID] = [t_x, t_y, t_w, t_h]

            # Distance Display
            distance = estimateDistance(t_w)
            cv2.putText(resultImage, f"{distance:.2f} m", (t_x, t_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show result and write output video
        cv2.imshow('result', resultImage)
        out.write(resultImage)

        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()
    out.release()

if __name__ == '__main__':
    trackMultipleObjects()
