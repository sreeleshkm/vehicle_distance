# Importing Libraries
import cv2
import dlib
import time
import math

# Classifier File (Haar Cascade)
carCascade = cv2.CascadeClassifier("vech.xml")

# Video file capture
video = cv2.VideoCapture("carsVideo2.mp4")

# Constant Declaration
WIDTH = 1280
HEIGHT = 720

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

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()
        frameCounter += 1

        carIDtoDelete = []

        # --- Remove Lost / Unmatched Objects ---
        for carID in list(carTracker.keys()):
            trackingQuality = carTracker[carID].update(image)

            pos = carTracker[carID].get_position()
            # If quality is low OR tracker is out of frame → mark for deletion
            if trackingQuality < 7 or pos.left() < 0 or pos.top() < 0 or pos.right() > WIDTH or pos.bottom() > HEIGHT:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print(f"Removing carID {carID}")
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)
            speed[carID] = None

        # --- Detect new cars every 10 frames ---
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    tracked_pos = carTracker[carID].get_position()
                    t_x, t_y = int(tracked_pos.left()), int(tracked_pos.top())
                    t_w, t_h = int(tracked_pos.width()), int(tracked_pos.height())
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and 
                        (t_y <= y_bar <= (t_y + t_h)) and 
                        (x <= t_x_bar <= (x + w)) and 
                        (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                # Create new tracker if unmatched
                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        # --- Draw tracked objects ---
        for carID in carTracker.keys():
            tracked_pos = carTracker[carID].get_position()
            t_x, t_y = int(tracked_pos.left()), int(tracked_pos.top())
            t_w, t_h = int(tracked_pos.width()), int(tracked_pos.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

            # Distance Display
            distance = estimateDistance(t_w)
            cv2.putText(resultImage, f"{distance:.2f} m", (t_x, t_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Speed Calculation ---
        # for i in carLocation1.keys():
        #     if frameCounter % 1 == 0:
        #         [x1, y1, w1, h1] = carLocation1[i]
        #         [x2, y2, w2, h2] = carLocation2.get(i, [0, 0, 0, 0])
        #         carLocation1[i] = [x2, y2, w2, h2]

        #         if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
        #             if (speed[i] is None or speed[i] == 0) and 275 <= y1 <= 285:
        #                 speed[i] = estimateSpeed([x1, y1, w1, h1], [x1, y2, w2, h2])

        #             if speed[i] is not None and y1 >= 180:
        #                 cv2.putText(resultImage, str(int(speed[i])) + " km/h",
        #                             (int(x1 + w1 / 2), int(y1 - 25)),
        #                             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # --- Show result ---
        cv2.imshow('result', resultImage)
        out.write(resultImage)

        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

    cv2.destroyAllWindows()
    out.release()

if __name__ == '__main__':
    trackMultipleObjects()
