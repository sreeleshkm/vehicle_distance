import os
os.environ["OPENCV_IO_ENABLE_JASPER"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "FFMPEG"
os.environ["OPENCV_VIDEOIO_USE_GPU"] = "0"
os.environ["OPENCV_OCL_RUNTIME"] = ""
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"

import numpy as np
import subprocess
import json
import time
import math
import cv2

# Configure OpenCV to use minimal features
cv2.setNumThreads(1)
if hasattr(cv2, 'ocl'):
    cv2.ocl.setUseOpenCL(False)

try:
    # Classifier File (Haar Cascade)
    carCascade = cv2.CascadeClassifier("vech.xml")
    if carCascade.empty():
        print("Error: Could not load classifier file 'vech.xml'")
        exit(1)
except Exception as e:
    print("Error loading classifier:", e)
    exit(1)

# Vehicle tracking using centroid-based approach
class VehicleTracker:
    def __init__(self, detection, car_id):
        self.car_id = car_id
        self.bbox = detection
        self.time_since_update = 0
        self.hits = 0
        self.history = []
        self.update(detection)
    
    def update(self, detection):
        self.bbox = detection
        self.time_since_update = 0
        self.hits += 1
        self.history.append(detection)
        if len(self.history) > 30:  # Keep only last 30 frames
            self.history.pop(0)
        return True

    def get_state(self):
        return self.bbox

    def get_centroid(self):
        x, y, w, h = self.bbox
        return (int(x + w/2), int(y + h/2))

def calculate_distance(centroid1, centroid2):
    return np.sqrt(pow(centroid1[0] - centroid2[0], 2) + pow(centroid1[1] - centroid2[1], 2))

def match_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return [], detections

    matches = []
    unmatched_detections = detections.copy()
    
    for tracker_id, tracker in trackers.items():
        tracker_bbox = tracker.get_state()
        best_iou = iou_threshold
        best_detection = None
        
        for detection in unmatched_detections:
            iou = calculate_iou(detection, tracker_bbox)
            if iou > best_iou:
                best_iou = iou
                best_detection = detection
        
        if best_detection is not None:
            matches.append((best_detection, tracker_id))
            unmatched_detections.remove(best_detection)
    
    return matches, unmatched_detections

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0
        
    return intersection_area / union_area

# Video file capture with minimal features
def create_video_capture(filename):
    try:
        cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # Fallback to minimal configuration
            cap = cv2.VideoCapture()
            cap.open(filename, cv2.CAP_FFMPEG)
        return cap
    except Exception as e:
        print(f"Error opening video: {e}")
        return None

video = create_video_capture("carsVideo4.mp4")
if video is None:
    print("Failed to open video file")
    exit(1)

# Constants for output video size (reduced size for performance)
WIDTH = 480
HEIGHT = 640

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
def create_video_writer(input_video_path, fps, width, height):
    try:
        # Get the directory of the input video
        output_dir = os.path.dirname(os.path.abspath(input_video_path))
        if not output_dir:
            output_dir = '/mnt/media'  # Fallback to media directory
        
        # Create timestamp for unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_dir, f'processed_video_{timestamp}.mp4')
        
        print(f"Will save processed video to: {output_filename}")
        
        try:
            # Use H.264 codec for MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' on some systems
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
            
            if out.isOpened():
                print("Successfully created MP4 video writer")
                return out
                
            # Fallback to alternative H.264 codec if first attempt fails
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
            
            if out.isOpened():
                print("Successfully created MP4 video writer using alternative codec")
                return out
                
        except Exception as e:
            print(f"Failed to create MP4 writer: {e}")
            
            # Last resort: try x264 codec
            try:
                fourcc = cv2.VideoWriter_fourcc(*'x264')
                out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                if out.isOpened():
                    print("Successfully created MP4 video writer using x264 codec")
                    return out
            except Exception as e:
                print(f"Failed to create video writer with x264 codec: {e}")
        
        print("Failed to create video writer with any available codec")
        return None
    except Exception as e:
        print(f"Error creating video writer: {e}")
        return None

def trackMultipleObjects():
    try:
        rectangleColor = (0, 255, 255)
        frameCounter = 0
        currentCarID = 0

        vehicle_trackers = {}
        last_centroids = {}
        speeds = {}

        # Get the input video path
        input_video = "carsVideo4.mp4"
        input_video_path = os.path.abspath(input_video)

        # Create video writer with error handling
        out = create_video_writer(input_video_path, 10, WIDTH, HEIGHT)
        if out is None:
            print("Failed to create output video file")
            return

        print(f"Processing input video: {input_video_path}")
        start_time = time.time()
        last_detection_time = time.time()
    except Exception as e:
        print(f"Error initializing tracking: {e}")
        return

    while True:
        rc, image = video.read()
        if type(image) == type(None):
            break

        # Resize frame to YouTube Shorts size (vertical)
        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()
        frameCounter += 1

        # Reset trackers periodically to prevent drift
        if time.time() - start_time > 10:
            vehicle_trackers.clear()
            last_centroids.clear()
            speeds.clear()
            start_time = time.time()

        # Detect vehicles every few frames
        if frameCounter % 5 == 0 or len(vehicle_trackers) == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
            
            # Match detections to existing trackers
            detections = [(int(_x), int(_y), int(_w), int(_h)) for (_x, _y, _w, _h) in cars]
            matches, unmatched_detections = match_detections_to_trackers(detections, vehicle_trackers)
            
            # Update matched trackers
            for detection, tracker_id in matches:
                vehicle_trackers[tracker_id].update(detection)
            
            # Create new trackers for unmatched detections
            for detection in unmatched_detections:
                new_tracker = VehicleTracker(detection, currentCarID)
                vehicle_trackers[currentCarID] = new_tracker
                currentCarID += 1

        # Update all trackers and draw results
        trackers_to_delete = []
        for car_id, tracker in vehicle_trackers.items():
            x, y, w, h = tracker.get_state()
            
            # Check if vehicle is still in frame
            if x < 0 or y < 0 or x + w > WIDTH or y + h > HEIGHT:
                trackers_to_delete.append(car_id)
                continue
                
            # Draw bounding box
            cv2.rectangle(resultImage, (x, y), (x + w, y + h), rectangleColor, 2)
            
            # Calculate and display distance
            distance = estimateDistance(w)
            cv2.putText(resultImage, f"ID: {car_id}", (x, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(resultImage, f"Dist: {distance:.1f}m", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate speed if we have previous position
            current_centroid = tracker.get_centroid()
            if car_id in last_centroids:
                last_centroid = last_centroids[car_id]
                time_diff = 1/30.0  # Assuming 30 fps
                pixel_distance = calculate_distance(current_centroid, last_centroid)
                speed_ms = pixel_distance * 0.1  # Approximate pixel to meter conversion
                speed_kmh = speed_ms * 3.6
                speeds[car_id] = speed_kmh
                
                if speeds[car_id] > 0:
                    cv2.putText(resultImage, f"{speeds[car_id]:.1f} km/h",
                              (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                              0.6, (255, 0, 0), 2)
            
            last_centroids[car_id] = current_centroid

        # Remove trackers that are out of frame
        for car_id in trackers_to_delete:
            vehicle_trackers.pop(car_id, None)
            last_centroids.pop(car_id, None)
            speeds.pop(car_id, None)

        # Write output video and show progress
        out.write(resultImage)
        
        # Print progress
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = (frameCounter / total_frames) * 100
        print(f"\rProcessing: {progress:.1f}%", end="")

    print("\nProcessing complete!")
    video.release()
    out.release()

if __name__ == '__main__':
    trackMultipleObjects()
