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
        self.last_update_time = time.time()
        self.confidence = 1.0
        self.update(detection)
    
    def update(self, detection):
        self.bbox = detection
        self.time_since_update = 0
        self.hits += 1
        self.last_update_time = time.time()
        
        # Store position history for movement analysis
        x, y, w, h = detection
        centroid = (x + w/2, y + h/2)
        self.history.append((centroid[0], centroid[1], w, h))
        
        if len(self.history) > 30:  # Keep only last 30 frames
            self.history.pop(0)
            
        # Update confidence based on detection consistency
        if len(self.history) >= 2:
            last_w = self.history[-2][2]
            current_w = w
            size_change = abs(current_w - last_w) / last_w
            self.confidence = max(0.1, self.confidence * (1 - size_change))
        
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

video = create_video_capture("carsVideo5.mp4")
if video is None:
    print("Failed to open video file")
    exit(1)

# Get input video dimensions and calculate output size maintaining aspect ratio
input_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
target_width = 640  # Set your desired width

# Calculate height to maintain aspect ratio
aspect_ratio = input_height / input_width
target_height = int(target_width * aspect_ratio)

# Set final dimensions
WIDTH = target_width
HEIGHT = target_height

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

# Distance Estimation with improved accuracy for long distances
def estimateDistance(width_in_pixels):
    if width_in_pixels == 0:
        return 0
    
    # Calculate basic distance
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / width_in_pixels
    
    # Apply non-linear correction for long distances
    if distance > 50:
        # Apply correction factor for better accuracy at long distances
        correction_factor = 1.0 + (distance - 50) * 0.01
        distance = distance * correction_factor
    
    return distance

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

        # Get original video FPS and frame count
        original_fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / original_fps if original_fps > 0 else 0
        
        if original_fps == 0:
            original_fps = 30  # fallback if FPS cannot be determined
        
        # Create video writer with error handling
        out = create_video_writer(input_video_path, original_fps, WIDTH, HEIGHT)
        if out is None:
            print("Failed to create output video file")
            return
            
        print(f"\nVideo Analysis:")
        print(f"- Input dimensions: {WIDTH}x{HEIGHT}")
        print(f"- Frames per second: {original_fps}")
        print(f"- Total frames: {total_frames}")
        print(f"- Video duration: {video_duration:.2f} seconds")
        print(f"- Frames per second (calculated): {original_fps:.2f}")
        print("\nProcessing video...")

        print(f"Processing input video: {input_video_path}")
        start_time = time.time()
        last_detection_time = time.time()
    except Exception as e:
        print(f"Error initializing tracking: {e}")
        return

    # Initialize FPS calculation variables
    frame_times = []
    fps_update_interval = 30  # Update FPS every 30 frames
    processing_start_time = time.time()
    
    while True:
        frame_start_time = time.time()
        
        rc, image = video.read()
        if type(image) == type(None):
            break

        # Resize frame maintaining aspect ratio
        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        resultImage = image.copy()
        
        # Calculate FPS
        frame_times.append(time.time() - frame_start_time)
        if len(frame_times) > fps_update_interval:
            frame_times.pop(0)
        
        current_fps = len(frame_times) / sum(frame_times)
        
        # Update FPS display every 30 frames
        if frameCounter % fps_update_interval == 0:
            elapsed_time = time.time() - processing_start_time
            avg_fps = frameCounter / elapsed_time if elapsed_time > 0 else 0
            print(f"\rProcessing FPS: {current_fps:.1f} | Average FPS: {avg_fps:.1f} | ", end="")
        
        # Add black borders if needed to maintain aspect ratio
        if resultImage.shape[0] != HEIGHT or resultImage.shape[1] != WIDTH:
            # Create black canvas
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            # Calculate position to center the image
            y_offset = (HEIGHT - resultImage.shape[0]) // 2
            x_offset = (WIDTH - resultImage.shape[1]) // 2
            # Place the image in the center
            canvas[y_offset:y_offset + resultImage.shape[0], 
                  x_offset:x_offset + resultImage.shape[1]] = resultImage
            resultImage = canvas
        frameCounter += 1

        # Clear all trackers for each new frame
        vehicle_trackers.clear()
        last_centroids.clear()
        speeds.clear()
        
        # Process new frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction to detect motion
        if frameCounter == 1:
            avg = gray.copy().astype("float")
            continue
        
        # Accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, 0.2)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        
        # If there's no significant motion, skip this frame
        if cv2.mean(frameDelta)[0] < 2.0:  # Adjust threshold as needed
            continue
            
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Detect vehicles with parameters optimized for long-distance detection
        cars = carCascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller scale factor to detect distant vehicles
            minNeighbors=5,    # Reduced to detect smaller objects
            minSize=(20, 20),  # Smaller minimum size for distant vehicles
            maxSize=(WIDTH, HEIGHT)  # Allow larger maximum size
        )
        
        # Skip frame if no cars detected
        if len(cars) == 0:
            continue
            
        # Create new trackers for each detection
        currentCarID = 0  # Reset ID counter for each frame
        for (x, y, w, h) in cars:
            # Adjusted size filtering for long-distance vehicles
            if w < 20 or h < 20 or w > WIDTH or h > HEIGHT:  # More permissive size constraints
                continue
                
            # Create a new tracker for this detection
            detection = (int(x), int(y), int(w), int(h))
            new_tracker = VehicleTracker(detection, currentCarID)
            vehicle_trackers[currentCarID] = new_tracker
            currentCarID += 1

        # Draw detection results
        for car_id, tracker in vehicle_trackers.items():
            x, y, w, h = tracker.get_state()
            
            # Draw bounding box
            cv2.rectangle(resultImage, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Calculate and display distance
            distance = estimateDistance(w)
            if 0 < distance < 50:  # Only show reasonable distances
                # Draw ID and distance
                cv2.putText(resultImage, f"ID: {car_id}", (x, y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(resultImage, f"Dist: {distance:.1f}m", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            # Draw bounding box
            cv2.rectangle(resultImage, (x, y), (x + w, y + h), rectangleColor, 2)
            
            # Calculate and display distance
            distance = estimateDistance(w)
            
            # Process and draw with extended range
            if distance > 0 and distance < 200:  # Extended range to 200 meters
                # Adjust text size based on distance
                text_scale = max(0.4, 0.8 - (distance/100))  # Text gets smaller with distance
                
                # Color coding based on distance
                if distance < 50:
                    distance_color = (0, 255, 0)  # Green for near vehicles
                elif distance < 100:
                    distance_color = (0, 255, 255)  # Yellow for medium distance
                else:
                    distance_color = (0, 0, 255)  # Red for far vehicles
                
                cv2.putText(resultImage, f"ID: {car_id}", (x, y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), 2)
                cv2.putText(resultImage, f"Dist: {distance:.1f}m", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale, distance_color, 2)
                
                # Calculate speed if we have previous position
                current_centroid = tracker.get_centroid()
                if car_id in last_centroids:
                    last_centroid = last_centroids[car_id]
                    time_diff = 1/30.0  # Assuming 30 fps
                    pixel_distance = calculate_distance(current_centroid, last_centroid)
                    speed_ms = pixel_distance * 0.1  # Approximate pixel to meter conversion
                    speed_kmh = speed_ms * 3.6
                    
                    # Only store reasonable speeds
                    if 0 < speed_kmh < 150:  # reasonable speed range
                        speeds[car_id] = speed_kmh
                        
                        cv2.putText(resultImage, f"{speed_kmh:.1f} km/h",
                                  (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (255, 0, 0), 2)
                
                last_centroids[car_id] = current_centroid

        # Only write frames that have valid detections
        if len(vehicle_trackers) > 0:
            # Draw frame information
            cv2.putText(resultImage, f"Frame: {frameCounter}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(resultImage, f"FPS: {current_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write the frame only if we have valid detections
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
