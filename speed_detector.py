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

    # Expand boxes slightly to be more aggressive in merging overlapping detections
    expansion = 0.1  # 10% expansion
    x1 -= w1 * expansion
    y1 -= h1 * expansion
    w1 += w1 * expansion * 2
    h1 += h1 * expansion * 2
    
    x2 -= w2 * expansion
    y2 -= h2 * expansion
    w2 += w2 * expansion * 2
    h2 += h2 * expansion * 2
    
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
    
    iou = intersection_area / union_area
    
    # Also consider the relative size difference
    size_ratio = min(box1_area, box2_area) / max(box1_area, box2_area)
    
    # Combine IoU with size ratio for better merging
    return iou * size_ratio

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
def is_overlapping(x, y, w, h, t_x, t_y, t_w, t_h, threshold=0.7):  # Increased threshold for stricter matching
    # Expand boxes slightly to be more aggressive in finding overlaps
    margin = 0.1  # 10% expansion
    x = x - w * margin
    y = y - h * margin
    w = w * (1 + 2 * margin)
    h = h * (1 + 2 * margin)
    
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
    
    # Calculate IoU (Intersection over Union)
    iou = overlap_area / union_area
    
    # Also consider relative size difference
    size_ratio = min(area, t_area) / max(area, t_area)
    
    # Consider both IoU and size ratio for better matching
    return (iou > threshold) and (size_ratio > 0.7)  # Objects should be similar in size

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

        # Get original video information
        original_fps = video.get(cv2.CAP_PROP_FPS)  # Use float FPS directly
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / original_fps if original_fps > 0 else 0
        
        # Create video writer with original FPS (float)
        out = create_video_writer(input_video_path, original_fps, WIDTH, HEIGHT)
        if out is None:
            print("Failed to create output video file")
            return
            
        print(f"\nVideo Analysis:")
        print(f"- Input dimensions: {WIDTH}x{HEIGHT}")
        print(f"- Frames per second: {original_fps:.2f}")
        print(f"- Total frames: {total_frames}")
        print(f"- Video duration: {video_duration:.2f} seconds")
        print("\nProcessing video...")

        print(f"Processing input video: {input_video_path}")
        start_time = time.time()
        last_detection_time = time.time()
    except Exception as e:
        print(f"Error initializing tracking: {e}")
        return

    processing_start_time = time.time()
    
    while True:
        # Read frame
        rc, image = video.read()
        if not rc or image is None:
            break
        
        frameCounter += 1  # Increment only once per frame read

        # Resize frame maintaining aspect ratio
        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        resultImage = image.copy()
        
        # Calculate processing progress
        if frameCounter % 30 == 0:  # Update progress every 30 frames
            elapsed_time = time.time() - processing_start_time
            current_fps = frameCounter / elapsed_time if elapsed_time > 0 else 0
            progress = (frameCounter / total_frames) * 100
            print(f"\rProcessing: {progress:.1f}% | FPS: {current_fps:.1f}", end="")
        
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

        # Clear all trackers for each new frame
        vehicle_trackers.clear()
        last_centroids.clear()
        speeds.clear()
        
        # Process new frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction to detect motion
        try:
            # Initialize avg on first frame
            if not hasattr(trackMultipleObjects, 'avg'):
                trackMultipleObjects.avg = gray.copy().astype("float")
                continue
            
            # Accumulate the weighted average between the current frame and
            # previous frames, then compute the difference between the current
            # frame and running average
            cv2.accumulateWeighted(gray, trackMultipleObjects.avg, 0.2)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(trackMultipleObjects.avg))
        except Exception as e:
            print(f"Error in motion detection: {e}")
            frameDelta = np.zeros_like(gray)
        
        # If there's no significant motion, skip this frame
        if cv2.mean(frameDelta)[0] < 2.0:  # Adjust threshold as needed
            # Instead of skipping, optionally write frame without detections here (if desired)
            continue
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Adjusted Haar Cascade parameters for long-distance detection
        cars = carCascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,    # Reduced scale factor for finer detection
            minNeighbors=5,      # Lowered to detect smaller objects
            minSize=(20, 20),    # Smaller minimum size for distant vehicles
            maxSize=(WIDTH//2, HEIGHT//2)  # Maximum size for a car
        )
        # Skip frame if no cars detected
        if len(cars) > 0:
            # Convert detections to the format expected by NMSBoxes
            boxes = [[x, y, x + w, y + h] for (x, y, w, h) in cars]
            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(
                boxes,
                [1.0] * len(boxes),  # confidence scores
                0.3,  # Score threshold
                0.3   # NMS threshold
            )
            
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                # Keep only the best detections
                cars = [cars[i] for i in indices]
            
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
                
            # Draw bounding box with thicker lines
            cv2.rectangle(resultImage, (x, y), (x + w, y + h), rectangleColor, 3)
            
            # Calculate and display distance
            distance = estimateDistance(w)
            
            # Process and draw with extended range
            if distance > 0 and distance < 200:  # Extended range to 200 meters
                # Adjust text size based on distance but keep it more readable
                text_scale = max(0.6, 1.0 - (distance/100))  # Increased minimum text size
                
                # Color coding based on distance
                if distance < 50:
                    distance_color = (0, 255, 0)  # Green for near vehicles
                elif distance < 100:
                    distance_color = (0, 255, 255)  # Yellow for medium distance
                else:
                    distance_color = (0, 0, 255)  # Red for far vehicles
                    
                # Calculate text size for background rectangle
                id_text = f"ID: {car_id}"
                dist_text = f"Dist: {distance:.1f}m"
                (id_width, id_height), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
                (dist_width, dist_height), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
                
                # Draw background rectangles for better text visibility
                padding = 4
                # Background for ID
                cv2.rectangle(resultImage, 
                              (x, y - id_height - 25), 
                              (x + id_width + padding * 2, y - 20),
                              (0, 0, 0), -1)  # Black background
                # Background for distance
                cv2.rectangle(resultImage, 
                              (x, y - dist_height - 10), 
                              (x + dist_width + padding * 2, y - 5),
                              (0, 0, 0), -1)  # Black background
                
                # Draw text with border for better visibility
                def putTextWithBorder(img, text, org, font, scale, color, thickness):
                    # Draw black border
                    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2)
                    # Draw colored text
                    cv2.putText(img, text, org, font, scale, color, thickness)
                
                # Draw ID and distance with border
                putTextWithBorder(resultImage, f"ID: {car_id}", 
                                (x + padding, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 2)
                putTextWithBorder(resultImage, f"Dist: {distance:.1f}m",
                                (x + padding, y - 5),
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
                        
                        # Calculate text size for speed
                        speed_text = f"{speed_kmh:.1f} km/h"
                        (speed_width, speed_height), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Draw background rectangle for speed
                        cv2.rectangle(resultImage, 
                                      (x, y + h + 5), 
                                      (x + speed_width + 8, y + h + 25),
                                      (0, 0, 0), -1)
                        
                        # Draw speed with border
                        putTextWithBorder(resultImage, speed_text,
                                          (x + 4, y + h + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6, (0, 255, 255), 2)
                
                last_centroids[car_id] = current_centroid

        # Highlight long-distance vehicles
        for (x, y, w, h) in cars:
            distance = estimateDistance(w)
            if distance > 50:  # Highlight vehicles farther than 50 meters
                cv2.rectangle(resultImage, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box
                cv2.putText(resultImage, f"Long Dist: {distance:.1f}m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Only write frames that have valid detections
        if len(vehicle_trackers) > 0:
            # Draw frame information
            seconds = frameCounter / original_fps
            cv2.putText(resultImage, f"Time: {int(seconds//60):02d}:{int(seconds%60):02d}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(resultImage, f"Frame: {frameCounter}/{total_frames}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write the frame
            out.write(resultImage)
        
        # Print progress
        progress = (frameCounter / total_frames) * 100
        print(f"\rProcessing: {progress:.1f}%", end="")

    print("\nProcessing complete!")
    video.release()
    out.release()

if __name__ == '__main__':
    trackMultipleObjects()
