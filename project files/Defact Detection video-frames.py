#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# === Import necessary libraries ===
import cv2  # For video and image processing
from ultralytics import YOLO  # YOLOv8 object detection model
import pytesseract  # OCR library to extract text (GPS) from frames
import random
import numpy as np
import os
import csv
import re  # Regular expressions for parsing GPS text
from collections import defaultdict  # For counting detections
from glob import glob  # To find image files if needed

# === Configuration ===
model_path = "E:\\road_project_final\\rmodelbest.pt"  # Path to trained YOLO model
input_path = "E:\\all videos\\darewadi.mp4"  # Input video file/image folder
output_folder = "E:\\road_project_all_final\\road_project_output_darewadi"  # Folder for saving outputs
os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

# Output file paths
output_video_path = os.path.join(output_folder, "road_annotated_video.mp4")
output_csv_path = os.path.join(output_folder, "detection_with_gps.csv")
output_txt_path = os.path.join(output_folder, "class_counts.csv")

# Output video settings
output_width, output_height = 1366, 768
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
fps = 4  # Frames per second for output video

# === Load YOLO Model ===
model = YOLO(model_path)  # Load YOLOv8 model
names = model.names  # Dictionary of class ID to class name
class_list = list(names.values())  # List of class names

# === Define preset colors for bounding boxes (one per class) ===
preset_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
    (255, 0, 255), (128, 0, 128), (0, 128, 255),
    (128, 128, 0), (0, 128, 128), (0, 0, 128)
]
colors = {i: preset_colors[i % len(preset_colors)] for i in names}  # Assign a color per class ID

# === Initialize counters and data structures ===
frame_count = 0  # To count number of frames processed
class_counts = defaultdict(int)  # To keep total count per class
csv_data = [['Frame', 'Latitude', 'Longitude', 'Defect Category'] + class_list]  # Header row for CSV
stopped_by_user = False  # Track whether user pressed ESC to stop

# === Resize and pad frame to fit output size ===
def resize_and_pad(image, target_size):
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

# === Extract GPS (Latitude, Longitude) from frame using OCR ===
def extract_gps_text(image):
    h = image.shape[0]
    gps_region = image[int(h * 0.75):, :]  # Crop bottom 25% where GPS text appears
    gray = cv2.cvtColor(gps_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduce noise
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarize
    config = "--psm 6"  # PSM 6 assumes a uniform block of text
    raw_text = pytesseract.image_to_string(binary, config=config)  # OCR

    # Clean and parse OCR text line-by-line
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    latitude, longitude = "NaN", "NaN"

    for line in lines:
        if 'lat' in line.lower() and 'long' in line.lower():
            gps_match = re.search(r'lat[:\s\-]*([\-+]?\d+\.\d+)[^\d]+long[:\s\-]*([\-+]?\d+\.\d+)', line, re.IGNORECASE)
            if gps_match:
                latitude, longitude = gps_match.groups()
                break

    return latitude, longitude

# === Setup Input: video or image folder ===
if os.path.isfile(input_path):  # If it's a video file
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    input_source = "video"
else:  # Else, process image sequence
    image_paths = sorted(glob(os.path.join(input_path, "*.jpg")) + glob(os.path.join(input_path, "*.png")))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    input_source = "images"

# === Main function to process a single frame ===
def process_frame(frame):
    global frame_count, stopped_by_user
    frame_count += 1

    if frame_count % 4 != 0:
        return  # Skip 3 out of 4 frames to reduce processing

    latitude, longitude = extract_gps_text(frame)  # Extract GPS from frame
    results = model.track(frame, persist=True, conf=0.1)  # Run YOLO object detection + tracking
    frame_class_counts = defaultdict(int)  # Count classes in current frame
    detected_labels = set()  # Track detected class labels

    # Draw detections if found
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs

        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = box
            label = names[class_id]
            color = colors.get(class_id, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw bounding box
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 4, y1), color, -1)  # Label background
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame_class_counts[label] += 1
            class_counts[label] += 1
            detected_labels.add(label)

    # Save detection info to CSV and save frame image per class
    if detected_labels:
        row = [frame_count, latitude, longitude, ", ".join(sorted(detected_labels))]
        row.extend(frame_class_counts.get(cls, 0) for cls in class_list)
        csv_data.append(row)

        # Save annotated frame into category-wise folder
        category_folder = os.path.join(output_folder, "category_wise_images")
        os.makedirs(category_folder, exist_ok=True)
        for label in detected_labels:
            class_dir = os.path.join(category_folder, label)
            os.makedirs(class_dir, exist_ok=True)
            frame_filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(class_dir, frame_filename), frame)

    # Resize and show frame
    output_frame = resize_and_pad(frame, (output_width, output_height))
    out.write(output_frame)
    cv2.imshow("FRAME", output_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
        stopped_by_user = True
        return "exit"

# === Process Video or Images ===
if input_source == "video":
    while True:
        ret, frame = cap.read()
        if not ret or process_frame(frame) == "exit":
            break
    cap.release()
else:
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if process_frame(frame) == "exit":
            break

out.release()
cv2.destroyAllWindows()

# === Save CSV: Detections per frame ===
if len(csv_data) == 1:  # No detections
    csv_data.append(["No detections found"] + [""] * (len(csv_data[0]) - 1))

with open(output_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

# === Save class count summary ===
with open(output_txt_path, 'w') as f:
    f.write("Class Name - Detection Count\n" + "=" * 30 + "\n")
    if class_counts:
        for label, count in class_counts.items():
            f.write(f"{label} - {count}\n")
    else:
        f.write("No detections recorded.\n")

# === Final Status Messages ===
if stopped_by_user:
    print("⏹️ Stopped by user (ESC key)")
print(f"✅ Video saved: {output_video_path}")
print(f"📍 CSV saved: {output_csv_path}")
print(f"📊 CSV saved: {output_txt_path}")
print(f"🗂️ Frames saved into category-wise folders at: {output_folder}\\category_wise_images")

