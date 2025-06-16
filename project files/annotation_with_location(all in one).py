#!/usr/bin/env python
# coding: utf-8

# In[14]:


# === Import necessary libraries ===
import cv2
import pytesseract
import numpy as np
import os
import csv
import re
import requests
import time
from collections import defaultdict
from glob import glob
from ultralytics import YOLO

# === Configuration ===
model_path = "E:\\road_project_final\\rmodelbest.pt"
input_path = "E:\\raw videos\\ambhora.mp4"
output_folder = "E:\\project outputs\\ambhora_road"
os.makedirs(output_folder, exist_ok=True)

output_video_path = os.path.join(output_folder, "road_annotated_video.mp4")
output_csv_path = os.path.join(output_folder, "detection_with_gps.csv")
output_txt_path = os.path.join(output_folder, "class_counts.csv")
labelled_images_folder = os.path.join(output_folder, "labelled_images_every_3m")
os.makedirs(labelled_images_folder, exist_ok=True)

output_width, output_height = 1366, 768
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 4

model = YOLO(model_path)
names = model.names
class_list = list(names.values())

preset_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
    (255, 0, 255), (128, 0, 128), (0, 128, 255),
    (128, 128, 0), (0, 128, 128), (0, 0, 128)
]
colors = {i: preset_colors[i % len(preset_colors)] for i in names}

frame_count = 0
class_counts = defaultdict(int)
csv_data = [['Frame', 'Latitude', 'Longitude', 'Location', 'Defect Category'] + class_list]
stopped_by_user = False
location_cache = {}

# === Location Extraction ===
def get_location(lat, lon, retries=3):
    key = f"{lat},{lon}"
    if key in location_cache:
        return location_cache[key]
    headers = {'User-Agent': 'Mozilla/5.0'}
    for _ in range(retries):
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                location = data.get('display_name', 'Unknown')
                location_cache[key] = location
                return location
        except:
            time.sleep(1)
    return "Unknown"

# === Resize frame ===
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

# === OCR for GPS ===
def extract_gps_text(image):
    h = image.shape[0]
    gps_region = image[int(h * 0.75):, :]
    gray = cv2.cvtColor(gps_region, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_text = pytesseract.image_to_string(binary, config="--psm 6")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    latitude, longitude = "NaN", "NaN"
    for line in lines:
        if 'lat' in line.lower() and 'long' in line.lower():
            gps_match = re.search(r'lat[:\s\-]([\-+]?\d+\.\d+)[^\d]+long[:\s\-]([\-+]?\d+\.\d+)', line, re.IGNORECASE)
            if gps_match:
                latitude, longitude = gps_match.groups()
                break
    return latitude, longitude

if os.path.isfile(input_path):
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    input_source = "video"
else:
    image_paths = sorted(glob(os.path.join(input_path, ".jpg")) + glob(os.path.join(input_path, ".png")))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    input_source = "images"

# === Process each frame ===
def process_frame(frame):
    global frame_count, stopped_by_user
    frame_count += 1
    if frame_count % 4 != 0:
        return
    latitude, longitude = extract_gps_text(frame)
    location = get_location(latitude, longitude) if latitude != "NaN" else "Unknown"
    results = model.track(frame, persist=True, conf=0.1)
    frame_class_counts = defaultdict(int)
    detected_labels = set()

    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = box
            label = names[class_id]
            color = colors.get(class_id, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            frame_class_counts[label] += 1
            class_counts[label] += 1
            detected_labels.add(label)

    if detected_labels:
        row = [frame_count, latitude, longitude, location, ", ".join(sorted(detected_labels))]
        row.extend(frame_class_counts.get(cls, 0) for cls in class_list)
        csv_data.append(row)

        category_folder = os.path.join(output_folder, "category_wise_images")
        os.makedirs(category_folder, exist_ok=True)
        for label in detected_labels:
            class_dir = os.path.join(category_folder, label)
            os.makedirs(class_dir, exist_ok=True)
            frame_filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(class_dir, frame_filename), frame)

        if frame_count % 12 == 0:
            cv2.imwrite(os.path.join(labelled_images_folder, f"frame_{frame_count}.jpg"), frame)

    output_frame = resize_and_pad(frame, (output_width, output_height))
    out.write(output_frame)
    cv2.imshow("FRAME", output_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        stopped_by_user = True
        return "exit"

# === Run Processing ===
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

if len(csv_data) == 1:
    csv_data.append(["No detections found"] + [""] * (len(csv_data[0]) - 1))

with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

with open(output_txt_path, 'w') as f:
    f.write("Class Name - Detection Count\n" + "=" * 30 + "\n")
    for label, count in class_counts.items():
        f.write(f"{label} - {count}\n")

if stopped_by_user:
    print("⏹ Stopped by user (ESC key)")
print(f"✅ Video saved: {output_video_path}")
print(f"📍 CSV saved: {output_csv_path}")
print(f"📊 Class count saved: {output_txt_path}")
print(f"🗂 Category-wise images: {os.path.join(output_folder, 'category_wise_images')}")
print(f"📸 Every-3m labelled images: {labelled_images_folder}")


# In[ ]:




