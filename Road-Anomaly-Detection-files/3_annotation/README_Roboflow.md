# 📘 Roboflow Labeling Guide

## ✅ Step-by-Step Instructions

### 1. Sign Up / Log In
- Go to [https://roboflow.com](https://roboflow.com)
- Create an account or log in.

### 2. Create a Project
- Click **"Create New Project"**.
- Name your project (e.g., "Road Defects").
- Choose project type: **Object Detection**.
- Annotation format: **VOC XML** (it will be converted later).

### 3. Upload Images
- Upload the extracted frames (`.jpg` format recommended).
- Organize them properly before uploading.

### 4. Annotate Images
- Use the Roboflow annotation tool to draw bounding boxes.
- Assign one of the predefined classes:
  - `crack`
  - `pothole`
  - `divider`
  - `edge line`
  - `lane`
  - `sign board`
  - `zebra crossing`

### 5. Generate Dataset
- After labeling, click **"Generate"**.
- Choose export format: **YOLOv8**.
- Download the ZIP file.

### 6. Use with YOLOv8
The downloaded dataset is ready to be used directly with YOLOv8 for training.

## 📎 Tips
- Label accurately to improve model performance.
- Review labels to ensure class consistency.
