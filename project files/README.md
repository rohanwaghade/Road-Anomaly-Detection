
# 🛣️ Road Defect Detection and Reporting System

This project detects road defects from videos/images using a YOLOv8 model, extracts GPS data, generates annotated videos, and creates a structured Excel report with location and defect summaries.

---

## 🧭 End-to-End Process

### 1. 📱 Record Road Video (with GPS)

- Use **GPS Map Camera app** on Android to record videos.
- Ensure GPS coordinates are overlaid on the video frame (bottom section).

### 2. 🏷️ Data Labeling using Roboflow

- Upload frames (extracted manually or using OpenCV) to [Roboflow](https://roboflow.com/).
- Label defect classes such as:
  - cracks
  - pothole
  - divider
  - edge line
  - lane
  - sign board
  - zebra crossing
- Export the dataset in YOLOv8 format.

### 3. 🧠 Model Training

- Train the model using **Ultralytics YOLOv8**:
```bash
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```
- Save best weights as `rmodelbest.pt`

---

## 📁 Project Structure

```bash
.
├── detect_and_extract.py       # Detects road defects and extracts GPS + address
├── generate_excel_report.py    # Merges all CSVs into a final Excel report
├── rmodelbest.pt               # Trained YOLOv8 model
├── /road_project_output_xx/    # Folder per run with outputs
└── README.md
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements:
- `opencv-python`
- `ultralytics`
- `pytesseract`
- `pandas`
- `openpyxl`
- `geopy`

Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and add it to your system PATH.

---

## ⚙️ Usage

### Step 1: Run Detection on Video or Images

```bash
python detect_and_extract.py
```

- Annotates frames with bounding boxes
- Extracts GPS coordinates via OCR
- Reverse-geocodes GPS to human-readable addresses
- Saves:
  - Annotated video
  - Defect summary CSV
  - Class-wise frame images

### Step 2: Generate Final Report

```bash
python generate_excel_report.py
```

- Merges all CSVs from multiple runs
- Creates a clean Excel report with merged headers

📄 Output: `final_report.xlsx`

---

## 📊 Output Example

| Location | Latitude  | Longitude | Defect Category             | cracks | damage surface | divider | edge line | lane | pothole | sign board | zebra crossing | Address                                             |
|----------|-----------|-----------|------------------------------|--------|----------------|---------|-----------|------|---------|-------------|----------------|------------------------------------------------------|
| Walunj   | 19.012939 | 74.787864 | edge line, lane, sign board  |   0    |       0        |    0    |     1     |  3   |    0    |      1      |       0        | Walunj, Ahmednagar, Maharashtra, 414110, India      |

---

## 📤 Output Files

- 🎥 `road_annotated_video.mp4` – Annotated detection video
- 📍 `detection_with_gps.csv` – Frame-wise detection data
- 📊 `class_counts.csv` – Summary of all detections
- 🗂️ `category_wise_images/` – Saved frames sorted by detected class
- 📈 `final_report.xlsx` – Merged and formatted report

---

## 💡 Ideas for Improvement

- Automatically detect GPS overlay zone
- Integrate feedback loop for retraining
- Add a web-based dashboard with filtering

---

## 👤 Author

**Rohan Waghade**  
Intern – Data Science

---

## 📄 License

This project is licensed under the **MIT License**.
