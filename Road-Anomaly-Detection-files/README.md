# 🛣️ Road Defect Detection and Feedback System

This project combines a **Road Defect Detection and Reporting System** with a **Streamlit-based Feedback Annotation Tool** to detect road defects from videos/images, extract GPS data, generate annotated outputs, and provide a web-based interface for reviewing and annotating images.

---

## 🧭 End-to-End Process for Detection System

### 1. 📱 Record Road Video (with GPS)

- Use **GPS Map Camera app** on Android to record videos.
- Ensure GPS coordinates are overlaid on the video frame (bottom section).

### 2. 🏷️ Data Labeling using Roboflow

- Upload frames (extracted manually or using OpenCV) to [Roboflow](https://roboflow.com/).
- Label defect classes such as:
  - Cracks
  - Damage surface
  - Divider
  - Edge line
  - Lane
  - Pothole
  - Sign board
  - Zebra crossing
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
├── detect_and_extract.py           # Detects road defects and extracts GPS + address
├── generate_excel_report.py        # Merges all CSVs into a final Excel report
├── rmodelbest.pt                   # Trained YOLOv8 model
├── /road_project_output_xx/        # Folder per run with detection outputs
├── labelled_images/                # Folder for pre-labelled images
├── unlabelled_images/              # Folder for images to annotate
├── feedback_annotated_images/      # Folder where annotated images are saved
├── feedback_results.csv            # Output file containing feedback logs
├── app.py                          # Streamlit application script
└── README.md
```

---

## 📦 Requirements

Install dependencies for the detection system:

```bash
pip install opencv-python ultralytics pytesseract pandas openpyxl geopy
```

Install dependencies for the feedback annotation tool:

```bash
pip install streamlit opencv-python-headless streamlit-drawable-canvas pandas pillow
```

Additional setup:
- Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and add it to your system PATH.

---

### 4. ⚙️ Usage

### Detection System

#### Step 1: Run Detection on Video or Images

```bash
python detect_and_extract.py
```

- Annotates frames with bounding boxes
- Extracts GPS coordinates via OCR
- Reverse-geocodes GPS to human-readable addresses
- Saves:
  - Annotated video (`road_annotated_video.mp4`)
  - Defect summary CSV (`detection_with_gps.csv`)
  - Class-wise frame images (`category_wise_images/`)
  - Class counts summary (`class_counts.csv`)

#### Step 2: Generate Final Report

```bash
python generate_excel_report.py
```

- Merges all CSVs from multiple runs
- Creates a clean Excel report with merged headers

📄 Output: `final_report.xlsx`

### 5. 📝Feedback Annotation Tool

#### Step 1: Run the Streamlit App

```bash
streamlit run app.py
```

#### Step 2: Upload Images

- Upload labelled and unlabelled images (matching file names) through the sidebar.

#### Step 3: Annotate & Submit Feedback

- Draw bounding boxes on unlabelled images, assign classes, and save annotated images.
- Provide structured feedback for each defect class.
- Navigate through images using "Previous" and "Next" buttons.

#### Outputs

- **Annotated Images**: Saved as `feedback_annotated_images/annotated_<image_name>`
- **Feedback CSV**: `feedback_results.csv` includes:
  - Image name
  - Class
  - Presence
  - Detection correctness
  - Optional comment

---

## 📊 Output Example (Detection System)

| Location     | Latitude  | Longitude | Defect Category             | Cracks | Damage surface | Divider | Edge line | Lane | Pothole | Sign board | Zebra crossing | Address                                             |
|--------------|-----------|-----------|-----------------------------|--------|----------------|---------|-----------|------|---------|------------|----------------|-----------------------------------------------------|
| Walunj Road  | 19.012939 | 74.787864 | Edge line, Lane, Sign board | 0      | 0              | 0       | 1         | 3    | 0       | 1          | 0              | Walunj, Ahmednagar, Maharashtra, 414110, India      |

---

## 📤 Output Files

- **Detection System**:
  - 🎥 `road_annotated_video.mp4` – Annotated detection video
  - 📍 `detection_with_gps.csv` – Frame-wise detection data
  - 📊 `class_counts.csv` – Summary of all detections
  - 🗂️ `category_wise_images/` – Saved frames sorted by detected class
  - 📈 `final_report.xlsx` – Merged and formatted report

- **Feedback Annotation Tool**:
  - 🖼️ `feedback_annotated_images/` – Annotated images
  - 📊 `feedback_results.csv` – Feedback logs

---


---

## 📌 Notes

- Ensure image filenames are **identical** in `labelled_images/` and `unlabelled_images/` for proper comparison in the feedback tool.
- The feedback tool skips already-reviewed images to avoid duplication.
