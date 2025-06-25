
# 🛣️ Road Defect Feedback Annotation Tool

This Streamlit web application is designed for reviewing and annotating **road defect images**. It allows users to:
- View labelled and unlabelled images side by side
- Provide structured feedback per defect class
- Annotate unlabelled images with bounding boxes and class labels

## 📂 Folder Structure

```
.
├── labelled_images/            # Folder for pre-labelled images
├── unlabelled_images/          # Folder for images to annotate
├── feedback_annotated_images/  # Folder where annotated images will be saved
├── feedback_results.csv        # Output file containing feedback logs
├── app.py                      # Streamlit application script
└── README.md
```

## ⚙️ Features

- **Dual Image Display**: Compares labelled vs unlabelled images.
- **Bounding Box Annotation**: Draw boxes on unlabelled images and assign predefined defect classes.
- **Multi-Class Feedback**: Submit structured feedback for 8 road defect classes:
  - Cracks
  - Damage surface
  - Divider
  - Edge line
  - Lane
  - Pothole
  - Sign board
  - Zebra crossing
- **Session Management**: Navigate through image list and maintain annotation state.
- **CSV Export**: Download all feedback as a CSV file.

## 🚀 Getting Started

1. **Install dependencies**

```bash
pip install streamlit opencv-python-headless streamlit-drawable-canvas pandas pillow
```

2. **Run the app**

```bash
streamlit run app.py
```

3. **Upload images**
   - Upload labelled and unlabelled images (matching file names) through the sidebar.

4. **Annotate & Submit Feedback**
   - Draw boxes → assign classes → save annotated images.
   - Fill feedback form and submit for each class.
   - Navigate through images using "Previous" and "Next" buttons.

## 📥 Output

- **Annotated Image**: Saved as `feedback_annotated_images/annotated_<image_name>`
- **Feedback CSV**: `feedback_results.csv` includes:
  - Image name
  - Class
  - Presence
  - Detection correctness
  - Optional comment

## 📌 Notes

- Ensure image filenames are **identical** in both folders for proper comparison.
- You can re-run the app anytime; it skips already-reviewed images.
