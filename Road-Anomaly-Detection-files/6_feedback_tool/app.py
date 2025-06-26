import streamlit as st
import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# === Folder Configuration ===
labelled_folder = "labelled_images"
unlabelled_folder = "unlabelled_images"
annotated_folder = "feedback_annotated_images"
output_file = "feedback_results.csv"

os.makedirs(labelled_folder, exist_ok=True)
os.makedirs(unlabelled_folder, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)

# === Default Class List ===
default_classes = [
    "cracks", "damage surface", "divider", "edge line",
    "lane", "pothole", "sign board", "zebra crossing"
]

# === Session State Initialization ===
if "img_index" not in st.session_state:
    st.session_state.img_index = 0
if "show_annotator" not in st.session_state:
    st.session_state.show_annotator = False
if "class_list" not in st.session_state:
    st.session_state.class_list = default_classes.copy()

# === Page Config ===
st.set_page_config(page_title="Road Defect Feedback", layout="wide")
st.title("🚧 Multi-Class Road Defect Feedback")

# === Sidebar: Upload Images ===
st.sidebar.header("📄 Upload Images")
uploaded_labelled = st.sidebar.file_uploader("Upload Labelled Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_unlabelled = st.sidebar.file_uploader("Upload Unlabelled Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# === Save Uploaded Files ===
if uploaded_labelled:
    for file in uploaded_labelled:
        with open(os.path.join(labelled_folder, file.name), "wb") as f:
            f.write(file.read())
    st.sidebar.success(f"✅ Uploaded {len(uploaded_labelled)} labelled images.")

if uploaded_unlabelled:
    for file in uploaded_unlabelled:
        with open(os.path.join(unlabelled_folder, file.name), "wb") as f:
            f.write(file.read())
    st.sidebar.success(f"✅ Uploaded {len(uploaded_unlabelled)} unlabelled images.")

# === Sidebar: Add New Class ===
st.sidebar.markdown("### ➕ Add New Defect Category")
new_class = st.sidebar.text_input("Enter new category")

if st.sidebar.button("Add Category"):
    cleaned_class = new_class.strip().lower()
    if cleaned_class and cleaned_class not in [cls.lower() for cls in st.session_state.class_list]:
        st.session_state.class_list.append(cleaned_class)
        st.sidebar.success(f"✅ Added class: {cleaned_class}")
        st.experimental_rerun()
    elif cleaned_class in [cls.lower() for cls in st.session_state.class_list]:
        st.sidebar.warning("⚠️ This category already exists.")
    else:
        st.sidebar.warning("⚠️ Please enter a valid category name.")

# === Load Image Lists ===
labelled_images = set(f for f in os.listdir(labelled_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
unlabelled_images = set(f for f in os.listdir(unlabelled_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
common_images = sorted(labelled_images & unlabelled_images)

if not common_images:
    st.warning("⚠️ No matching images found in both folders.")
    st.stop()

# === Filter Already Reviewed ===
reviewed_images = set()
if os.path.exists(output_file):
    try:
        reviewed_images = set(pd.read_csv(output_file)["image"].unique())
    except:
        pass

unreviewed_images = [img for img in common_images if img not in reviewed_images]
if not unreviewed_images:
    st.success("🎉 All images have been reviewed!")
    st.stop()

if st.session_state.img_index >= len(unreviewed_images):
    st.session_state.img_index = 0

selected_image = unreviewed_images[st.session_state.img_index]
labelled_path = os.path.join(labelled_folder, selected_image)
unlabelled_path = os.path.join(unlabelled_folder, selected_image)

# === Show Images ===
col1, col2 = st.columns(2)
with col1:
    st.image(labelled_path, caption="🟩 Labelled Image")
with col2:
    st.image(unlabelled_path, caption="🗭 Unlabelled Image")
    if st.button("✏️ Annotate this Image"):
        st.session_state.show_annotator = True

# === Annotator ===
if st.session_state.show_annotator:
    st.subheader("✍️ Annotate Image with Bounding Boxes")
    image = Image.open(unlabelled_path)
    image_np = np.array(image.convert("RGB")).copy()

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="blue",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="rect",
        key="canvas",
        display_toolbar=True
    )

    if canvas_result.json_data and canvas_result.json_data["objects"]:
        st.markdown("### 🔖 Assign Classes to Each Box")
        box_data = []
        for i, obj in enumerate(canvas_result.json_data["objects"]):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**Box {i+1}**")
            with col2:
                class_label = st.selectbox(
                    f"Select class for Box {i+1}",
                    st.session_state.class_list,
                    key=f"class_select_{i}"
                )
            box_data.append({
                "left": int(obj["left"]),
                "top": int(obj["top"]),
                "width": int(obj["width"]),
                "height": int(obj["height"]),
                "class": class_label
            })

        if st.button("💾 Save Annotated Image"):
            for box in box_data:
                left, top, width, height = box["left"], box["top"], box["width"], box["height"]
                cls = box["class"]
                color = (0, 255, 0)
                cv2.rectangle(image_np, (left, top), (left + width, top + height), color, 2)
                cv2.putText(image_np, cls, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            save_path = os.path.join(annotated_folder, f"annotated_{selected_image}")
            Image.fromarray(image_np).save(save_path)
            st.success(f"✅ Annotated image saved to: {save_path}")
    else:
        st.warning("🖍️ Draw bounding boxes on the image to begin annotation.")

# === Feedback Form ===
st.markdown(f"### 🖼️ Image {st.session_state.img_index + 1} of {len(unreviewed_images)}")
st.subheader("📝 Provide Feedback for Each Class")
feedback_data = []
for cls in st.session_state.class_list:
    st.markdown(f"**{cls}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        present = st.radio(f"Is '{cls}' present?", ["No", "Yes"], key=f"{cls}_present_{selected_image}")
    with col2:
        correct = st.radio(f"Correct detection?", ["N/A", "Yes", "No"], key=f"{cls}_correct_{selected_image}")
    with col3:
        comment = st.text_input("Comment", key=f"{cls}_comment_{selected_image}")
    feedback_data.append({
        "image": selected_image,
        "class": cls,
        "present": present,
        "correct": correct,
        "comment": comment.strip()
    })

# === Submit Feedback ===
if st.button("📌 Submit Feedback"):
    valid_feedback = [row for row in feedback_data if row["present"] == "Yes" or row["correct"] != "N/A" or row["comment"]]
    if not valid_feedback:
        st.warning("⚠️ No valid feedback to save.")
    else:
        df_new = pd.DataFrame(valid_feedback)
        if os.path.exists(output_file):
            df_existing = pd.read_csv(output_file)
            df_new = pd.concat([df_existing, df_new], ignore_index=True)
        df_new.to_csv(output_file, index=False)
        st.success("✅ Feedback saved.")
        st.session_state.img_index += 1
        st.session_state.show_annotator = False
        st.rerun()

# === Navigation Buttons ===
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("⬅️ Previous"):
        st.session_state.img_index = max(0, st.session_state.img_index - 1)
        st.session_state.show_annotator = False
        st.rerun()
with col_next:
    if st.button("➡️ Next"):
        st.session_state.img_index = min(len(unreviewed_images) - 1, st.session_state.img_index + 1)
        st.session_state.show_annotator = False
        st.rerun()

# === Download Feedback CSV ===
if os.path.exists(output_file):
    with open(output_file, "rb") as f:
        st.download_button(
            label="📅 Download Feedback CSV",
            data=f,
            file_name="feedback_results.csv",
            mime="text/csv"
        )
