#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2

def extract_frames_every_3m(video_dir, output_dir, speed_m_per_s):
    """
    Extract frames from videos located in category subfolders at every 3 meters travelled.

    Args:
        video_dir (str): Root directory containing category subfolders with videos.
        output_dir (str): Directory where extracted frames will be saved preserving category folders.
        speed_m_per_s (float): Speed of vehicle/camera in meters per second.

    Note:
        - Assumes constant speed during video.
        - Extracts frames at intervals corresponding to 3 meters of travel.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = os.listdir(video_dir)
    for category in categories:
        category_path = os.path.join(video_dir, category)
        if not os.path.isdir(category_path):
            continue

        output_category_path = os.path.join(output_dir, category)
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        print(f"Processing category: {category}")

        for video_filename in os.listdir(category_path):
            if not (video_filename.lower().endswith(('.mp4', '.avi', '.mov'))):
                continue

            video_path = os.path.join(category_path, video_filename)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                print(f"Unable to get FPS for video {video_path}, skipping.")
                cap.release()
                continue

            # time interval in seconds to cover 3 meters
            time_interval_sec = 3.0 / speed_m_per_s  
            # frame interval to capture a frame every 3 meters
            frame_interval = max(1, int(round(fps * time_interval_sec)))

            frame_count = 0
            saved_frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frame_filename = f"{os.path.splitext(video_filename)[0]}_frame_{saved_frame_count}.jpg"
                    frame_path = os.path.join(output_category_path, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    saved_frame_count += 1
                frame_count += 1
            cap.release()
            print(f"Extracted {saved_frame_count} frames from {video_filename}")

if __name__ == "__main__":
    # Example usage: set your video dataset path and output path here
    video_input_path = "E:\\road_project_final\\0603.mp4"
    frames_output_path = "extracted_frames_samples"
    # Provide your vehicle/camera speed in meters per second
    vehicle_speed_m_per_sec = 10.0  # example: 5 m/s (approx 18 km/h)

    print(f"Extracting frames every 3 meters at speed {vehicle_speed_m_per_sec} m/s ...")
    extract_frames_every_3m(video_input_path, frames_output_path, vehicle_speed_m_per_sec)
    print("Frame extraction complete.")


