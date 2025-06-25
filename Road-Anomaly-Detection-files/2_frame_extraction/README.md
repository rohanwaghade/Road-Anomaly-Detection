# Frame Extraction Every 3 Meters

This script extracts frames from videos every **3 meters** of travel, assuming a constant vehicle/camera speed.

## 📄 Description

The script processes videos stored in subfolders (categories) of a specified directory. For each video, it calculates the frame interval required to extract a frame approximately every 3 meters based on the given speed, then saves the extracted frames into corresponding output folders.

## 📁 Folder Structure

```
video_dir/
├── category1/
│   ├── video1.mp4
│   └── video2.mp4
├── category2/
│   └── video3.mp4
...
```

After processing:

```
output_dir/
├── category1/
│   ├── video1_frame_0.jpg
│   ├── video1_frame_1.jpg
│   └── ...
├── category2/
│   └── video3_frame_0.jpg
...
```

## ⚙️ How It Works

- Calculates frame interval using:

  ```
  frame_interval = fps * (3 / speed_m_per_s)
  ```

- Extracts and saves a frame at every `frame_interval`.

## 🚀 Usage

Edit the main block in the script:

```python
video_input_path = "path_to_videos"
frames_output_path = "path_to_save_frames"
vehicle_speed_m_per_sec = 10.0  # example speed in meters/second
```

Then run the script:

```bash
python frameextraction.py
```

## 📦 Requirements

- Python 3.x
- OpenCV

Install dependencies:

```bash
pip install opencv-python
```

## 📝 Notes

- Only `.mp4`, `.avi`, and `.mov` formats are supported.
- Assumes constant speed and uniform video FPS.
