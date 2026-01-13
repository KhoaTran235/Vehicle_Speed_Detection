# Multithreaded Vehicle Detection & Speed Estimation (YOLOv8 + TensorRT)

This repository implements a **real-time vehicle detection, tracking, and speed estimation system**
using **YOLOv8 optimized with TensorRT**, combined with a **multithreaded pipeline** for high FPS video processing.

The system is designed to:
- Run **YOLOv8** inference for vehicles detection in a separate thread
- Track vehicles with **ByteTrack**
- Estimate vehicle speed using **perspective (homography) transformation**
- Save the annotated result as an MP4 video

[![Vehicle Detection & Speed Estimation Demo](https://img.youtube.com/vi/r839jl9Ub2M/0.jpg)](https://youtu.be/r839jl9Ub2M)

**YouTube Demo**: https://youtu.be/r839jl9Ub2M

---

## Features

- YOLOv8 object detection (TensorRT `.engine`)
- Multithreading with `Queue` for high-throughput inference
- Vehicle tracking using **ByteTrack**
- Speed estimation in km/h
- Perspective transform (bird’s-eye projection)
- Real-time FPS monitoring (YOLO FPS & Display FPS)
- Save output video (`.mp4`)

---

## Installation

### 1. Create environment
```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Model Optimization 

### Step 1: Model Prunning (Recommended: Google Colab)
#### Notebook
```text
optimize_yolo.ipynb
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KhoaTran235/Vehicle_Speed_Detection/blob/main/optimize_yolo.ipynb)

What this step does:

- Prunes YOLOv8n to reduce redundant parameters

- Reduces model size and memory footprint

- Improves inference speed for real-time deployment
### Step 2: Export to TensorRT
```bash
python export.py
```
Export configuration:

- Dynamic batch

- FP16

- Integrated NMS

- Output: pruned_yolov8n.engine

--- 
## Running the system

```bash
python process_multithread.py
```

Output:

- Real-time visualization window

- Annotated video saved to: ```outputs/vehicles_result.mp4```

Press ```q``` to stop execution.

---
## Multithreading
```bash
VideoCapture
     │
     ▼
[ Frame Reader Thread ]
     │
     ▼
Frame Queue
     │
     ▼
[ YOLO Inference Thread ]
     │
     ▼
Result Queue
     │
     ▼
[ Tracking + Speed Estimation + Display Thread ]
```
Benefits:

- IO, inference, and rendering run independently

- Stable FPS even with heavy models

---
## Speed Estimation
Speed is estimated using:

1. Bottom-center anchor of bounding boxes

2. Perspective transform (homography)

3. Distance calculation in transformed space

4. Speed formula:
$$v = \frac{distance}{time} \times 3.6 \quad (km/h)$$

---
## FPS Monitoring

Displayed on video:

- YOLO FPS – object detection model inference speed

- Display FPS – visualization pipeline speed


---
## Notes

- Designed for NVIDIA GPU + TensorRT

- Perspective polygon must be calibrated per camera

- Speed estimation is relative (depends on homography accuracy)
