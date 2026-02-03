# 👁️ Semantic Visual Search & Tracking Pipeline

### Real-Time Object Detection, Tracking, and Zero-Shot Re-Identification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-teal)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-black)

## 📖 Overview
This project is a modular, production-ready computer vision system designed to go beyond simple object detection. By integrating **YOLOv8** (for detection), **SORT** (for multi-object tracking), and **CLIP** (for semantic understanding), this pipeline creates a **Visual Search Engine**.

Unlike standard pipelines that only classify objects (e.g., "Person"), this system allows users to upload a reference image (e.g., a specific "Missing Person" or "Stolen Vehicle") and locate that specific target within a live video stream in real-time.

---

## 🚀 Use Cases & Business Value
This architecture addresses real-world problems in security and retail analytics:
* **Targeted Surveillance:** Locate a specific suspect or lost child in a crowd using a reference photo (Re-ID).
* **Retail Loss Prevention:** Track individuals across camera feeds even after they leave the frame.
* **Smart Asset Tracking:** Identify specific packages or vehicles in logistics hubs without retraining the model.

---

## ⚡ System Architecture

The pipeline is built on a modular "Assembly Line" pattern to ensure low latency and high scalability.

| Component | Technology | Function |
| :--- | :--- | :--- |
| **Ingestion** | Threaded OpenCV | Fault-tolerant buffering for RTSP/Webcam/Video inputs. |
| **Preprocessing** | NumPy / OpenCV | Letterbox resizing ($640\times640$), Normalization, BGR $\to$ RGB. |
| **Inference** | **YOLOv8 (ONNX)** | Optimized CPU/GPU object detection. |
| **Tracking** | **SORT** | Kalman Filter-based tracking to maintain object IDs over time. |
| **Logic Core** | **CLIP (ViT-B-32)** | Extracting feature embeddings to compare detections against user queries. |
| **Interface** | **FastAPI** | REST endpoint for dynamic query image uploading. |

---

## 📁 Project Structure

```text
computer-vision-pipeline/
├── src/
│   ├── api.py                 # FastAPI endpoint for remote image upload
│   ├── data_ingestion/        # Threaded buffer for lag-free streaming
│   ├── preprocessing/         # Image standardization & transformations
│   ├── inference/             # ONNX Runtime engine (YOLOv8)
│   ├── feature_extraction/    # CLIP Wrapper for semantic comparison
│   ├── postprocessing/        # Visualizer & SORT Tracker logic
│   └── main.py                # Pipeline orchestrator
├── models/
│   └── yolov8n.onnx           # Quantized/Optimized weights
├── query_images/              # Storage for uploaded reference targets
├── requirements.txt           # Dependencies
└── README.md
