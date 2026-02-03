# 👁️ Semantic Visual Search & Tracking Pipeline
### Real-Time Zero-Shot Object Re-Identification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-green) ![OpenAI CLIP](https://img.shields.io/badge/OpenAI-CLIP-black) ![FastAPI](https://img.shields.io/badge/FastAPI-REST-teal)

## 📖 Overview
This project goes beyond standard object detection by implementing a **Visual Search Engine** for live video streams. It integrates **YOLOv8** for detection, **SORT** for temporal tracking, and **OpenAI's CLIP** for semantic feature extraction.

The system allows users to upload a **reference image** (e.g., a photo of a specific backpack, person, or vehicle) via a REST API. The pipeline then scans the live feed, generates high-dimensional vector embeddings for every detected object, and highlights the target based on semantic similarity—all in real-time.

---

## 🎥 Demo
![Visual Search Demo](media/demo.gif)
> *The system detects multiple objects, tracks them across frames, and specifically identifies the target (Reference Image) using Vector Similarity.*

---

## ⚡ System Architecture

The pipeline uses a modular "Assembly Line" architecture designed to minimize latency (lag) while running heavy neural networks.

| Stage | Component | Technology | Responsibility |
| :--- | :--- | :--- | :--- |
| **1. Ingestion** | `BufferLoader` | **Threaded OpenCV** | Decouples frame reading from processing to prevent I/O blocking. |
| **2. Inference** | `ObjectDetector` | **YOLOv8 (ONNX)** | Detects objects and extracts bounding boxes at high FPS. |
| **3. Tracking** | `Tracker` | **SORT (Kalman Filter)** | Assigns unique IDs to objects to maintain identity across frames. |
| **4. Analysis** | `FeatureExtractor` | **CLIP (ViT-B-32)** | "Crops" detections and converts them into 512-D vector embeddings. |
| **5. Matching** | `LogicCore` | **Cosine Similarity** | Compares the *Target Vector* vs. *Live Object Vectors* to find a match. |
| **6. Control** | `API` | **FastAPI** | Allows dynamic updating of the target image without stopping the stream. |



[Image of System Architecture Diagram]


---

## 🛠️ The "Magic": How it Works
The core innovation here is **Zero-Shot Re-Identification**. We do not train the model on specific objects. Instead, we use vector math:

1.  **Vectorizing the Target:** When a user uploads a reference photo, CLIP converts it into a mathematical list of numbers (a 512-dimensional vector).
2.  **Vectorizing the World:** Every object YOLO sees is cropped and passed through CLIP to get its own vector.
3.  **The Match:** We calculate the **Cosine Similarity** between the *Target Vector* and the *Live Vectors*.
    * If Similarity > 0.85 $\rightarrow$ **TARGET FOUND**.
    * If Similarity < 0.85 $\rightarrow$ Ignore.

---

## 🚀 Use Cases & Business Value
* **Security & Surveillance:** "Find this missing person in the crowd." (Re-ID without retraining).
* **Retail Analytics:** "Track customer #45 across different camera aisles."
* **Logistics:** "Locate the package labeled 'Fragile' on the conveyor belt."

---

## 📁 Project Structure
The codebase is organized as a production-grade Python package.

```text
visual-search-pipeline/
├── src/
│   ├── api.py               # FastAPI endpoint for dynamic target upload
│   ├── data_ingestion.py    # Threaded buffer for RTSP/Webcam
│   ├── inference.py         # ONNX Runtime engine (YOLOv8)
│   ├── feature_extraction.py# CLIP Wrapper for semantic embeddings
│   ├── tracker.py           # SORT logic for ID persistence
│   └── pipeline.py          # Main orchestrator
├── models/
│   └── yolov8n.onnx         # Quantized weights for CPU/GPU speed
├── query_images/            # Storage for user-uploaded targets
├── requirements.txt         # Dependencies
└── README.md

---
*Created by [Abdullakim](https://github.com/Abdullakim1)*
