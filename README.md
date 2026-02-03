# Real-Time Computer Vision Pipeline

A modular, production-oriented stack built around ONNX Runtime YOLOv8 inference, SORT multi-object tracking, and CLIP-based object re-identification. The pipeline ingests live or recorded video, preprocesses frames, runs detection + tracking, and—when requested—re-identifies user-specified objects on the fly.

---

## 📦 Key Features

| Capability | Description |
|------------|-------------|
| **Data Ingestion** | Threaded `Streamer` captures frames from webcams, files, or network streams while buffering to avoid drops. |
| **Preprocessing** | Letterbox resize to `640×640`, BGR→RGB conversion, and normalization to `[0, 1]` via the `Preprocessor`. |
| **Inference** | ONNX Runtime executes a YOLOv8 ONNX model for fast CPU/GPU detections. |
| **Tracking** | SORT assigns persistent IDs to detections across frames. |
| **Re-Identification** | CLIP (`ViT-B-32`) feature extractor filters detections by similarity to a user-provided query image. |
| **Visualization** | Bounding boxes include track IDs, class labels, confidence scores, and similarity values when available. |
| **API Support** | FastAPI endpoint (`/upload-query-image/`) allows remote upload of query images. |

---

## 📁 Project Layout

computer-vision-pipeline/ ├── README.md ├── requirements.txt ├── src/ │ ├── api.py # FastAPI upload endpoint │ ├── data_ingestion/ # Streamer implementation │ ├── feature_extraction/ # CLIP feature extractor wrapper │ ├── inference/ # YOLOv8 ONNX inference engine │ ├── postprocessing/ # Visualizer + SORT tracker │ ├── preprocessing/ # Letterbox + normalization │ └── main.py # Pipeline entry point └── models/ └── yolov8n.onnx # Detector weights (not versioned)


---

## ⚙️ Installation

1. **Create / activate** a Python virtual environment (recommended).
2. **Install system codecs** (FFmpeg/GStreamer) required by OpenCV if not already present.
3. **Install Python packages**:
   ```bash
   pip install -r requirements.txt


4. Download a YOLOv8 ONNX model (e.g., yolov8n.onnx) and place it under models/.
Optional: Create a query_images/ directory if you plan to upload reference images via the API (it will be created automatically when needed).

▶️ Usage
1. Detection + Tracking

python3 src/main.py --source 0                      # webcam
python3 src/main.py --source /path/to/video.mp4     # local file
python3 src/main.py --source http://example.com/stream.m3u8

2. Re-Identification Mode
Add a reference image of the object you want to find. The pipeline embeds each detection with CLIP and keeps only those above the similarity threshold

python3 src/main.py \
    --source /path/to/video.mp4 \
    --query_image /path/to/query.jpg \
    --similarity_threshold 0.30


Arguments

--source – Video source (file path, URL, or camera index).
--query_image – Path to the reference image for re-identification.
--similarity_threshold – Cosine similarity cutoff (0–1); lower values are more permissive.
💡 Running on CPU-only hardware is fully supported, though CLIP extraction may become the bottleneck. For higher throughput, consider GPU acceleration, cropping from the original frame resolution, or caching embeddings for repeated detections.

🌐 FastAPI Query Image Upload (Optional)
Serve the upload endpoint:

uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Upload an image via HTTP:

curl -F "file=@bird.jpg" http://localhost:8000/upload-query-image/
The response returns the stored path (e.g., query_images/bird.jpg), which you can pass to --query_image when running the pipeline.


🛠️ Configuration Notes
Model Path – Update InferenceEngine initialization in src/main.py if you use a different ONNX model filename or location.
Tracker Tuning – Adjust max_age, min_hits, and iou_threshold in src/postprocessing/tracker.py to suit your scenario.
Visualization – Customize label formatting/colors in src/postprocessing/visualizer.py.
Re-ID Optimizations – Crop detections from the original-resolution frame, down-sample crops, or cache embeddings for speed/accuracy trade-offs.

🚀 Roadmap Ideas
Persist detections/tracks to JSON/CSV and export cropped frames.
Automatically consume query images uploaded via the API without manual CLI input.
Package the pipeline as a REST/gRPC microservice for remote inference.
Support multi-stream ingestion and distributed deployment.