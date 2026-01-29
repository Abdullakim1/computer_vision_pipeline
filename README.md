# Computer Vision Pipeline

This project is a comprehensive computer vision pipeline designed for real-world applications. It handles everything from data ingestion and preprocessing to model inference and post-processing.

## Features

- **Data Ingestion**: Supports various camera types and streaming protocols.
- **Preprocessing**: Includes geometric and color space transformations, and normalization.
- **Inference Engine**: Optimized for high-performance with support for TensorRT and OpenVINO.
- **Post-Processing**: Implements logic like NMS and object tracking.
- **Deployment**: Ready for containerization and scaling with Docker and FastAPI/Triton.

## Project Structure

```
computer-vision-pipeline/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_ingestion/
│   ├── preprocessing/
│   ├── inference/
│   ├── postprocessing/
│   └── main.py
└── ...
```
Primary webcam
python3 src/main.py --source 0

local video
python3 src/main.py --source /path/video.mp4