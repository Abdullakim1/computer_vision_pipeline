import argparse
import cv2
import numpy as np
from PIL import Image

from data_ingestion.streamer import Streamer
from preprocessing.preprocessor import Preprocessor
from inference.engine import InferenceEngine
from postprocessing.visualizer import draw_detections
from postprocessing.tracker import Sort
from feature_extraction.extractor import FeatureExtractor

def main():
    """Main function to run the computer vision pipeline with Re-ID."""
    parser = argparse.ArgumentParser(description="Computer Vision Pipeline with Re-ID")
    parser.add_argument('--source', type=str, required=True, help='Path to video file or camera index.')
    parser.add_argument('--query_image', type=str, required=True, help='Path to the query image for re-identification.')
    parser.add_argument('--similarity_threshold', type=float, default=0.8, help='Threshold for matching detections to the query image.')
    args = parser.parse_args()

    print(f"Starting pipeline with source: {args.source}")
    print(f"Query image: {args.query_image}, Similarity Threshold: {args.similarity_threshold}")

    try:
        source = int(args.source) if args.source.isdigit() else args.source

        # Initialize pipeline components
        preprocessor = Preprocessor(output_size=(640, 640))
        engine = InferenceEngine(model_path='models/yolov8n.onnx', confidence_threshold=0.3)
        tracker = Sort()
        feature_extractor = FeatureExtractor()

        # Load and process the query image
        query_image = Image.open(args.query_image).convert("RGB")
        query_features = feature_extractor.extract_features(query_image)
        print("Query features extracted successfully.")

        with Streamer(source) as stream:
            while True:
                frame = stream.get_frame()
                original_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Preprocess for inference
                processed_frame_rgb = preprocessor.process(frame)
                
                # Inference
                detections = engine.infer(processed_frame_rgb)

                # Re-identification and Filtering
                matched_detections = []
                for det in detections:
                    x1, y1, x2, y2 = map(int, det['box'])
                    
                    # Crop the detected object from the *original* frame for feature extraction
                    # Note: This requires scaling the box coordinates back to the original frame size.
                    # For simplicity, we'll crop from the processed frame, but this can be improved.
                    cropped_img_np = processed_frame_rgb[y1:y2, x1:x2]
                    if cropped_img_np.size == 0: continue
                    
                    cropped_img_pil = Image.fromarray((cropped_img_np * 255).astype(np.uint8))
                    
                    # Extract features and compare with the query
                    det_features = feature_extractor.extract_features(cropped_img_pil)
                    similarity = FeatureExtractor.cosine_similarity(query_features, det_features)
                    
                    if similarity > args.similarity_threshold:
                        det['similarity'] = similarity
                        matched_detections.append(det)

                # Tracking matched detections
                dets_for_tracking = np.array([[d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['score']] for d in matched_detections], dtype=np.float32)
                if dets_for_tracking.size == 0:
                    dets_for_tracking = np.empty((0, 5), dtype=np.float32)
                tracked_objects = tracker.update(dets_for_tracking)

                # Prepare for visualization
                vis_detections = []
                for x1, y1, x2, y2, track_id in tracked_objects:
                    class_id = None # Class ID is less relevant in Re-ID context
                    vis_detections.append({'box': [int(x1), int(y1), int(x2), int(y2)], 'track_id': int(track_id), 'class_id': class_id})

                # Visualization
                display_frame = cv2.cvtColor((processed_frame_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                if vis_detections:
                    display_frame = draw_detections(display_frame, vis_detections)

                cv2.imshow('Computer Vision Pipeline', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except IOError as e:
        print(e)
    except StopIteration:
        print("End of video stream.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
