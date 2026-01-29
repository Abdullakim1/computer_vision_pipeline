import argparse
import cv2
import numpy as np

from data_ingestion.streamer import Streamer
from preprocessing.preprocessor import Preprocessor
from inference.engine import InferenceEngine
from postprocessing.visualizer import draw_detections
from postprocessing.tracker import Sort


def main():
    """Main function to run the computer vision pipeline."""
    parser = argparse.ArgumentParser(description="Computer Vision Pipeline")
    parser.add_argument('--source', type=str, required=True, help='Path to video file or camera index.')
    args = parser.parse_args()

    print(f"Starting pipeline with source: {args.source}")

    try:
        # The 'source' argument can be an integer for a webcam (e.g., 0) or a string for a file path.
        # We need to convert it if it's a digit.
        source = int(args.source) if args.source.isdigit() else args.source

        preprocessor = Preprocessor(output_size=(640, 640))
        engine = InferenceEngine(model_path='models/yolov8n.onnx', confidence_threshold=0.3)
        tracker = Sort()

        with Streamer(source) as stream:
            while True:  # Loop indefinitely until 'q' is pressed or stream ends
                # 1. Ingestion
                frame = stream.get_frame()

                # 2. Preprocessing
                processed_frame_rgb = preprocessor.process(frame)  # This is now a float32, normalized RGB image

                # For visualization, we need to reverse the normalization and color conversion
                # 1. Denormalize from [0, 1] to [0, 255]
                denormalized_frame = (processed_frame_rgb * 255).astype(np.uint8)

                # 2. Convert back from RGB to BGR for OpenCV display
                display_frame = cv2.cvtColor(denormalized_frame, cv2.COLOR_RGB2BGR)

                # 3. Inference
                detections = engine.infer(processed_frame_rgb)

                # 4. Tracking
                # Convert detections to the format expected by the tracker
                dets_for_tracking = np.array([[det['box'][0], det['box'][1], det['box'][2], det['box'][3], det['score']] for det in detections], dtype=np.float32)
                if dets_for_tracking.size == 0:
                    dets_for_tracking = np.empty((0, 5), dtype=np.float32)
                tracked_objects = tracker.update(dets_for_tracking)

                # Convert tracker output back to the format expected by the visualizer
                # Propagate class_id to tracked_detections by matching boxes
                tracked_detections = []
                for x1, y1, x2, y2, track_id in tracked_objects:
                    # Find the detection with the closest box (exact match or highest IoU)
                    class_id = None
                    for det in detections:
                        if [int(det['box'][0]), int(det['box'][1]), int(det['box'][2]), int(det['box'][3])] == [int(x1), int(y1), int(x2), int(y2)]:
                            class_id = det.get('class_id', None)
                            break
                    tracked_detections.append({'box': [int(x1), int(y1), int(x2), int(y2)], 'track_id': int(track_id), 'class_id': class_id})

                # 5. Post-processing & Visualization
                if tracked_detections:
                    # Draw the tracked detections on the display frame
                    display_frame = draw_detections(display_frame, tracked_detections)

                # Display the final frame with detections
                cv2.imshow('Computer Vision Pipeline', display_frame)

                # Press 'q' to exit the loop
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
