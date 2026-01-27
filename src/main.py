import argparse
import cv2

from data_ingestion.streamer import Streamer
from preprocessing.preprocessor import Preprocessor
from inference.engine import InferenceEngine
from postprocessing.visualizer import draw_detections

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
                # 4. Post-processing & Visualization
                if detections:
                    print(f"Detected {len(detections)} objects: {detections}")
                    # Draw the detections on the display frame
                    display_frame = draw_detections(display_frame, detections)

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
