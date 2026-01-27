import cv2
import numpy as np
import onnxruntime

class InferenceEngine:
    """Handles model inference using ONNX Runtime for a YOLOv8 model."""

    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.5):
        """
        Initializes the ONNX inference engine.

        Args:
            model_path (str): Path to the ONNX model file.
            confidence_threshold (float): Minimum score for a detection to be considered.
            iou_threshold (float): IoU threshold for Non-Maximum Suppression.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        print(f"Initializing ONNX Runtime session for model: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path)
        print("ONNX Runtime session initialized successfully.")

        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def infer(self, frame):
        """
        Performs inference on a preprocessed frame and returns detections.

        Args:
            frame (np.ndarray): The preprocessed frame (H, W, C), normalized, in RGB.

        Returns:
            list: A list of detections, where each detection is a dictionary.
        """
        # 1. Prepare the input tensor
        # The frame is already resized, padded, and normalized.
        # We just need to change it from HWC to CHW and add a batch dimension.
        input_tensor = frame.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension -> (1, C, H, W)

        # 2. Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # 3. Post-process the output
        detections = self._postprocess(outputs)
        return detections

    def _postprocess(self, outputs):
        """
        Decodes the raw YOLOv8 output, applies NMS, and formats detections.
        """
        # The output of YOLOv8 is a tensor of shape (batch_size, 84, num_proposals)
        # where 84 = 4 (box) + 80 (class scores)
        predictions = np.squeeze(outputs[0]).T  # Transpose to (num_proposals, 84)

        # Filter out detections with low confidence
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.confidence_threshold]
        scores = scores[scores > self.confidence_threshold]

        if predictions.shape[0] == 0:
            return []

        # Get the class with the highest score
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for NMS
        boxes = self._extract_boxes(predictions)

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)

        detections = []
        for i in indices:
            detections.append({
                'box': boxes[i],
                'score': scores[i],
                'class_id': class_ids[i]
            })

        return detections

    def _extract_boxes(self, predictions):
        """
        Converts YOLO's (center_x, center_y, width, height) box format to (x1, y1, x2, y2).
        Also scales the boxes from the model's input size back to the original padded frame size.
        """
        # Extract the boxes from the predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions (it's already 640x640 from preprocessing)
        boxes = self._rescale_boxes(boxes)

        # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
        return [
            [
                int(box[0] - box[2] / 2),
                int(box[1] - box[3] / 2),
                int(box[0] + box[2] / 2),
                int(box[1] + box[3] / 2)
            ] for box in boxes
        ]

    def _rescale_boxes(self, boxes):
        """
        The model output is relative to its input size. Since our preprocessor already
        scales the image to the model's input size, no further scaling is needed here.
        This function is a placeholder for scenarios with more complex preprocessing.
        """
        # Input shape from model is (1, 3, 640, 640)
        # Preprocessed frame is (640, 640, 3)
        # The coordinates are already in the correct scale.
        return boxes

