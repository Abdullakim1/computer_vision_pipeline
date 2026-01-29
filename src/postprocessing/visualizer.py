import cv2
import numpy as np

# COCO class names and a color palette for visualization
# We generate a unique color for each class ID.
np.random.seed(42) # for consistent colors
COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

CLASS_NAMES = {i: (name, tuple(map(int, COLORS[i]))) for i, name in enumerate(COCO_CLASSES)}


def draw_detections(frame, detections):
    """
    Draws bounding boxes and labels on a frame based on model detections.

    Args:
        frame (np.ndarray): The frame to draw on. Should be a uint8 BGR image.
        detections (list): A list of detections from the inference engine.

    Returns:
        np.ndarray: The frame with detections drawn on it.
    """
    for det in detections:
        box = det['box']
        track_id = det.get('track_id', None)
        class_id = det.get('class_id', None)

        # Get class name and color, with a fallback for unknown classes
        name, color = CLASS_NAMES.get(class_id, ('unknown', (255, 255, 255)))

        # Bounding box coordinates must be integers for drawing
        x1, y1, x2, y2 = map(int, box)

        # Draw the bounding box rectangle on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare the label text
        label = f'ID {track_id} {name}' if track_id is not None else f'{name}'

        # Calculate text size to draw a background rectangle for the label
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw a filled rectangle behind the text for better readability
        cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)

        # Put the label text on the background rectangle
        cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame
