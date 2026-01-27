import cv2
import numpy as np

class Preprocessor:
    """Handles image preprocessing tasks like resizing and color conversion."""

    def __init__(self, output_size=(640, 640)):
        """
        Initializes the preprocessor.

        Args:
            output_size (tuple): The target output size for frames (width, height).
        """
        # Ensure output_size is a tuple of (width, height)
        assert isinstance(output_size, tuple) and len(output_size) == 2
        self.output_size = output_size

    def process(self, frame):
        """
        Applies letterbox resizing, color conversion, and normalization.

        Args:
            frame (np.ndarray): The input frame (H, W, C) in BGR format.

        Returns:
            np.ndarray: The preprocessed frame (H, W, C) in RGB format,
                        resized, padded, and normalized to [0, 1].
        """
        # 1. Letterbox Resizing (maintaining aspect ratio)
        h, w, _ = frame.shape
        target_w, target_h = self.output_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a new image (canvas) with the target size and a neutral gray background
        # The padding color (114, 114, 114) is a common choice.
        padded_frame = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        # Calculate padding dimensions
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        # Paste the resized image onto the canvas
        padded_frame[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_frame

        # 2. Color Space Conversion: BGR to RGB
        rgb_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)

        # 3. Normalization: Scale pixel values from [0, 255] to [0.0, 1.0]
        normalized_frame = rgb_frame.astype(np.float32) / 255.0

        return normalized_frame
