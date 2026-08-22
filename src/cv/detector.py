"""YOLOv8 detection via ONNX Runtime, with graceful degrade when optional
model/engine files are absent (returns empty predictions rather than crash).

Kept dependency-light: ONNX Runtime is only imported if actually present.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:  # pragma: no cover
    ort = None
    _HAS_ORT = False


def engine_available():
    return _HAS_ORT


class ObjectDetector:
    """Thin wrapper around an ONNX YOLOv8 session.

    Parameters
    ----------
    model_path : str
        Path to ``.onnx`` weights (e.g. ``models/yolov8n.onnx``).
    confidence_threshold : float
        Minimum class score to keep a proposal.
    iou_threshold : float
        NMS overlap threshold.
    """

    def __init__(self, model_path, confidence_threshold=0.5, iou_threshold=0.5, labels=None):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.labels = labels or _COCO80
        if not _HAS_ORT:
            warnings.warn("onnxruntime not installed; detector disabled", stacklevel=2)
            self.session = None
            return
        try:
            self.session = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"could not load {model_path}: {exc}", stacklevel=2)
            self.session = None

    @property
    def available(self):
        return self.session is not None

    # ------------------------------------------------------------------
    def infer(self, tensor01):
        """``tensor01`` is a HWC, [0,1] RGB float32 array (see Preprocessor)."""
        if not self.available:
            return []
        inp = tensor01.transpose(2, 0, 1)[None].astype(np.float32)
        name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {name: inp})
        return self._postprocess(outputs)

    def _postprocess(self, outputs):
        row = np.squeeze(outputs[0])  # (84, N) for COCO
        if row.ndim != 2:
            return []
        preds = row.T  # (N, 84)
        scores = preds[:, 4:].max(axis=1)
        keep = scores > self.confidence_threshold
        preds, scores = preds[keep], scores[keep]
        if not preds.size:
            return []
        class_ids = preds[:, 4:].argmax(axis=1)
        boxes = np.asarray(
            [self._cxcywh_to_xyxy(b) for b in preds[:, :4]], dtype=np.float32
        )
        order = scores.argsort()[::-1]
        boxes, scores, class_ids = boxes[order], scores[order], class_ids[order]

        picked = []
        for i in range(len(scores)):
            if scores[i] == 0:
                continue
            picked.append(i)
            x1, y1, x2, y2 = boxes[i]
            area = (x2 - x1) * (y2 - y1)
            for j in range(i + 1, len(scores)):
                if scores[j] == 0:
                    continue
                xx1, yy1 = max(x1, boxes[j][0]), max(y1, boxes[j][1])
                xx2, yy2 = min(x2, boxes[j][2]), min(y2, boxes[j][3])
                inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                union = area + (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - inter
                if inter / (union + 1e-9) > self.iou_threshold:
                    scores[j] = 0.0
        picked.sort()

        return [
            {
                "box": [int(v) for v in boxes[i]],
                "score": float(scores[i]),
                "class_id": int(class_ids[i]),
                "label": self.labels[int(class_ids[i])] if int(class_ids[i]) < len(self.labels) else "obj",
            }
            for i in picked
        ]

    @staticmethod
    def _boxeswh_to_xyxy(b):
        cx, cy, w, h = b
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def nms(boxes, scores, iou_threshold):
    """NMS that tolerates no GPU and matches the classic greedy approach."""
    if not len(boxes):
        return []
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep, sup = [], set()
    for i_order, idx in enumerate(order):
        if idx in sup:
            continue
        keep.append(int(idx))
        if i_order == len(order) - 1:
            break
        xx1 = np.maximum(x1[idx], x1[order])
        yy1 = np.maximum(y1[idx], y1[order])
        xx2 = np.minimum(x2[idx], x2[order])
        yy2 = np.minimum(y2[idx], y2[order])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[idx] + areas[order] - inter
        iou = inter / (union + 1e-9)
        iou[iou < 0] = 0
        sup.update(order[iou > iou_threshold].tolist())
    return keep


_COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]