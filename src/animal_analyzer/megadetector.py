import os

import cv2
import numpy as np
from typing import Dict, List

CLASS_NAMES = ('animal', 'person', 'vehicle')
INPUT_SIZE = 640
IOU_THRESHOLD = 0.5


class MegaDetector:
    """
    MegaDetector V6 (MDV6-mit-yolov9-c) in the ONNX form written by scripts/export_megadetector.py. It finds animals, people, and vehicles; it does not name species.

    Preprocessing and suppression of overlapping boxes follow Pytorch-Wildlife: the frame is scaled to fit 640x640 and padded with gray, and boxes of the same class that overlap by more than iou_threshold are suppressed.
    """
    def __init__(self, model_path: str, confidence_threshold: float, engine_cache_dir: str, iou_threshold: float = IOU_THRESHOLD):
        import onnxruntime
        available = onnxruntime.get_available_providers()
        providers = []
        if 'TensorrtExecutionProvider' in available:
            # Building the TensorRT engine takes about 8 minutes on a Jetson Orin Nano, so it is kept for the next start.
            os.makedirs(engine_cache_dir, exist_ok=True)
            providers.append(('TensorrtExecutionProvider', {
                'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': engine_cache_dir}))
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detections in a BGR frame: label, confidence, and box (x1, y1, x2, y2) in frame pixels."""
        h, w = frame.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        left, top = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
        image = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        image[top:top + new_h, left:left + new_w] = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        boxes, scores = self.session.run(None, {'image': image})
        boxes, scores = boxes[0], scores[0]
        anchors, classes = np.nonzero(scores > self.confidence_threshold)
        if len(anchors) == 0:
            return []
        confidences = scores[anchors, classes]
        boxes = (boxes[anchors] - [left, top, left, top]) / scale

        # Shifting each class to its own region keeps suppression within a class.
        shifted = boxes + (classes * 4 * INPUT_SIZE / scale)[:, None]
        keep = cv2.dnn.NMSBoxes((np.concatenate([shifted[:, :2], shifted[:, 2:] - shifted[:, :2]], axis=1)).tolist(),
                                confidences.tolist(), self.confidence_threshold, self.iou_threshold)
        detections = []
        for i in np.array(keep).flatten():
            x1, y1, x2, y2 = boxes[i]
            detections.append({
                'label': CLASS_NAMES[classes[i]],
                'confidence': float(confidences[i]),
                'box_xyxy': (max(0.0, float(x1)), max(0.0, float(y1)), min(float(w), float(x2)), min(float(h), float(y2))),
            })
        return sorted(detections, key=lambda d: -d['confidence'])
