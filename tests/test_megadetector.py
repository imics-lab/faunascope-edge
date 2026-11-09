"""MegaDetector's pre- and postprocessing, with a stand-in for the ONNX session."""
import numpy as np

from src.animal_analyzer.megadetector import MegaDetector


class FakeSession:
    """Returns fixed model outputs and keeps the input it was given."""

    def __init__(self, boxes, scores):
        self.outputs = [np.array([boxes], np.float32), np.array([scores], np.float32)]

    def run(self, _, feeds):
        self.image = feeds["image"]
        return self.outputs


def detector(boxes, scores):
    md = MegaDetector.__new__(MegaDetector)
    md.session = FakeSession(boxes, scores)
    md.confidence_threshold = 0.2
    md.iou_threshold = 0.5
    return md


def test_a_640x480_frame_is_padded_to_640x640_without_scaling():
    md = detector([[0, 0, 1, 1]], [[0, 0, 0]])
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, :, 2] = 255  # red in BGR
    md.detect(frame)
    image = md.session.image
    assert image.shape == (1, 3, 640, 640)
    assert image[0, 0, 80, 0] == 1.0 and image[0, 2, 80, 0] == 0.0
    assert image[0, 0, 79, 0] == np.float32(114 / 255)
    assert image[0, 0, 560, 0] == np.float32(114 / 255)


def test_boxes_are_returned_in_frame_pixels():
    md = detector([[100, 180, 200, 280]], [[0.9, 0, 0]])
    [d] = md.detect(np.zeros((480, 640, 3), np.uint8))
    assert d["label"] == "animal" and d["confidence"] == np.float32(0.9)
    assert d["box_xyxy"] == (100, 100, 200, 200)


def test_a_downscaled_frame_maps_boxes_back_to_full_size():
    # 1280x720 is scaled by 0.5 to 640x360 and padded by 140 rows at the top.
    md = detector([[100, 190, 200, 290]], [[0, 0.8, 0]])
    [d] = md.detect(np.zeros((720, 1280, 3), np.uint8))
    assert d["label"] == "person"
    assert d["box_xyxy"] == (200, 100, 400, 300)


def test_overlapping_boxes_of_one_class_are_suppressed_but_not_across_classes():
    boxes = [[100, 180, 200, 280], [105, 185, 205, 285], [100, 180, 200, 280], [400, 180, 500, 280]]
    scores = [[0.9, 0, 0], [0.7, 0, 0], [0, 0, 0.6], [0.1, 0, 0]]
    detections = detector(boxes, scores).detect(np.zeros((480, 640, 3), np.uint8))
    assert [(d["label"], round(d["confidence"], 2)) for d in detections] == [("animal", 0.9), ("vehicle", 0.6)]


def test_nothing_above_the_threshold_gives_no_detections():
    assert detector([[0, 0, 10, 10]], [[0.1, 0.2, 0]]).detect(np.zeros((480, 640, 3), np.uint8)) == []
