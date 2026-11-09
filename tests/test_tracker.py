import cv2
import numpy as np

from src.animal_analyzer.tracker import BoxTracker


def detection(x, y, label="animal", size=40):
    return {"label": label, "confidence": 0.9, "box_xyxy": (x, y, x + size, y + size)}


def blank():
    return np.zeros((240, 320), np.uint8)


def test_new_detections_get_new_ids():
    tracker = BoxTracker(match_iou=0.3, max_missed=2)
    visible = tracker.update(blank(), [detection(10, 10), detection(200, 100)])
    assert [t.track_id for t in visible] == [1, 2]


def test_an_overlapping_detection_continues_its_track():
    tracker = BoxTracker(match_iou=0.3, max_missed=2)
    tracker.update(blank(), [detection(10, 10)])
    visible = tracker.update(blank(), [detection(15, 12)])
    assert [(t.track_id, t.hits, t.box) for t in visible] == [(1, 2, (15, 12, 55, 52))]


def test_a_detection_of_another_class_starts_a_new_track():
    tracker = BoxTracker(match_iou=0.3, max_missed=2)
    tracker.update(blank(), [detection(10, 10)])
    visible = tracker.update(blank(), [detection(10, 10, label="person")])
    assert [(t.track_id, t.label) for t in visible] == [(2, "person")]


def test_each_detection_continues_at_most_one_track():
    tracker = BoxTracker(match_iou=0.3, max_missed=2)
    tracker.update(blank(), [detection(10, 10), detection(30, 10)])
    tracker.update(blank(), [detection(20, 10)])
    assert sorted(t.hits for t in tracker.tracks) == [1, 2]


def test_a_missed_track_is_hidden_and_ends_after_max_missed():
    tracker = BoxTracker(match_iou=0.3, max_missed=2)
    tracker.update(blank(), [detection(10, 10)])
    for _ in range(2):
        assert tracker.update(blank(), []) == []
        assert len(tracker.tracks) == 1
    tracker.update(blank(), [])
    assert tracker.tracks == []


def test_a_track_resumes_when_detected_again_within_max_missed():
    tracker = BoxTracker(match_iou=0.3, max_missed=2)
    tracker.update(blank(), [detection(10, 10)])
    tracker.update(blank(), [])
    visible = tracker.update(blank(), [detection(12, 10)])
    assert [(t.track_id, t.hits) for t in visible] == [(1, 2)]


def test_between_detections_the_box_follows_the_optical_flow():
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 255, (40, 40), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)

    def frame(x, y):
        image = np.zeros((240, 320), np.uint8)
        image[y:y + 40, x:x + 40] = texture
        return image

    tracker = BoxTracker(match_iou=0.3, max_missed=2)
    tracker.update(frame(100, 100), [detection(100, 100)])
    visible = tracker.update(frame(104, 103), None)
    x1, y1, _, _ = visible[0].box
    assert abs(x1 - 104) < 0.5 and abs(y1 - 103) < 0.5
    assert not visible[0].detected


def test_without_texture_the_box_stays_in_place():
    tracker = BoxTracker(match_iou=0.3, max_missed=2)
    tracker.update(blank(), [detection(10, 10)])
    visible = tracker.update(blank(), None)
    assert visible[0].box == (10, 10, 50, 50)
