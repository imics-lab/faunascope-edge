from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.device_info import StatusCounters
from src.motion_detector.detector import MotionDetector


@pytest.fixture
def detector(config):
    return MotionDetector(config)


def test_frames_are_resized_to_the_motion_width_and_grayscale(detector):
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert detector._preprocess_frame(frame).shape == (360, 640)


def test_only_large_moving_regions_are_significant(detector):
    mask = np.zeros((360, 640), np.uint8)
    mask[100:140, 100:140] = 255   # 1,600 px: an animal
    mask[300:310, 500:510] = 255   # 100 px: an insect
    mask[50, 50] = 255             # sensor noise
    contours = detector._find_significant_contours(detector._clean_mask(mask))
    assert len(contours) == 1


class FakeCamera:
    """Delivers the given frames, then reports the end of the file."""

    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        pass


def test_event_frames_go_to_the_queue_with_their_read_times_and_motion(config):
    still = np.zeros((480, 640, 3), np.uint8)
    moving = still.copy()
    moving[100:200, 100:200] = 255
    # The background model starts empty, so the first frame starts the event; the moving square shows motion again later.
    frames = [still] * 30 + [moving] + [still] * 5
    detector = MotionDetector(config, video_source="file.mp4", status=StatusCounters())
    detector.camera = FakeCamera(frames)
    detector.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    sent = []
    detector.queue = SimpleNamespace(put=sent.append)
    detector.is_running = True
    detector._processing_loop()

    assert len(sent) == len(frames) + 1 and sent[-1] is None
    motion = [m for _, _, _, m in sent[:-1]]
    assert motion[0] and motion[30] and not any(motion[10:30])
    read_at = [t for _, t, _, _ in sent[:-1]]
    boot_seconds = [b for _, _, b, _ in sent[:-1]]
    assert read_at == sorted(read_at) and boot_seconds == sorted(boot_seconds)
    assert detector.status.camera_frame_at.value >= boot_seconds[-1]
