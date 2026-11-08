import numpy as np
import pytest

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
