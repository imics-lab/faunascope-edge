import pytest


@pytest.fixture
def config(tmp_path):
    """A configuration like config/config.yaml, with all directories under tmp_path."""
    return {
        "motion_detector": {
            "motion_frame_width": 640,
            "min_motion_contour_area": 500,
            "motion_cooldown_seconds": 20.0,
            "blur_kernel_size": [21, 21],
        },
        "animal_analyzer": {
            "device_id": "test-device",
            "location": {"latitude": 29.9, "longitude": -97.9, "timezone": "America/Chicago"},
            "output_fps": 30.0,
            "detector_model_path": "model_weight/MDV6-mit-yolov9-c.onnx",
            "detector_engine_cache_dir": str(tmp_path / "detector_cache"),
            "trigger_classes": ["animal", "person"],
            "confidence_threshold": 0.2,
            "detection_interval_frames": 5,
            "confirmation_detections": 2,
            "track_match_iou": 0.3,
            "track_max_missed_detections": 2,
            "ram_frame_limit": 700,
            "output_pending_dir": str(tmp_path / "pending_upload"),
            "debug_mode": False,
            "debug_temp_dir": str(tmp_path / "temp_debug"),
            "output_temp_dir": str(tmp_path / "temp"),
        },
    }
