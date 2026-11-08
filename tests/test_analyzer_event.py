"""One event through AnimalAnalyzer._event_loop, with scripted detections in place of YOLO."""
import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pytest

from src.animal_analyzer.analyzer import AnimalAnalyzer

FRAMES = 30


class OneEventQueue:
    """Delivers one event's frames, then stops the analyzer."""

    def __init__(self, analyzer, frames):
        self.analyzer = analyzer
        self.items = list(frames) + [None]

    def get(self):
        if self.items:
            return self.items.pop(0)
        self.analyzer.is_running = False
        return None


def run_event(config, script):
    """Run one event of FRAMES frames; script(i) gives the detections in frame i."""
    analyzer = AnimalAnalyzer(None, config)
    analyzer.logger = __import__("logging").getLogger("test")
    analyzer.model = lambda frame, verbose: frame
    analyzer._parse_yolo_results = lambda frame: script(int(frame[0, 0, 0]))
    frames = [np.full((48, 64, 3), i, np.uint8) for i in range(FRAMES)]
    analyzer.queue = OneEventQueue(analyzer, frames)
    analyzer.is_running = True
    analyzer._event_loop()
    pending = config["animal_analyzer"]["output_pending_dir"]
    return sorted(os.listdir(pending)), pending


def fox(x=32, y=24):
    return {"label": "Fox", "confidence": 0.9, "centroid": (x, y), "box_xywh": (x, y, 10, 10)}


def test_a_confirmed_animal_produces_a_video_and_metadata(config):
    files, pending = run_event(config, lambda i: [fox()] if 3 <= i < 20 else [])
    assert len(files) == 2 and files[0].endswith(".json") and files[1].endswith(".mp4")

    with open(os.path.join(pending, files[0])) as f:
        metadata = json.load(f)
    assert metadata["eventId"] == files[0][:-len(".json")]
    assert metadata["deviceId"] == "test-device"
    assert metadata["event_summary"]["species_list"] == ["Fox"]
    assert metadata["event_summary"]["primary_species"] == "Fox"
    # Detections are recorded from the frame that confirms the track (its fifth sighting).
    assert [d["frame_index"] for d in metadata["detections"]] == list(range(7, 20))
    assert datetime.fromisoformat(metadata["timestamp_start_utc"]).utcoffset().total_seconds() == 0
    assert metadata["video_duration_seconds"] == pytest.approx(FRAMES / 30, abs=0.1)

    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", os.path.join(pending, files[1])],
        capture_output=True, text=True, check=True).stdout
    assert int(frames) == FRAMES


def test_an_animal_seen_too_briefly_is_discarded(config):
    files, _ = run_event(config, lambda i: [fox()] if 3 <= i < 7 else [])
    assert files == []
    assert os.listdir(config["animal_analyzer"]["output_temp_dir"]) == []


def test_species_outside_the_list_are_discarded(config):
    files, _ = run_event(config, lambda i: [dict(fox(), label="Domestic_Cow")] if 3 <= i < 20 else [])
    assert files == []
