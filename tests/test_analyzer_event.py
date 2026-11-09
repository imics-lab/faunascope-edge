"""One event through AnimalAnalyzer._event_loop, with scripted detections in place of MegaDetector."""
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


class ScriptedDetector:
    """script(i) gives the detections in frame i; records which frames were run."""

    def __init__(self, script):
        self.script = script
        self.frames_run = []

    def detect(self, frame):
        i = int(frame[0, 0, 0])
        self.frames_run.append(i)
        return self.script(i)


def run_event(config, script):
    """Run one event of FRAMES frames; frame i is filled with the value i."""
    analyzer = AnimalAnalyzer(None, config)
    analyzer.logger = __import__("logging").getLogger("test")
    analyzer.detector = ScriptedDetector(script)
    analyzer.detector_info = {"model": "scripted"}
    frames = [np.full((48, 64, 3), i, np.uint8) for i in range(FRAMES)]
    analyzer.queue = OneEventQueue(analyzer, frames)
    analyzer.is_running = True
    analyzer._event_loop()
    pending = config["animal_analyzer"]["output_pending_dir"]
    return sorted(os.listdir(pending)), pending, analyzer.detector.frames_run


def animal(label="animal", x=20, y=10):
    return {"label": label, "confidence": 0.9, "box_xyxy": (x, y, x + 10, y + 10)}


def test_the_detector_runs_on_every_nth_frame_from_the_first(config):
    _, _, frames_run = run_event(config, lambda i: [])
    assert frames_run == [0, 5, 10, 15, 20, 25]


def test_a_confirmed_animal_produces_a_video_and_metadata(config):
    files, pending, _ = run_event(config, lambda i: [animal()] if 3 <= i < 20 else [])
    assert len(files) == 2 and files[0].endswith(".json") and files[1].endswith(".mp4")

    with open(os.path.join(pending, files[0])) as f:
        metadata = json.load(f)
    assert metadata["eventId"] == files[0][:-len(".json")]
    assert metadata["deviceId"] == "test-device"
    assert metadata["event_summary"]["species_list"] == ["animal"]
    assert metadata["event_summary"]["primary_species"] == "animal"
    assert metadata["detector"] == {"model": "scripted"}
    # Detected in frames 5, 10, and 15 and confirmed in frame 10; its boxes are recorded from its first detection until the detection frame that misses it.
    assert [d["frame_index"] for d in metadata["detections"]] == list(range(5, 20))
    assert {d["track_id"] for d in metadata["detections"]} == {1}
    assert [d["frame_index"] for d in metadata["detections"] if d["detected"]] == [5, 10, 15]
    assert metadata["detections"][0]["box_xywh"] == [25, 15, 10, 10]
    assert datetime.fromisoformat(metadata["timestamp_start_utc"]).utcoffset().total_seconds() == 0
    assert metadata["video_duration_seconds"] == pytest.approx(FRAMES / 30, abs=0.1)

    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", os.path.join(pending, files[1])],
        capture_output=True, text=True, check=True).stdout
    assert int(frames) == FRAMES


def test_an_animal_detected_once_is_discarded(config):
    files, _, _ = run_event(config, lambda i: [animal()] if 3 <= i < 7 else [])
    assert files == []
    assert os.listdir(config["animal_analyzer"]["output_temp_dir"]) == []


def test_classes_outside_the_triggers_are_discarded(config):
    files, _, _ = run_event(config, lambda i: [animal("vehicle")] if 3 <= i < 20 else [])
    assert files == []


def test_only_confirmed_tracks_are_written(config):
    def script(i):
        detections = [animal()] if 3 <= i < 20 else []
        return detections + ([animal("person", x=40)] if i == 25 else [])
    files, pending, _ = run_event(config, script)
    with open(os.path.join(pending, files[0])) as f:
        metadata = json.load(f)
    assert {d["label"] for d in metadata["detections"]} == {"animal"}
