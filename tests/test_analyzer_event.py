"""One event through AnimalAnalyzer._event_loop, with scripted detections and species in place of MegaDetector and SpeciesNet."""
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


class ScriptedClassifier:
    """Names each track by its detector label; records what it was asked."""

    def __init__(self, species):
        self.species = species
        self.calls = []

    def classify(self, crops, detector_label, detector_confidence):
        self.calls.append((len(crops), crops[0].shape, detector_label, detector_confidence))
        prediction = self.species[detector_label]
        return {"prediction": prediction, "taxonomy": prediction, "score": 0.9, "source": "classifier",
                "blank": prediction == "blank", "top_classes": []}


def run_event(config, script, species={"animal": "white-tailed deer", "person": "human"}):
    """Run one event of FRAMES frames; frame i is filled with the value i."""
    analyzer = AnimalAnalyzer(None, config)
    analyzer.logger = __import__("logging").getLogger("test")
    analyzer.detector = ScriptedDetector(script)
    analyzer.detector_info = {"model": "scripted"}
    analyzer.classifier = ScriptedClassifier(species)
    frames = [np.full((48, 64, 3), i, np.uint8) for i in range(FRAMES)]
    analyzer.queue = OneEventQueue(analyzer, frames)
    analyzer.is_running = True
    analyzer._event_loop()
    pending = config["animal_analyzer"]["output_pending_dir"]
    run_event.classifier_calls = analyzer.classifier.calls
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
    assert metadata["event_summary"]["species_list"] == ["white-tailed deer"]
    assert metadata["event_summary"]["primary_species"] == "white-tailed deer"
    assert metadata["detector"] == {"model": "scripted"}
    assert [(c["track_id"], c["detector_label"], c["prediction"], c["crops"]) for c in metadata["classifications"]] == [(1, "animal", "white-tailed deer", 3)]
    # The track was detected in three frames, so SpeciesNet gets three 480x480 crops.
    assert run_event.classifier_calls == [(3, (480, 480, 3), "animal", 0.9)]
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


def test_at_most_classification_crops_crops_are_classified_the_most_confident_first(config):
    config["animal_analyzer"]["classification_crops"] = 2
    analyzer = AnimalAnalyzer(None, config)
    crops = []
    frame = np.zeros((48, 64, 3), np.uint8)
    for confidence in (0.3, 0.9, 0.5, 0.7):
        frame[:] = int(confidence * 100)
        analyzer._keep_crop(crops, confidence, frame, (0, 0, 20, 20))
    assert [(c, crop[0, 0, 0]) for c, crop in crops] == [(0.9, 90), (0.7, 70)]


def test_an_event_speciesnet_calls_blank_is_held_instead_of_uploaded(config):
    files, _, _ = run_event(config, lambda i: [animal()] if 3 <= i < 20 else [], species={"animal": "blank"})
    assert files == []
    held = config["animal_analyzer"]["output_held_dir"]
    held_files = sorted(os.listdir(held))
    assert len(held_files) == 2
    with open(os.path.join(held, held_files[0])) as f:
        metadata = json.load(f)
    assert metadata["event_summary"]["species_list"] == [] and metadata["event_summary"]["primary_species"] == "blank"


def test_an_event_with_one_blank_track_and_one_named_track_is_uploaded(config):
    def script(i):
        return ([animal()] if 3 <= i < 20 else []) + ([animal("person", x=40)] if 8 <= i < 30 else [])
    files, pending, _ = run_event(config, script, species={"animal": "blank", "person": "human"})
    with open(os.path.join(pending, files[0])) as f:
        metadata = json.load(f)
    assert metadata["event_summary"]["species_list"] == ["human"]
    assert [c["prediction"] for c in metadata["classifications"]] == ["blank", "human"]
