"""One event through AnimalAnalyzer._event_loop, with scripted detections and species in place of MegaDetector and SpeciesNet."""
import json
import os
import subprocess
import time

import numpy as np

from src.animal_analyzer.analyzer import AnimalAnalyzer, measured_frame_rate, summarize

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
        name = self.species[detector_label]
        return {"taxon": f"uuid;;;;;;{name}", "name": name, "score": 0.9, "source": "classifier",
                "blank": name == "blank", "top": []}


START = 1_700_000_000.0   # the first frame's UTC time
BOOT = 1000.0             # and its time since boot
FPS = 25


def run_event(config, script, species={"animal": "white-tailed deer", "person": "human"}):
    """Run one event of FRAMES frames captured at FPS, with motion in the first ten; frame i is filled with the value i."""
    analyzer = AnimalAnalyzer(None, config)
    analyzer.logger = __import__("logging").getLogger("test")
    analyzer.detector = ScriptedDetector(script)
    analyzer.models = {"detector": {"name": "scripted"}}
    analyzer.classifier = ScriptedClassifier(species)
    frames = [(np.full((48, 64, 3), i, np.uint8), START + i / FPS, BOOT + i / FPS, i < 10) for i in range(FRAMES)]
    analyzer.queue = OneEventQueue(analyzer, frames)
    analyzer.is_running = True
    analyzer._event_loop()
    pending = config["animal_analyzer"]["output_pending_dir"]
    run_event.classifier_calls = analyzer.classifier.calls
    run_event.counts = analyzer.status.take()
    return sorted(os.listdir(pending)), pending, analyzer.detector.frames_run


def read_metadata(directory, files):
    with open(os.path.join(directory, files[0])) as f:
        return json.load(f)


def animal(label="animal", x=20, y=10):
    return {"label": label, "confidence": 0.9, "box_xyxy": (x, y, x + 10, y + 10)}


def test_the_detector_runs_on_every_nth_frame_from_the_first(config):
    _, _, frames_run = run_event(config, lambda i: [])
    assert frames_run == [0, 5, 10, 15, 20, 25]


def test_a_confirmed_animal_produces_a_video_and_metadata(config):
    files, pending, _ = run_event(config, lambda i: [animal()] if 3 <= i < 20 else [])
    assert len(files) == 2 and files[0].endswith(".json") and files[1].endswith(".mp4")
    metadata = read_metadata(pending, files)

    assert metadata["format_version"] == 2
    assert metadata["event_id"] == files[0][:-len(".json")]
    assert metadata["event_id"].startswith("event_" + time.strftime("%Y%m%d-%H%M%S", time.localtime(START)))
    assert metadata["device_id"] == "test-device"
    assert metadata["edge_decision"] == "upload"
    assert metadata["models"] == {"detector": {"name": "scripted"}}
    assert metadata["settings"]["tracker"] == {"match_iou": 0.3, "max_missed_detections": 2, "confirmation_detections": 2}
    assert len(metadata["software"]["source_sha256"]) == 64
    assert metadata["clock"]["boot_seconds_at_start"] == BOOT
    assert metadata["time"] == {"start_utc": "2023-11-14T22:13:20.000Z", "end_utc": "2023-11-14T22:13:21.160Z", "timezone": "America/Chicago"}
    assert metadata["video"] == {"file": files[1], "width": 64, "height": 48, "frames": FRAMES, "frames_dropped": 0,
                                 "frame_rate": 25.0, "duration_seconds": 1.2, "frame_offsets_ms": [40 * i for i in range(FRAMES)]}
    assert metadata["motion"] == {"last_motion_frame": 9}

    # Detected in frames 5, 10, and 15 and confirmed in frame 10; its boxes are recorded from its first detection until the detection frame that misses it.
    [track] = metadata["tracks"]
    assert (track["track_id"], track["class"], track["first_frame"], track["last_frame"], track["confirmed_frame"]) == (1, "animal", 5, 19, 10)
    boxes = track["boxes"]
    assert boxes["frame"] == list(range(5, 20))
    assert [f for f, d in zip(boxes["frame"], boxes["detected"]) if d] == [5, 10, 15]
    assert [c for c in boxes["confidence"] if c is not None] == [0.9, 0.9, 0.9] and boxes["confidence"][1] is None
    assert boxes["xyxy"][0] == [20, 10, 30, 20]
    # SpeciesNet gets the track's three 480x480 crops.
    assert run_event.classifier_calls == [(3, (480, 480, 3), "animal", 0.9)]
    assert track["species"]["name"] == "white-tailed deer" and track["species"]["crop_frames"] == [5, 10, 15]
    assert metadata["summary"] == {"species": [{"taxon": "uuid;;;;;;white-tailed deer", "name": "white-tailed deer", "tracks": 1, "max_count": 1}],
                                   "primary_species": "white-tailed deer"}
    assert run_event.counts["motion"] == 1 and run_event.counts["kept"] == 1

    # The video plays at the rate the frames were captured.
    stream = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "csv=p=0", os.path.join(pending, files[1])],
        capture_output=True, text=True, check=True).stdout.strip()
    assert stream == f"25/1,{FRAMES}"


def test_the_frame_rate_is_measured_from_the_frame_times():
    assert measured_frame_rate([0, 40, 100], 30.0) == 20.0
    assert measured_frame_rate([0], 30.0) == 30.0


def test_an_animal_detected_once_is_discarded(config):
    files, _, _ = run_event(config, lambda i: [animal()] if 3 <= i < 7 else [])
    assert files == []
    assert os.listdir(config["animal_analyzer"]["output_temp_dir"]) == []
    assert run_event.counts["discarded"] == 1


def test_classes_outside_the_triggers_are_discarded(config):
    files, _, _ = run_event(config, lambda i: [animal("vehicle")] if 3 <= i < 20 else [])
    assert files == []


def test_only_confirmed_tracks_are_written(config):
    def script(i):
        detections = [animal()] if 3 <= i < 20 else []
        return detections + ([animal("person", x=40)] if i == 25 else [])
    files, pending, _ = run_event(config, script)
    assert [t["class"] for t in read_metadata(pending, files)["tracks"]] == ["animal"]


def test_two_animals_of_one_species_count_two(config):
    files, pending, _ = run_event(config, lambda i: [animal(), animal(x=45, y=30)] if 3 <= i < 20 else [])
    [entry] = read_metadata(pending, files)["summary"]["species"]
    assert (entry["name"], entry["tracks"], entry["max_count"]) == ("white-tailed deer", 2, 2)


def test_at_most_classification_crops_crops_are_classified_the_most_confident_first(config):
    config["animal_analyzer"]["classification_crops"] = 2
    analyzer = AnimalAnalyzer(None, config)
    crops = []
    frame = np.zeros((48, 64, 3), np.uint8)
    for i, confidence in enumerate((0.3, 0.9, 0.5, 0.7)):
        frame[:] = int(confidence * 100)
        analyzer._keep_crop(crops, confidence, i, frame, (0, 0, 20, 20))
    assert [(c, i, crop[0, 0, 0]) for c, i, crop in crops] == [(0.9, 1, 90), (0.7, 3, 70)]


def test_an_event_speciesnet_calls_blank_is_held_instead_of_uploaded(config):
    files, _, _ = run_event(config, lambda i: [animal()] if 3 <= i < 20 else [], species={"animal": "blank"})
    assert files == []
    held = config["animal_analyzer"]["output_held_dir"]
    held_files = sorted(os.listdir(held))
    assert len(held_files) == 2
    metadata = read_metadata(held, held_files)
    assert metadata["edge_decision"] == "held_blank"
    assert metadata["summary"] == {"species": [], "primary_species": "blank"}
    assert run_event.counts["held_blank"] == 1


def test_an_event_with_one_blank_track_and_one_named_track_is_uploaded(config):
    def script(i):
        return ([animal()] if 3 <= i < 20 else []) + ([animal("person", x=40)] if 8 <= i < 30 else [])
    files, pending, _ = run_event(config, script, species={"animal": "blank", "person": "human"})
    metadata = read_metadata(pending, files)
    assert [s["name"] for s in metadata["summary"]["species"]] == ["human"]
    assert [t["species"]["name"] for t in metadata["tracks"]] == ["blank", "human"]


def test_a_named_species_is_primary_over_an_unplaced_animal():
    def track(taxon, frames):
        return {"species": {"taxon": taxon, "name": taxon.split(";")[-1], "blank": False}, "boxes": {"frame": list(frames)}}
    unplaced = "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal"
    summary = summarize([track(unplaced, range(100)), track("t;;;;;;wild turkey", range(10)), track("b;;;;;;bird", range(20))])
    assert [s["name"] for s in summary["species"]] == ["bird", "wild turkey", "animal"]
    assert summary["primary_species"] == "bird"
    assert summarize([track(unplaced, range(5))])["primary_species"] == "animal"
