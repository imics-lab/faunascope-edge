"""
Runs the animal analyzer (MegaDetector and SpeciesNet, with the settings in config/config.yaml) over recorded event videos and writes one JSON result per video, for comparing the edge models with the archive's labels and tuning their settings.

Each video is treated as one motion event. Videos are not re-encoded. Besides the metadata the analyzer would write, each result records every MegaDetector detection down to --raw-threshold on the detection frames, so other confidence thresholds and tracking settings can be evaluated later without running the detector again. A result that already exists is skipped, so an interrupted run can be restarted.

    python scripts/replay_archive.py --videos <folder of .mp4> --output <results folder> [--workers 4]
"""
import argparse
import glob
import json
import logging
import multiprocessing
import os
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.animal_analyzer.analyzer import AnimalAnalyzer  # noqa: E402
from src.animal_analyzer.megadetector import MegaDetector  # noqa: E402
from src.animal_analyzer.species_classifier import SpeciesClassifier  # noqa: E402


class RecordingDetector:
    """Runs MegaDetector at a lower threshold, records every detection, and passes on those above the analyzer's threshold."""

    def __init__(self, detector, threshold):
        self.detector = detector
        self.threshold = threshold
        self.frame_index = 0
        self.raw = []

    def detect(self, frame):
        detections = self.detector.detect(frame)
        self.raw.append({"frame_index": self.frame_index, "detections": [
            {"label": d["label"], "confidence": round(d["confidence"], 4), "box_xyxy": [round(v, 1) for v in d["box_xyxy"]]}
            for d in detections]})
        return [d for d in detections if d["confidence"] >= self.threshold]


class VideoQueue:
    """Delivers a video's frames as one event, then stops the analyzer; tells the recorder the frame index."""

    def __init__(self, analyzer, path, recorder):
        self.analyzer = analyzer
        self.capture = cv2.VideoCapture(path)
        self.recorder = recorder
        self.index = -1
        self.done = False

    def get(self):
        if self.done:
            self.analyzer.is_running = False
            return None
        ok, frame = self.capture.read()
        if not ok:
            self.done = True
            self.capture.release()
            return None
        self.index += 1
        self.recorder.frame_index = self.index
        return frame


def worker(args):
    videos, output, raw_threshold, engine_cache_dir = args
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'config', 'config.yaml')) as f:
        config = yaml.safe_load(f.read().replace('${', 'UNSET{'))
    settings = config['animal_analyzer']
    scratch = os.path.join(output, '.scratch', str(os.getpid()))
    settings.update(device_id='archive-replay', location={'latitude': 0.0, 'longitude': 0.0},
                    output_pending_dir=scratch + '/pending', output_held_dir=scratch + '/held', output_temp_dir=scratch, debug_temp_dir=scratch,
                    debug_mode=False, engine_cache_dir=engine_cache_dir)
    for key in ('detector_model_path', 'classifier_model_dir'):
        settings[key] = os.path.join(root, settings[key])

    logging.basicConfig(level=logging.WARNING)
    analyzer = AnimalAnalyzer(None, config)
    analyzer.logger = logging.getLogger('replay')
    # Load the models as AnimalAnalyzer.start() does, without entering its loop.
    detector = MegaDetector(settings['detector_model_path'], raw_threshold, engine_cache_dir)
    analyzer.classifier = SpeciesClassifier(settings['classifier_model_dir'], engine_cache_dir,
                                            settings['geofence_country'], settings['geofence_admin1_region'])
    analyzer.detector_info = {"model": os.path.basename(settings['detector_model_path']), "confidence_threshold": settings['confidence_threshold'],
                              "detection_interval_frames": settings['detection_interval_frames'], "confirmation_detections": settings['confirmation_detections']}
    analyzer.classifier_info = {"model": f"SpeciesNet {analyzer.classifier.version}", "crops_per_track": settings['classification_crops']}

    # The analyzer's video writing is replaced: the result only needs the metadata.
    written = {}
    analyzer._write_video_part = lambda event_id, frames, part_num: 'part' if frames else ''
    analyzer._concatenate_parts = lambda event_id, parts, output_dir: os.path.join(output_dir, event_id + '.mp4')
    analyzer._cleanup_temp_files = lambda paths: None
    analyzer._save_for_debug = lambda paths: None
    analyzer._get_video_duration = lambda path: written['duration']

    def capture_metadata(event_id, video_path, detections, classifications, start, end, duration, output_dir):
        written['metadata'] = {'held': output_dir == settings['output_held_dir'], 'detections': detections, 'classifications': classifications}
    analyzer._create_json_metadata = capture_metadata

    for path in videos:
        name = os.path.basename(path)[:-len('.mp4')]
        result_path = os.path.join(output, name + '.json')
        if os.path.exists(result_path):
            continue
        capture = cv2.VideoCapture(path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        capture.release()
        written.clear()
        written['duration'] = frames / fps
        recorder = RecordingDetector(detector, settings['confidence_threshold'])
        analyzer.detector = recorder
        analyzer.queue = VideoQueue(analyzer, path, recorder)
        analyzer.is_running = True
        started = time.time()
        analyzer._event_loop()
        metadata = written.get('metadata')
        result = {
            'video': name, 'frames': recorder.frame_index + 1, 'fps': fps, 'size': size,
            'seconds': round(time.time() - started, 1),
            'outcome': 'discarded' if metadata is None else ('held' if metadata['held'] else 'uploaded'),
            'classifications': metadata['classifications'] if metadata else [],
            'detections': metadata['detections'] if metadata else [],
            'raw_detections': recorder.raw,
        }
        with open(result_path + '.tmp', 'w') as f:
            json.dump(result, f)
        os.replace(result_path + '.tmp', result_path)
        print(f"{name}: {result['outcome']} {[c['prediction'] for c in result['classifications']]} ({result['frames']} frames, {result['seconds']} s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--videos', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--raw-threshold', type=float, default=0.1)
    parser.add_argument('--engine-cache-dir', default='data/engine_cache/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(args.videos, '*.mp4')))
    shards = [(videos[i::args.workers], args.output, args.raw_threshold, args.engine_cache_dir) for i in range(args.workers)]
    print(f"{len(videos)} videos, {args.workers} workers", flush=True)
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(args.workers) as pool:
        pool.map(worker, shards)


if __name__ == '__main__':
    main()
