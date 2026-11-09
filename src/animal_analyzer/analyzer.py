import cv2
import time
import json
import os
import shutil
import hashlib
import subprocess
import uuid
import logging
from multiprocessing import Queue
from typing import Dict, List
from collections import Counter
from datetime import datetime, timezone

from .megadetector import MegaDetector
from .species_classifier import SpeciesClassifier
from .tracker import BoxTracker


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class AnimalAnalyzer:
    """
    Analyzes video frame events from a queue, detects animals, people, and vehicles with MegaDetector,
    and packages significant events as video and JSON metadata files.

    MegaDetector runs on every detection_interval_frames-th frame of an event, starting with the first, and a tracker follows its boxes through the frames in between. An event is significant once a track of one of the trigger_classes has been detected in confirmation_detections detection frames; only confirmed tracks are written to the metadata.

    When a significant event ends, SpeciesNet classifies each confirmed track from its classification_crops most confident detections. If it calls every confirmed track of the trigger_classes blank, the event goes to output_held_dir, which is not uploaded, instead of output_pending_dir.
    """

    def __init__(self, shared_queue: Queue, config: Dict):
        """
        Initializes the Animal Analyzer.
        :param shared_queue: The multiprocessing.Queue to receive frames from.
        :param config: A dictionary with operational parameters.
        """
        self.queue = shared_queue
        # We only need the 'animal_analyzer' part of the config
        self.config = config['animal_analyzer']
        self.device_id = self.config.get('device_id', 'unknown_device')
        self.location  = self.config.get('location', {'latitude':0.0,'longitude':0.0})
        self.output_fps = self.config.get('output_fps', 30.0) 
        self.is_running = False

        # The models are loaded in start(), in the analyzer's own process (required for 'spawn' multiprocessing)
        self.detector = None
        self.detector_info = {}
        self.classifier = None
        self.classifier_info = {}

        # Track which video codec works on this system (will be detected on first write)
        self.working_codec = None
        self.codec_candidates = [
            ('mp4v', '.mp4'),  # MPEG-4 Part 2 (requires libxvidcore)
            ('XVID', '.avi'),  # Xvid MPEG-4
            ('MJPG', '.avi'),  # Motion JPEG (widely supported)
        ]

        # --- Ensure Directories Exist ---
        os.makedirs(self.config['output_pending_dir'], exist_ok=True)
        os.makedirs(self.config['output_held_dir'], exist_ok=True)
        os.makedirs(self.config['output_temp_dir'], exist_ok=True)
        os.makedirs(self.config['debug_temp_dir'], exist_ok=True)
        print(f"Output directories ensured: {self.config['output_pending_dir']}, {self.config['output_temp_dir']}, and {self.config['debug_temp_dir']}")
        
        # Logger will be initialized in start() method
        self.logger = None


    def start(self) -> None:
        """Starts the main event processing loop."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/animal_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AnimalAnalyzer')
        
        model_path = self.config['detector_model_path']
        self.logger.info("Loading MegaDetector from %s...", model_path)
        self.detector = MegaDetector(model_path, self.config['confidence_threshold'], self.config['engine_cache_dir'])
        model_sha256 = file_sha256(model_path)
        self.detector_info = {
            "model": os.path.basename(model_path),
            "model_sha256": model_sha256,
            "confidence_threshold": self.config['confidence_threshold'],
            "detection_interval_frames": self.config['detection_interval_frames'],
            "confirmation_detections": self.config['confirmation_detections'],
        }
        self.logger.info("MegaDetector loaded (sha256 %s).", model_sha256[:12])

        model_dir = self.config['classifier_model_dir']
        self.logger.info("Loading SpeciesNet from %s...", model_dir)
        self.classifier = SpeciesClassifier(model_dir, self.config['engine_cache_dir'],
                                            self.config['geofence_country'], self.config['geofence_admin1_region'])
        self.classifier_info = {
            "model": f"SpeciesNet {self.classifier.version}",
            "model_sha256": file_sha256(os.path.join(model_dir, 'speciesnet.onnx')),
            "crops_per_track": self.config['classification_crops'],
            "geofence": f"{self.config['geofence_country']}-{self.config['geofence_admin1_region']}",
        }
        self.logger.info("SpeciesNet %s loaded.", self.classifier.version)
        
        if self.is_running:
            self.logger.warning("Animal analyzer is already running.")
            return
        self.is_running = True
        self.logger.info("Starting Animal Analyzer...")
        self._event_loop()


    def stop(self) -> None:
        """Signals the main processing loop to terminate."""
        self.is_running = False
        print("Stopping Animal Analyzer...")


    def _event_loop(self) -> None:
        """The main loop that waits for and processes whole events."""
        while self.is_running:
            self.logger.info("Waiting for a new event...")
            first_frame = self.queue.get()

            if first_frame is None:
                continue
            
            # --- A NEW EVENT HAS STARTED ---
            event_id = f"event_{time.strftime('%Y%m%d-%H%M%S')}_{str(uuid.uuid4())[:8]}"
            event_start_time = time.time()
            self.logger.info("New event received: %s", event_id)

            ram_buffer = []
            temp_file_parts = []
            tracker = BoxTracker(self.config['track_match_iou'], self.config['track_max_missed_detections'])
            confirmed_tracks = set()
            boxes_by_track = {}
            crops_by_track = {}
            significant_event_detected = False
            frame_index = 0
            frame = first_frame

            # --- Process all frames for this event ---
            while frame is not None:
                ram_buffer.append(frame)

                # 1. DETECT ON EVERY N-TH FRAME, FOLLOW THE BOXES IN BETWEEN
                is_detection_frame = frame_index % self.config['detection_interval_frames'] == 0
                detections = self.detector.detect(frame) if is_detection_frame else None
                visible = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detections)

                # 2. CHECK FOR EVENT SIGNIFICANCE & GATHER BOXES
                for track in visible:
                    x1, y1, x2, y2 = track.box
                    boxes_by_track.setdefault(track.track_id, []).append({
                        "frame_index": frame_index,
                        "track_id": track.track_id,
                        "label": track.label,
                        "confidence": round(track.confidence, 4),
                        "detected": track.detected,
                        "box_xywh": [int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2 - x1), int(y2 - y1)]
                    })
                    if track.detected:
                        self._keep_crop(crops_by_track.setdefault(track.track_id, []), track.confidence, frame, track.box)
                    if track.hits >= self.config['confirmation_detections'] and track.track_id not in confirmed_tracks:
                        confirmed_tracks.add(track.track_id)
                        self.logger.info("  > Confirmed %s (track %s)", track.label, track.track_id)
                        if track.label in self.config['trigger_classes']:
                            significant_event_detected = True

                # 3. HANDLE MEMORY MANAGEMENT (SPILL TO DISK)
                if len(ram_buffer) >= self.config['ram_frame_limit']:
                    self.logger.info("RAM limit reached. Spilling %d frames to disk...", len(ram_buffer))
                    temp_path = self._write_video_part(event_id, ram_buffer, len(temp_file_parts))
                    if temp_path:
                        temp_file_parts.append(temp_path)
                    ram_buffer.clear()

                frame_index += 1
                frame = self.queue.get()

            all_confirmed_detections = sorted(
                (box for track_id in confirmed_tracks for box in boxes_by_track[track_id]),
                key=lambda box: (box["frame_index"], box["track_id"]))

            # --- EVENT FINALIZATION ---
            event_end_time = time.time()
            self.logger.info("Event %s finished. Finalizing...", event_id)

            if significant_event_detected and ram_buffer:
                temp_path = self._write_video_part(event_id, ram_buffer, len(temp_file_parts))
                if temp_path:
                    temp_file_parts.append(temp_path)
            ram_buffer.clear()

            if significant_event_detected and temp_file_parts:
                classifications = self._classify_tracks(confirmed_tracks, boxes_by_track, crops_by_track)
                triggers = [c for c in classifications if c['detector_label'] in self.config['trigger_classes']]
                if triggers and all(c['blank'] for c in triggers):
                    output_dir = self.config['output_held_dir']
                    self.logger.info("SpeciesNet calls every track blank. Holding %s in %s.", event_id, output_dir)
                else:
                    output_dir = self.config['output_pending_dir']
                    self.logger.info("Significant event! Packaging video and metadata for %s.", event_id)
                final_video_path = self._concatenate_parts(event_id, temp_file_parts, output_dir)
                if final_video_path:
                    actual_duration = self._get_video_duration(final_video_path)
                    self._create_json_metadata(event_id, final_video_path, all_confirmed_detections, classifications, event_start_time, event_end_time, actual_duration, output_dir)
            else:
                if self.config.get('debug_mode', False):
                    self.logger.info("Insignificant event or no frames. Moving temporary files to debug directory for %s.", event_id)
                    self._save_for_debug(temp_file_parts)
                else:
                    self.logger.info("Insignificant event or no frames. Cleaning up temporary files for %s.", event_id)
                    self._cleanup_temp_files(temp_file_parts)
    
    def _keep_crop(self, crops: List, confidence: float, frame, box) -> None:
        """Keep a track's classification_crops most confident detections as classifier crops."""
        x1, y1, x2, y2 = box
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        if len(crops) < self.config['classification_crops'] or confidence > crops[-1][0]:
            crops.append((confidence, SpeciesClassifier.crop(frame, box)))
            crops.sort(key=lambda c: -c[0])
            del crops[self.config['classification_crops']:]

    def _classify_tracks(self, track_ids, boxes_by_track: Dict, crops_by_track: Dict) -> List[Dict]:
        """SpeciesNet's result for each confirmed track, in track order."""
        classifications = []
        for track_id in sorted(track_ids):
            crops = crops_by_track.get(track_id)
            if not crops:
                continue
            boxes = boxes_by_track[track_id]
            label = boxes[0]['label']
            result = self.classifier.classify([crop for _, crop in crops], label, max(confidence for confidence, _ in crops))
            classifications.append({"track_id": track_id, "detector_label": label, "crops": len(crops), **result})
            self.logger.info("  > Track %s (%s): %s (%.2f, %s)", track_id, label, result['prediction'], result['score'], result['source'])
        return classifications

    def _save_for_debug(self, file_paths: List[str]) -> None:
        """Moves temporary files to a persistent debug directory instead of deleting them."""
        for path in file_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                dest = os.path.join(self.config['debug_temp_dir'], os.path.basename(path))
                shutil.move(path, dest)
                self.logger.info("  > Debug file saved: %s", dest)
            except Exception as e:
                self.logger.warning("Could not move debug file %s: %s", path, e)

    def _get_video_duration(self, video_path: str) -> float:
        """Calculates the precise duration of a video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0
            
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps > 0:
                return frame_count / fps
            return 0.0
        except Exception as e:
            self.logger.error("Could not calculate video duration for %s: %s", video_path, e)
            return 0.0

    def _write_video_part(self, event_id: str, frames: List, part_num: int) -> str:
        """
        Writes a video part using the best available codec.
        Tries multiple codecs and remembers which one works for future writes.
        """
        if not frames:
            return ""
        height, width, _ = frames[0].shape
        
        # If we already found a working codec, use it
        if self.working_codec:
            codecs_to_try = [self.working_codec]
        else:
            codecs_to_try = self.codec_candidates
        
        for fourcc_name, extension in codecs_to_try:
            temp_path = os.path.join(self.config['output_temp_dir'], f"{event_id}_part_{part_num}{extension}")
            
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                writer = cv2.VideoWriter(temp_path, fourcc, self.output_fps, (width, height))
                
                if not writer.isOpened():
                    self.logger.warning("Codec '%s' failed to initialize VideoWriter", fourcc_name)
                    continue
                
                for frame in frames:
                    writer.write(frame)
                writer.release()
                
                # Verify the file was actually created and has content
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    # Success! Remember this codec for future writes
                    if not self.working_codec:
                        self.working_codec = (fourcc_name, extension)
                        self.logger.info("Detected working video codec: %s (%s)", fourcc_name, extension)
                    
                    self.logger.info("  > Wrote %s", temp_path)
                    return temp_path
                else:
                    self.logger.warning("Codec '%s' created no output or empty file", fourcc_name)
                    # Clean up empty file if it exists
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
                    
            except Exception as e:
                self.logger.warning("VideoWriter failed with codec '%s': %s", fourcc_name, e)
                continue
        
        # If we get here, all codecs failed
        self.logger.error(
            "Failed to write video part %s. Tried codecs: %s",
            event_id,
            ", ".join(name for name, _ in self.codec_candidates)
        )
        return ""

    def _concatenate_parts(self, event_id: str, temp_parts: List[str], output_dir: str) -> str:
        if not temp_parts:
            return ""
        
        # Filter out empty strings and non-existent files
        valid_parts = [p for p in temp_parts if p and os.path.exists(p)]
        if not valid_parts:
            self.logger.error("No valid temporary video parts found for event %s", event_id)
            return ""
        
        final_video_path = os.path.join(output_dir, f"{event_id}.mp4")
        file_list_path = os.path.join(self.config['output_temp_dir'], f"{event_id}_filelist.txt")
        
        with open(file_list_path, 'w') as f:
            for path in valid_parts:
                f.write(f"file '{os.path.abspath(path)}'\n")
        
        command = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', file_list_path,
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '24',
            '-profile:v', 'high', '-pix_fmt', 'yuv420p',
            '-an',
            '-movflags', '+faststart',
            final_video_path
        ]
        
        try:
            self.logger.info("Running FFmpeg to create %s...", final_video_path)
            subprocess.run(command, check=True, capture_output=True, text=True)
            self._cleanup_temp_files(valid_parts)
            try:
                os.remove(file_list_path)
            except OSError:
                pass
            return final_video_path
        except subprocess.CalledProcessError as e:
            self.logger.error("Error during FFmpeg concatenation: %s", e.stderr)
            self._cleanup_temp_files(valid_parts)
            # Clean up potentially broken output file
            if os.path.exists(final_video_path):
                try:
                    os.remove(final_video_path)
                except OSError:
                    pass
            try:
                os.remove(file_list_path)
            except OSError:
                pass
            return ""

    def _create_json_metadata(self, event_id: str, video_path: str, detections: List, classifications: List, start_time: float, end_time: float, actual_duration: float, output_dir: str) -> None:
        # Species of the tracks SpeciesNet does not call blank, the track with the most boxes first.
        boxes_per_track = Counter(d['track_id'] for d in detections)
        named = sorted((c for c in classifications if not c['blank']), key=lambda c: -boxes_per_track[c['track_id']])
        unique_species = list(dict.fromkeys(c['prediction'] for c in named))
        primary = unique_species[0] if unique_species else "blank"
        metadata = {
            "eventId": event_id,
            "deviceId": self.device_id,
            "location": {
                "latitude": self.location['latitude'],
                "longitude": self.location['longitude'],
                "timezone": self.location.get('timezone', 'UTC')
            },
            "timestamp_start_utc": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
            "timestamp_end_utc": datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat(),
            "local_video_path": video_path,
            "video_duration_seconds": round(actual_duration, 2),
            "event_summary": {
                "species_list": unique_species,
                "primary_species": primary,
                "max_confidence": max((d['confidence'] for d in detections), default=0.0)
            },
            "detector": self.detector_info,
            "classifier": self.classifier_info,
            "classifications": classifications,
            "detections": detections
        }
        json_path = os.path.join(output_dir, f"{event_id}.json")
        # Written under another name and renamed, so the uploader never reads a half-written file.
        with open(json_path + '.tmp', 'w') as f: json.dump(metadata, f, separators=(',', ':'))
        os.replace(json_path + '.tmp', json_path)
        self.logger.info("  > Wrote JSON metadata to %s", json_path)

    def _cleanup_temp_files(self, file_paths: List[str]) -> None:
        for path in file_paths:
            try:
                os.remove(path)
            except OSError as e:
                self.logger.warning("Error deleting temp file %s: %s", path, e)