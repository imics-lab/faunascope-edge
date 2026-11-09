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
from typing import Dict, List, Optional
from collections import Counter
from datetime import datetime, timezone

from src import device_info
from .megadetector import IOU_THRESHOLD, MegaDetector
from .species_classifier import SpeciesClassifier
from .speciesnet.constants import Classification
from .tracker import BoxTracker

FORMAT_VERSION = 2
# SpeciesNet's labels for an animal it could not place; any named taxon is preferred as the primary species.
UNPLACED = (Classification.ANIMAL, Classification.UNKNOWN)
# How ONNX Runtime ran a model, by the first of the session's providers.
RUNTIMES = {'TensorrtExecutionProvider': 'TensorRT FP16', 'CUDAExecutionProvider': 'CUDA FP32', 'CPUExecutionProvider': 'CPU FP32'}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime(session) -> str:
    provider = session.get_providers()[0]
    return RUNTIMES.get(provider, provider)


def iso_utc(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def measured_frame_rate(frame_offsets_ms: List[int], fallback: float) -> float:
    """The average rate at which the frames were captured, or fallback if it cannot be measured."""
    if len(frame_offsets_ms) < 2 or frame_offsets_ms[-1] <= 0:
        return fallback
    return round((len(frame_offsets_ms) - 1) / (frame_offsets_ms[-1] / 1000), 3)


def summarize(tracks: List[Dict]) -> Dict:
    """
    One entry per species SpeciesNet named (not blank): its tracks, and max_count, the most of its tracks visible in one frame. The species with the most boxes comes first and is the primary species, but "animal" and "no cv result" come after every named taxon.
    """
    entries, boxes, per_frame = {}, Counter(), Counter()
    for track in tracks:
        species = track["species"]
        if not species or species["blank"]:
            continue
        taxon = species["taxon"]
        entry = entries.setdefault(taxon, {"taxon": taxon, "name": species["name"], "tracks": 0, "max_count": 0})
        entry["tracks"] += 1
        boxes[taxon] += len(track["boxes"]["frame"])
        per_frame.update((taxon, frame) for frame in track["boxes"]["frame"])
    for (taxon, _), count in per_frame.items():
        entries[taxon]["max_count"] = max(entries[taxon]["max_count"], count)
    ordered = sorted(entries.values(), key=lambda entry: (entry["taxon"] in UNPLACED, -boxes[entry["taxon"]]))
    return {"species": ordered, "primary_species": ordered[0]["name"] if ordered else "blank"}


class AnimalAnalyzer:
    """
    Analyzes video frame events from a queue, detects animals, people, and vehicles with MegaDetector,
    and packages significant events as video and JSON metadata files.

    MegaDetector runs on every detection_interval_frames-th frame of an event, starting with the first, and a tracker follows its boxes through the frames in between. An event is significant once a track of one of the trigger_classes has been detected in confirmation_detections detection frames; only confirmed tracks are written to the metadata.

    When a significant event ends, SpeciesNet classifies each confirmed track from its classification_crops most confident detections. If it calls every confirmed track of the trigger_classes blank, the event goes to output_held_dir, which is not uploaded, instead of output_pending_dir.

    Frames arrive with the times the motion detector read them from the camera. The video is encoded at the average rate of those times, so it plays at real speed, and the metadata (format version 2, docs/metadata.md) records each frame's time.
    """

    def __init__(self, shared_queue: Queue, config: Dict, status: Optional[device_info.StatusCounters] = None):
        """
        Initializes the Animal Analyzer.
        :param shared_queue: The multiprocessing.Queue to receive frames from.
        :param config: A dictionary with operational parameters.
        :param status: Counters shared with the heartbeat, which counts events by outcome.
        """
        self.queue = shared_queue
        self.status = status or device_info.StatusCounters()
        # We only need the 'animal_analyzer' part of the config
        self.config = config['animal_analyzer']
        self.device_id = self.config.get('device_id', 'unknown_device')
        self.location  = self.config.get('location', {'latitude':0.0,'longitude':0.0})
        self.output_fps = self.config.get('output_fps', 30.0) 
        self.is_running = False

        # The models are loaded in start(), in the analyzer's own process (required for 'spawn' multiprocessing)
        self.detector = None
        self.classifier = None
        self.models = {}
        self.software = device_info.software()
        motion = config['motion_detector']
        self.settings = {
            "motion": {"frame_width": motion['motion_frame_width'], "blur_kernel": list(motion.get('blur_kernel_size', [21, 21])),
                       "min_contour_area": motion['min_motion_contour_area'], "cooldown_seconds": motion['motion_cooldown_seconds']},
            "detector": {"confidence_threshold": self.config['confidence_threshold'], "iou_threshold": IOU_THRESHOLD,
                         "interval_frames": self.config['detection_interval_frames']},
            "tracker": {"match_iou": self.config['track_match_iou'], "max_missed_detections": self.config['track_max_missed_detections'],
                        "confirmation_detections": self.config['confirmation_detections']},
            "classifier": {"crops_per_track": self.config['classification_crops'],
                           "geofence": f"{self.config['geofence_country']}-{self.config['geofence_admin1_region']}"},
            "trigger_classes": list(self.config['trigger_classes']),
        }

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
        self.models["detector"] = {"name": os.path.splitext(os.path.basename(model_path))[0], "sha256": file_sha256(model_path),
                                   "runtime": runtime(self.detector.session)}
        self.logger.info("MegaDetector loaded (sha256 %s, %s).", self.models["detector"]["sha256"][:12], self.models["detector"]["runtime"])

        model_dir = self.config['classifier_model_dir']
        self.logger.info("Loading SpeciesNet from %s...", model_dir)
        self.classifier = SpeciesClassifier(model_dir, self.config['engine_cache_dir'],
                                            self.config['geofence_country'], self.config['geofence_admin1_region'])
        self.models["classifier"] = {"name": "SpeciesNet", "version": self.classifier.version,
                                     "sha256": file_sha256(os.path.join(model_dir, 'speciesnet.onnx')),
                                     "runtime": runtime(self.classifier.session)}
        self.logger.info("SpeciesNet %s loaded (%s).", self.classifier.version, self.models["classifier"]["runtime"])
        self.status.analyzer_ready.value = 1
        
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
            item = self.queue.get()

            if item is None:
                continue
            
            # --- A NEW EVENT HAS STARTED ---
            frame, start_time, start_boot_seconds, _ = item
            event_id = f"event_{time.strftime('%Y%m%d-%H%M%S', time.localtime(start_time))}_{str(uuid.uuid4())[:8]}"
            event = {
                "event_id": event_id,
                "start_time": start_time,
                "clock": device_info.clock(start=start_boot_seconds),
                "frame_size": frame.shape[1::-1],
                "frame_offsets_ms": [],
                "last_motion_frame": 0,
                "tracks": {},
            }
            self.status.add('motion')
            self.logger.info("New event received: %s", event_id)

            ram_buffer = []
            temp_file_parts = []
            tracker = BoxTracker(self.config['track_match_iou'], self.config['track_max_missed_detections'])
            crops_by_track = {}
            significant_event_detected = False
            frame_index = 0

            # --- Process all frames for this event ---
            while item is not None:
                frame, _, boot_seconds, motion = item
                ram_buffer.append(frame)
                event["frame_offsets_ms"].append(round((boot_seconds - start_boot_seconds) * 1000))
                if motion:
                    event["last_motion_frame"] = frame_index

                # 1. DETECT ON EVERY N-TH FRAME, FOLLOW THE BOXES IN BETWEEN
                is_detection_frame = frame_index % self.config['detection_interval_frames'] == 0
                detections = self.detector.detect(frame) if is_detection_frame else None
                visible = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detections)

                # 2. CHECK FOR EVENT SIGNIFICANCE & GATHER BOXES
                for track in visible:
                    record = event["tracks"].setdefault(track.track_id, {
                        "class": track.label, "confirmed_frame": None,
                        "boxes": {"frame": [], "xyxy": [], "detected": [], "confidence": []}})
                    boxes = record["boxes"]
                    boxes["frame"].append(frame_index)
                    boxes["xyxy"].append([int(round(v)) for v in track.box])
                    boxes["detected"].append(track.detected)
                    boxes["confidence"].append(round(track.confidence, 4) if track.detected else None)
                    if track.detected:
                        self._keep_crop(crops_by_track.setdefault(track.track_id, []), track.confidence, frame_index, frame, track.box)
                    if track.hits >= self.config['confirmation_detections'] and record["confirmed_frame"] is None:
                        record["confirmed_frame"] = frame_index
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
                item = self.queue.get()

            # Only confirmed tracks are recorded.
            event["tracks"] = {track_id: record for track_id, record in event["tracks"].items() if record["confirmed_frame"] is not None}

            # --- EVENT FINALIZATION ---
            self.logger.info("Event %s finished. Finalizing...", event_id)

            if significant_event_detected and ram_buffer:
                temp_path = self._write_video_part(event_id, ram_buffer, len(temp_file_parts))
                if temp_path:
                    temp_file_parts.append(temp_path)
            ram_buffer.clear()

            if significant_event_detected and temp_file_parts:
                species = self._classify_tracks(event["tracks"], crops_by_track)
                triggers = [track_id for track_id, record in event["tracks"].items() if record["class"] in self.config['trigger_classes']]
                if triggers and all(track_id in species and species[track_id]['blank'] for track_id in triggers):
                    decision, output_dir = 'held_blank', self.config['output_held_dir']
                    self.logger.info("SpeciesNet calls every track blank. Holding %s in %s.", event_id, output_dir)
                else:
                    decision, output_dir = 'upload', self.config['output_pending_dir']
                    self.logger.info("Significant event! Packaging video and metadata for %s.", event_id)
                frame_rate = measured_frame_rate(event["frame_offsets_ms"], self.output_fps)
                if self._concatenate_parts(event_id, temp_file_parts, frame_rate, output_dir):
                    self._write_metadata(self._event_metadata(event, species, frame_rate, decision), output_dir)
                    self.status.add('held_blank' if decision == 'held_blank' else 'kept')
            else:
                self.status.add('discarded')
                if self.config.get('debug_mode', False):
                    self.logger.info("Insignificant event or no frames. Moving temporary files to debug directory for %s.", event_id)
                    self._save_for_debug(temp_file_parts)
                else:
                    self.logger.info("Insignificant event or no frames. Cleaning up temporary files for %s.", event_id)
                    self._cleanup_temp_files(temp_file_parts)
    
    def _keep_crop(self, crops: List, confidence: float, frame_index: int, frame, box) -> None:
        """Keep a track's classification_crops most confident detections, with their frame indices, as classifier crops."""
        x1, y1, x2, y2 = box
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        if len(crops) < self.config['classification_crops'] or confidence > crops[-1][0]:
            crops.append((confidence, frame_index, SpeciesClassifier.crop(frame, box)))
            crops.sort(key=lambda c: -c[0])
            del crops[self.config['classification_crops']:]

    def _classify_tracks(self, tracks: Dict, crops_by_track: Dict) -> Dict[int, Dict]:
        """SpeciesNet's result for each track that has crops, by track ID, with the frames of the crops."""
        species = {}
        for track_id in sorted(tracks):
            crops = crops_by_track.get(track_id)
            if not crops:
                continue
            label = tracks[track_id]["class"]
            result = self.classifier.classify([crop for _, _, crop in crops], label, max(confidence for confidence, _, _ in crops))
            species[track_id] = {**result, "crop_frames": sorted(frame_index for _, frame_index, _ in crops)}
            self.logger.info("  > Track %s (%s): %s (%.2f, %s)", track_id, label, result['name'], result['score'], result['source'])
        return species

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

    def _concatenate_parts(self, event_id: str, temp_parts: List[str], frame_rate: float, output_dir: str) -> str:
        """Encode the parts as one H.264 video at frame_rate, the rate at which the camera captured them, so the video plays at real speed."""
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
            # The parts are written at a nominal rate; their frames are renumbered at the measured one.
            '-vf', f'settb=AVTB,setpts=N/({frame_rate}*TB)', '-r', str(frame_rate),
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

    def _event_metadata(self, event: Dict, species: Dict[int, Dict], frame_rate: float, decision: str) -> Dict:
        """The event's metadata in format version 2 (docs/metadata.md)."""
        offsets = event["frame_offsets_ms"]
        width, height = event["frame_size"]
        tracks = []
        for track_id, record in sorted(event["tracks"].items()):
            frames = record["boxes"]["frame"]
            tracks.append({
                "track_id": track_id,
                "class": record["class"],
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "confirmed_frame": record["confirmed_frame"],
                "species": species.get(track_id),
                "boxes": record["boxes"],
            })
        return {
            "format_version": FORMAT_VERSION,
            "event_id": event["event_id"],
            "device_id": self.device_id,
            "edge_decision": decision,
            "software": self.software,
            "models": self.models,
            "settings": self.settings,
            "clock": event["clock"],
            "time": {
                "start_utc": iso_utc(event["start_time"]),
                "end_utc": iso_utc(event["start_time"] + offsets[-1] / 1000),
                "timezone": self.location.get('timezone') or 'UTC',
            },
            "location": {"latitude": self.location['latitude'], "longitude": self.location['longitude'], "source": "device configuration"},
            "video": {
                "file": f"{event['event_id']}.mp4",
                "width": width,
                "height": height,
                "frames": len(offsets),
                "frames_dropped": 0,
                "frame_rate": frame_rate,
                "duration_seconds": round(len(offsets) / frame_rate, 3),
                "frame_offsets_ms": offsets,
            },
            "motion": {"last_motion_frame": event["last_motion_frame"]},
            "summary": summarize(tracks),
            "tracks": tracks,
        }

    def _write_metadata(self, metadata: Dict, output_dir: str) -> None:
        json_path = os.path.join(output_dir, f"{metadata['event_id']}.json")
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