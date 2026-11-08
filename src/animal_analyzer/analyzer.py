import cv2
import time
import json
import os
import shutil
import subprocess
import uuid
import logging
import numpy as np
from multiprocessing import Queue
from typing import Dict, List, Tuple
from scipy.spatial import distance as dist
from ultralytics import YOLO
from collections import Counter
import torch
from datetime import datetime, timezone

class AnimalAnalyzer:
    """
    Analyzes video frame events from a queue, detects animals using YOLO,
    and packages significant events as video and JSON metadata files.
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

        # YOLO model will be loaded in start() method (required for 'spawn' multiprocessing)
        self.model = None

        # Track which video codec works on this system (will be detected on first write)
        self.working_codec = None
        self.codec_candidates = [
            ('mp4v', '.mp4'),  # MPEG-4 Part 2 (requires libxvidcore)
            ('XVID', '.avi'),  # Xvid MPEG-4
            ('MJPG', '.avi'),  # Motion JPEG (widely supported)
        ]

        # --- Ensure Directories Exist ---
        os.makedirs(self.config['output_pending_dir'], exist_ok=True)
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
        
        # On Jetson with 'spawn', the cuDNN backend can be problematic.
        # Disabling it entirely forces PyTorch to use its own, more stable
        # CUDA implementations for convolutions, which can prevent engine errors.
        # if torch.cuda.is_available():
        #     torch.backends.cudnn.enabled = False

        # Load YOLO model in child process (required for 'spawn' start method)
        self.logger.info("Loading YOLO model in child process...")
        self.model = YOLO(self.config['yolo_model_path'])
        self.logger.info("YOLO model loaded successfully.")
        
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

            ram_buffer = [first_frame]
            temp_file_parts = []
            tracked_objects = {}
            next_object_id = 0
            all_confirmed_detections = []
            significant_event_detected = False
            frame_index = 0
            
            # --- Process all frames for this event ---
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                
                ram_buffer.append(frame)
                frame_index += 1
                
                # 1. RUN YOLO INFERENCE AND PARSE RESULTS
                results = self.model(frame, verbose=False)
                current_detections = self._parse_yolo_results(results)

                # 2. UPDATE OBJECT TRACKING
                tracked_objects, next_object_id = self._update_tracker(
                    tracked_objects, current_detections, next_object_id
                )
                
                # 3. CHECK FOR EVENT SIGNIFICANCE & GATHER DETECTIONS
                for obj_id, obj_data in tracked_objects.items():
                    # Check if this object just became significant
                    if (obj_data['label'] in self.config['species_of_interest'] and
                        obj_data['frames_seen'] == self.config['event_confirmation_frames'] and
                        not obj_data.get('confirmed', False)):
                        
                        self.logger.info("  > Confirmed significant object: %s (ID: %s)", obj_data['label'], obj_id)
                        significant_event_detected = True
                        obj_data['confirmed'] = True

                    # If the event is significant, log all new detections of tracked, interesting species
                    if significant_event_detected and obj_data['label'] in self.config['species_of_interest'] and obj_data['just_seen']:
                        cx, cy = obj_data['centroid']
                        w,  h  = obj_data['last_box_xywh'][2], obj_data['last_box_xywh'][3]
                        all_confirmed_detections.append({
                            "frame_index": frame_index,
                            "label": obj_data['label'],
                            "confidence": float(obj_data['last_confidence']),
                            "box_xywh": [cx, cy, int(w), int(h)]
                        })

                # 4. HANDLE MEMORY MANAGEMENT (SPILL TO DISK)
                if len(ram_buffer) >= self.config['ram_frame_limit']:
                    self.logger.info("RAM limit reached. Spilling %d frames to disk...", len(ram_buffer))
                    temp_path = self._write_video_part(event_id, ram_buffer, len(temp_file_parts))
                    if temp_path:
                        temp_file_parts.append(temp_path)
                    ram_buffer.clear()
            
            # --- EVENT FINALIZATION ---
            event_end_time = time.time()
            self.logger.info("Event %s finished. Finalizing...", event_id)

            if significant_event_detected and ram_buffer:
                temp_path = self._write_video_part(event_id, ram_buffer, len(temp_file_parts))
                if temp_path:
                    temp_file_parts.append(temp_path)
            ram_buffer.clear()

            if significant_event_detected and temp_file_parts:
                self.logger.info("Significant event! Packaging video and metadata for %s.", event_id)
                final_video_path = self._concatenate_parts(event_id, temp_file_parts)
                if final_video_path:
                    actual_duration = self._get_video_duration(final_video_path)
                    self._create_json_metadata(event_id, final_video_path, all_confirmed_detections, event_start_time, event_end_time, actual_duration)
            else:
                if self.config.get('debug_mode', False):
                    self.logger.info("Insignificant event or no frames. Moving temporary files to debug directory for %s.", event_id)
                    self._save_for_debug(temp_file_parts)
                else:
                    self.logger.info("Insignificant event or no frames. Cleaning up temporary files for %s.", event_id)
                    self._cleanup_temp_files(temp_file_parts)
    
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

    def _parse_yolo_results(self, results) -> List[Dict]:
        """Parses the output from the YOLO model into a standardized list of dicts."""
        detections = []
        if results and results[0]:
            boxes = results[0].boxes
            for box in boxes:
                if box.conf[0] >= self.config['confidence_threshold']:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    cx, cy = int(x1 + w/2), int(y1 + h/2)
                    detections.append({
                        'label': self.model.names[int(box.cls[0])],
                        'confidence': box.conf[0].cpu().numpy(),
                        'box_xywh':  (cx, cy, int(w), int(h)),
                        'centroid': (cx, cy)
                    })
        return detections

    def _update_tracker(self, tracked_objects: Dict, current_detections: List, next_object_id: int) -> Tuple[Dict, int]:
        """Updates the state of tracked objects using a simple centroid-based algorithm."""
        
        # If there are no existing objects, register all new detections
        if len(tracked_objects) == 0:
            for det in current_detections:
                tracked_objects[next_object_id] = {
                    'id': next_object_id, 'label': det['label'], 'centroid': det['centroid'],
                    'frames_seen': 1, 'frames_since_seen': 0, 'just_seen': True,
                    'last_confidence': det['confidence'], 'last_box_xywh':  det['box_xywh'],
                }
                next_object_id += 1
            return tracked_objects, next_object_id

        # Prepare lists of existing and new centroids
        existing_obj_ids = list(tracked_objects.keys())
        existing_centroids = [tracked_objects[oid]['centroid'] for oid in existing_obj_ids]
        new_centroids = [det['centroid'] for det in current_detections]

        # Mark all objects as not seen in this frame initially
        for oid in tracked_objects:
            tracked_objects[oid]['just_seen'] = False

        # If no new detections, just increment disappearance counter for all existing objects
        if len(current_detections) == 0:
            for oid in tracked_objects:
                tracked_objects[oid]['frames_since_seen'] += 1
        # Otherwise, perform matching
        else:
            # Calculate the distance between all pairs of existing and new centroids
            D = dist.cdist(np.array(existing_centroids), np.array(new_centroids))

            # Find the best match for each existing object (smallest distance)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # If distance is too large, it's not the same object
                if D[row, col] > self.config.get('tracking_max_distance', 75): # Use a default if not in config
                    continue

                # It's a match! Update the tracked object.
                object_id = existing_obj_ids[row]
                tracked_objects[object_id]['centroid'] = new_centroids[col]
                tracked_objects[object_id]['frames_seen'] += 1
                tracked_objects[object_id]['frames_since_seen'] = 0
                tracked_objects[object_id]['just_seen'] = True
                tracked_objects[object_id]['last_confidence'] = current_detections[col]['confidence']
                tracked_objects[object_id]['last_box_xywh'] = current_detections[col]['box_xywh']
                
                used_rows.add(row)
                used_cols.add(col)

            # Handle objects that were not matched
            unmatched_rows = set(range(len(existing_obj_ids))) - used_rows
            unmatched_cols = set(range(len(new_centroids))) - used_cols

            for row in unmatched_rows:
                object_id = existing_obj_ids[row]
                tracked_objects[object_id]['frames_since_seen'] += 1

            for col in unmatched_cols:
                det = current_detections[col]
                tracked_objects[next_object_id] = {
                    'id': next_object_id, 'label': det['label'], 'centroid': det['centroid'],
                    'frames_seen': 1, 'frames_since_seen': 0, 'just_seen': True,
                    'last_confidence': det['confidence'], 'last_box_xywh': det['box_xywh']
                }
                next_object_id += 1

        # Prune stale tracks that have been gone for too long
        final_tracked_objects = {}
        for obj_id, obj_data in tracked_objects.items():
            if obj_data['frames_since_seen'] <= self.config['tracking_inactivity_frames']:
                final_tracked_objects[obj_id] = obj_data

        return final_tracked_objects, next_object_id

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

    def _concatenate_parts(self, event_id: str, temp_parts: List[str]) -> str:
        if not temp_parts:
            return ""
        
        # Filter out empty strings and non-existent files
        valid_parts = [p for p in temp_parts if p and os.path.exists(p)]
        if not valid_parts:
            self.logger.error("No valid temporary video parts found for event %s", event_id)
            return ""
        
        final_video_path = os.path.join(self.config['output_pending_dir'], f"{event_id}.mp4")
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

    def _create_json_metadata(self, event_id: str, video_path: str, detections: List, start_time: float, end_time: float, actual_duration: float) -> None:
        species_counts = Counter(d['label'] for d in detections)
        unique_species = list(species_counts.keys())
        if species_counts:
            primary = species_counts.most_common(1)[0][0]
        else:
            primary = "N/A"
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
            "detections": detections
        }
        json_path = os.path.join(self.config['output_pending_dir'], f"{event_id}.json")
        # Written under another name and renamed, so the uploader never reads a half-written file.
        with open(json_path + '.tmp', 'w') as f: json.dump(metadata, f, indent=4)
        os.replace(json_path + '.tmp', json_path)
        self.logger.info("  > Wrote JSON metadata to %s", json_path)

    def _cleanup_temp_files(self, file_paths: List[str]) -> None:
        for path in file_paths:
            try:
                os.remove(path)
            except OSError as e:
                self.logger.warning("Error deleting temp file %s: %s", path, e)