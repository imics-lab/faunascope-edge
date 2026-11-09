import cv2
import time
import os
import logging
from multiprocessing import Queue
from typing import Dict, Optional, List
import numpy as np
import threading
import queue
from datetime import datetime
import piexif

from src import device_info


class MotionDetector:
    """
    Detects motion in a video stream and passes event frames to a queue.
    This module acts as a computationally cheap "gatekeeper" to trigger
    more intensive analysis.
    """
    def __init__(self, config: Dict, debug_mode: bool = False, video_source=0, status: Optional[device_info.StatusCounters] = None):
        """
        Initializes the Motion Detector.
        :param config: A dictionary with operational parameters.
        :param debug_mode: If True, displays visual feedback windows.
        :param video_source: The camera index (e.g., 0) or a path to a video file.
        :param status: Counters shared with the heartbeat, which learns from them when the camera last delivered a frame.
        """
        # Isolate the relevant configuration section
        self.config = config['motion_detector']
        
        # --- Extract configuration parameters for easy access ---
        self.motion_frame_width = self.config['motion_frame_width']
        self.min_area = self.config['min_motion_contour_area']
        self.cooldown = self.config['motion_cooldown_seconds']
        
        # Kernel size for Gaussian blur - must be odd
        self.blur_kernel = tuple(self.config.get('blur_kernel_size', [21,21]))
        
        # --- Live View Configuration ---
        live_view_config = config.get('live_view', {})
        self.live_view_enabled = live_view_config.get('enabled', False)
        if self.live_view_enabled:
            self.live_frame_path = live_view_config.get('ram_disk_path')
            self.lock_file_path = live_view_config.get('lock_file_path')
            self.lock_stale_timeout_seconds = float(live_view_config.get('lock_stale_timeout_seconds', 2.0))
            self.jpeg_quality = int(live_view_config.get('jpeg_quality', 60))
            print(f"Live view enabled. Will write to {self.live_frame_path} when lock file is present.")
        
        # --- Initialize components ---
        self.video_source = video_source
        self.camera = None # Initialize camera as None

        # Initialize the background subtractor.
        # `detectShadows=True` is crucial for outdoor settings to help filter out shadows.
        self.bg_subtractor = None
        
        self.is_running = False
        self.queue = None
        self.debug_mode = debug_mode
        self.status = status
        self.frame_delay = 0
        
    def _read_frame_with_timeout(self, timeout=10.0):
        """
        Reads a frame from the camera in a separate thread with a timeout.
        This prevents the main loop from blocking indefinitely if the camera hangs.
        """
        frame_queue = queue.Queue()

        def reader_thread():
            try:
                ret, frame = self.camera.read()
                frame_queue.put((ret, frame))
            except Exception as e:
                frame_queue.put((False, e))

        thread = threading.Thread(target=reader_thread)
        thread.daemon = True
        thread.start()

        try:
            ret, frame = frame_queue.get(timeout=timeout)
            if isinstance(frame, Exception):
                raise frame
            return ret, frame
        except queue.Empty:
            return False, None

    def initialize_camera(self):
        """
        Initialize camera with hardware acceleration support for Jetson.
        Uses nvjpegdec for hardware-accelerated MJPEG decoding.
        Falls back to standard methods if hardware pipeline fails.
        """
        logger = logging.getLogger('MotionDetector')
        
        # Check if the source is a camera device (integer)
        if isinstance(self.video_source, int):
            # Get camera settings from config with defaults
            cam_width = self.config.get('camera_width', 1920)
            cam_height = self.config.get('camera_height', 1080)
            cam_fps = self.config.get('camera_fps', 30)
            
            logger.info("Attempting hardware-accelerated camera initialization for device %d", self.video_source)
            
            # GStreamer pipeline with software JPEG decoding and hardware format conversion
            # jpegdec provides CPU-based JPEG decoding (stable and compatible)
            # nvvidconv provides GPU-accelerated format conversion
            pipeline = (
                f"v4l2src device=/dev/video{self.video_source} ! "
                f"image/jpeg,width={cam_width},height={cam_height},framerate={cam_fps}/1 ! "
                "jpegdec ! "  # Software JPEG decoder (stable)
                "nvvidconv ! "  # Hardware format converter (GPU accelerated!)
                "video/x-raw,format=BGRx ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=true"
            )
            self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.camera.isOpened():
                logger.warning("GStreamer pipeline failed, falling back to standard camera access")
                self.camera = cv2.VideoCapture(self.video_source)
            else:
                logger.info("Camera initialized successfully with hardware-accelerated format conversion")
        
        # If the source is a file path (string), use hardware decoder
        elif isinstance(self.video_source, str):
            logger.info("Attempting hardware-accelerated file decoding for: %s", self.video_source)
            
            # Try hardware-accelerated pipeline for video files
            # Uses decodebin to auto-detect format, then routes to nvv4l2decoder
            # This handles H.264, H.265, and various container formats automatically
            pipeline = (
                f"filesrc location={self.video_source} ! "
                "decodebin ! "  # Auto-detect format and use hardware decoder if available
                "nvvidconv ! "  # Hardware format converter
                "video/x-raw,format=BGRx ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=true"
            )
            self.camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.camera.isOpened():
                logger.warning("Hardware decoder pipeline failed, using standard file reader")
                self.camera = cv2.VideoCapture(self.video_source)
            else:
                logger.info("Video file opened successfully with hardware acceleration")
        
        else:
            logger.error("FATAL: Unsupported video source type.")
            return False

        if not self.camera.isOpened():
            logger.error("FATAL: Cannot open video source: %s", self.video_source)
            return False
        
        logger.info("Video source opened successfully.")
        return True
        
    def start(self, shared_queue: Optional[Queue] = None) -> None:
        """
        Starts the main processing loop of the motion detector.
        This method will block until stop() is called or an error occurs.
        :param shared_queue: The multiprocessing.Queue to send frames to. Can be None in debug mode.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/motion_detector.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger('MotionDetector')
        
        if self.is_running:
            logger.warning("Motion detector is already running.")
            return
        
        logger.info("Initializing video source: %s", self.video_source)
        try:
            # Use hardware-accelerated initialization
            if not self.initialize_camera():
                logger.error("FATAL: Failed to initialize video source")
                return
            
            # Set frame delay for video file playback
            if isinstance(self.video_source, str):
                fps = self.camera.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    self.frame_delay = 1 / fps
                    logger.info("Video file detected. Simulating FPS: %.2f (Delay: %.4fs)", fps, self.frame_delay)
                else:
                    self.frame_delay = 1 / 30
                    logger.warning("Video file detected, but FPS not readable. Defaulting to 30 FPS.")
            else:
                logger.info("Using camera device: %s", self.video_source)
                    
            logger.info("Video source initialized successfully.")
            
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
                
            self.queue = shared_queue
            self.is_running = True
            logger.info("Starting Motion Detector processing loop...")
            self._processing_loop()
            
        except Exception as e:
            logger.exception("Exception occurred during video source initialization: %s", e)


    def stop(self) -> None:
        """
        Signals the main processing loop to terminate gracefully.
        """
        self.is_running = False
        print("Stopping Motion Detector...")


    def _processing_loop(self) -> None:
        """
        The private method containing the main while loop for frame processing.

        Each frame of an event goes to the queue as (frame, read_at, boot_seconds, motion): the time the frame was read from the camera, as UTC seconds and as seconds since boot (which a change to the system clock does not affect), and whether this frame showed motion. None ends the event.
        """
        logger = logging.getLogger('MotionDetector')
        state = "IDLE"
        last_motion_time = 0
        consecutive_failures = 0
        max_failures = 10
        camera_read_timeout = 15.0

        while self.is_running:
            # --- Watchdog Frame Read ---
            # If camera.read() blocks, this will timeout and return (False, None),
            # preventing the entire process from freezing.
            try:
                ret, original_frame = self._read_frame_with_timeout(timeout=camera_read_timeout)
            except Exception as e:
                logger.error("An unexpected exception occurred during camera.read(): %s", e)
                ret, original_frame = False, None
            read_at, read_boot_seconds = time.time(), device_info.boot_seconds()
            if ret and self.status:
                self.status.camera_frame_at.value = read_boot_seconds

            # --- Live View Frame Writing (On-Demand) ---
            should_write_live = False
            if self.live_view_enabled and os.path.exists(self.lock_file_path):
                try:
                    mtime = os.path.getmtime(self.lock_file_path)
                    if (time.time() - mtime) <= self.lock_stale_timeout_seconds:
                        should_write_live = True
                except Exception:
                    should_write_live = False

            if should_write_live and ret:
                # Get the current time for the timestamp
                now = datetime.now()
                
                # Resize frame for the live view
                height, width = original_frame.shape[:2]
                resized_frame = cv2.resize(
                    original_frame,
                    (width // 2, height // 2),
                    interpolation=cv2.INTER_AREA,
                )

                # Encode to JPEG format in memory
                ok, buffer = cv2.imencode(
                    ".jpg",
                    resized_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if ok:
                    # Create EXIF data with the capture time
                    now_str = now.strftime("%Y:%m:%d %H:%M:%S")
                    exif_dict = {
                        "0th": {
                            piexif.ImageIFD.DateTime: now_str,
                        },
                        "Exif": {
                            piexif.ExifIFD.DateTimeOriginal: now_str,
                        },
                        "1st": {},
                        "thumbnail": None,
                    }
                    exif_bytes = piexif.dump(exif_dict)

                    # Write JPEG bytes to a temporary file, then insert EXIF and publish atomically
                    raw_tmp_path = f"{self.live_frame_path}.rawtmp"
                    try:
                        with open(raw_tmp_path, "wb") as raw_file:
                            raw_file.write(buffer.tobytes())
                            raw_file.flush()
                            os.fsync(raw_file.fileno())
                        
                        # Insert EXIF into the temporary JPEG file
                        piexif.insert(exif_bytes, raw_tmp_path)
                        
                        # Atomically publish to the live frame path
                        os.replace(raw_tmp_path, self.live_frame_path)
                    except Exception:
                        # Fallback: try publishing the non-EXIF JPEG if insertion fails
                        try:
                            os.replace(raw_tmp_path, self.live_frame_path)
                        except Exception:
                            pass
            
            if not ret:
                # This block handles both read failures and timeouts.
                if isinstance(self.video_source, str):
                    logger.info("End of video file or read error. Finalizing event.")
                    if self.queue and state == "DETECTING":
                        self.queue.put(None)
                    break
                
                consecutive_failures += 1
                logger.error(
                    "Failed to grab frame or timed out (attempt %d/%d).",
                    consecutive_failures,
                    max_failures
                )

                if consecutive_failures >= max_failures:
                    logger.critical("Max failures reached. Attempting to re-initialize camera...")
                    if self.camera:
                        self.camera.release()
                    
                    # Enter a loop to periodically attempt camera re-initialization.
                    reinitialized = False
                    while not reinitialized and self.is_running:
                        time.sleep(5.0) # Wait before retrying.
                        logger.info("Attempting to re-initialize camera...")
                        if self.initialize_camera():
                            reinitialized = True
                            consecutive_failures = 0
                            logger.info("Camera re-initialized successfully!")
                        else:
                            logger.error("Failed to re-initialize camera. Will retry...")
                    
                    if not self.is_running: # Exit if stop() was called during retry.
                        break
                else:
                    time.sleep(1.0) # Short delay before next read attempt.
                
                continue
            
            consecutive_failures = 0

            # 2. Pre-process the frame for motion analysis
            processed_frame = self._preprocess_frame(original_frame)

            # 3. Apply background subtraction to get a motion mask
            fg_mask = self.bg_subtractor.apply(processed_frame)

            # 4. Clean the mask to remove noise
            cleaned_mask = self._clean_mask(fg_mask)

            # 5. Find contours and check for significant motion
            significant_contours = self._find_significant_contours(cleaned_mask)
            motion_found_this_frame = len(significant_contours) > 0

            # 6. Implement the state machine logic
            if state == "IDLE":
                if motion_found_this_frame:
                    logger.info("Motion detected! Changing to DETECTING state.")
                    state = "DETECTING"
                    last_motion_time = time.time()
                    if self.queue: self.queue.put((original_frame, read_at, read_boot_seconds, True))
            
            elif state == "DETECTING":
                if self.queue: self.queue.put((original_frame, read_at, read_boot_seconds, motion_found_this_frame))

                if motion_found_this_frame:
                    last_motion_time = time.time()
                else:
                    if time.time() - last_motion_time > self.cooldown:
                        logger.info("Cooldown of %.1fs expired. Event finished. Changing to IDLE state.", self.cooldown)
                        if self.queue: self.queue.put(None)
                        state = "IDLE"

            if self.frame_delay > 0:
                time.sleep(self.frame_delay)
            
            # If in debug mode, display the visual feedback window
            if self.debug_mode:
                self._show_debug_window(original_frame, cleaned_mask, significant_contours, state, last_motion_time)
                # Allow 'q' to quit the loop when in debug mode
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    self.is_running = False
        
        # --- Cleanup ---
        logger.info("Motion detector loop terminated. Releasing resources.")
        if self.camera is not None:
            self.camera.release()
            logger.info("Camera released successfully.")
        if self.debug_mode:
            cv2.destroyAllWindows()


    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resizes, converts to grayscale, and blurs a frame."""
        # Resize to a consistent width to speed up processing
        h, w, _ = frame.shape
        ratio = self.motion_frame_width / float(w)
        dim = (self.motion_frame_width, int(h * ratio))
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to smooth the image and reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        return blurred


    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Applies thresholding and morphological operations to a mask."""
        # Threshold the mask to get a binary image (black and white)
        # A lower threshold (e.g., 25) is often better for MOG2's shadow detection
        _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
        
        # Create a kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Erode to remove small white noise specks
        eroded = cv2.erode(thresh, kernel, iterations=2)
        
        # Dilate to close gaps in remaining objects
        dilated = cv2.dilate(eroded, kernel, iterations=2)
        
        return dilated


    def _find_significant_contours(self, mask: np.ndarray) -> List:
        """Finds contours and returns a list of those large enough to be significant."""
        # Find the outlines (contours) of all white objects in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_contours = []
        for contour in contours:
            # If the area of a contour is less than our minimum, ignore it
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            significant_contours.append(contour)
            
        return significant_contours


    def _show_debug_window(self, frame: np.ndarray, mask: np.ndarray, contours: List, state: str, last_motion_time: float) -> None:
        """Displays the visual feedback window for debugging."""
        # Draw bounding boxes on the original frame
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Scale coordinates back to the original frame size
            proc_h, proc_w = mask.shape
            orig_h, orig_w, _ = frame.shape
            scale_x, scale_y = orig_w / proc_w, orig_h / proc_h
            
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            orig_w = int(w * scale_x)
            orig_h = int(h * scale_y)
            
            cv2.rectangle(frame, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), (0, 255, 0), 2)

        # Prepare text for display
        state_text = f"State: {state}"
        color = (0, 255, 0) if state == "DETECTING" else (0, 0, 255)
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if state == "DETECTING":
            cooldown_remaining = max(0, self.cooldown - (time.time() - last_motion_time))
            cooldown_text = f"Cooldown: {cooldown_remaining:.1f}s"
            cv2.putText(frame, cooldown_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Prepare mask for display (convert to 3-channel BGR)
        display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_mask, "Cleaned Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Resize mask to match original frame height for stacking
        h1, _, _ = frame.shape
        h2, w2, _ = display_mask.shape
        display_mask_resized = cv2.resize(display_mask, (int(w2 * h1/h2), h1))
        
        # Combine frames for a side-by-side view
        combined_view = np.hstack((frame, display_mask_resized))
        
        # Resize the combined view to fit better on screen
        # Scale down to 50% of original size - adjust this value if needed
        scale_factor = 0.5
        height, width = combined_view.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_view = cv2.resize(combined_view, (new_width, new_height))
        
        cv2.imshow("Motion Detector Test", resized_view)