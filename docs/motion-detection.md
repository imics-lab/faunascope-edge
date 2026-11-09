# Motion detection

The motion detector (`src/motion_detector/detector.py`, class `MotionDetector`) is the cheap, always-on stage. It reads the camera, decides when something is moving, and passes the frames of each motion event to the animal analyzer. It does not try to tell animals from wind or shadows beyond simple filtering; that is the analyzer's job.

## Input

`VIDEO_SOURCE` in `.env` is either a camera index (`0` for `/dev/video0`) or the path of a video file.

- **Camera.** The detector first opens a GStreamer pipeline that reads MJPEG from the camera at `CAMERA_WIDTH` x `CAMERA_HEIGHT` and `CAMERA_FPS`, decodes it on the CPU, and converts the color format on the Jetson's hardware (`nvvidconv`). If that pipeline does not open (for example on a machine without Jetson plugins), it falls back to OpenCV's default camera access. Reads have a 10-second timeout; after repeated failures the camera is released and re-opened every 5 seconds until it works again.
- **Video file.** A file is decoded through GStreamer (`decodebin`, with the hardware decoder where available) or OpenCV, and played at its own frame rate, so a recording behaves like a live camera. The event ends when the file ends.

## Detecting motion

Each frame is processed in four steps:

1. A copy is resized to `motion_frame_width` pixels wide (640), converted to grayscale, and blurred with a `blur_kernel_size` Gaussian (21 x 21).
2. A MOG2 background subtractor (`history=500`, `varThreshold=50`, shadow detection on) compares it with a background model that adapts to slow changes in light.
3. The foreground mask is thresholded at 25, which also drops the pixels MOG2 marks as shadow, then eroded and dilated twice with a 3 x 3 kernel to remove specks.
4. Motion is present if any outer contour of the mask covers at least `min_motion_contour_area` pixels (500, measured on the 640-pixel copy).

## Events

The detector is either `IDLE` or `DETECTING`.

- In `IDLE`, the first frame with motion starts an event: the state becomes `DETECTING` and the full-resolution frame goes into the queue.
- In `DETECTING`, every frame goes into the queue, with or without motion, so an animal that stands still for a moment is not cut out. Each frame with motion restarts the cooldown. When `motion_cooldown_seconds` (20) pass without motion, the detector puts `None` into the queue, which ends the event, and returns to `IDLE`.

An event therefore reaches the analyzer as `[item, item, ..., item, None]`. Each item is `(frame, read_at, boot_seconds, motion)`: the frame as a BGR `numpy` array at the camera's resolution; the time it was read from the camera, as UTC seconds and as seconds since boot (`CLOCK_BOOTTIME`, which a change to the system clock does not affect); and whether this frame showed motion. The analyzer takes the event's times and frame rate from these readings (see [Event metadata](metadata.md#time)). The detector also records when the camera last delivered a frame, which the heartbeat counts as camera operating time.

The queue is unbounded. If the analyzer falls behind for long, frames accumulate in memory; bounding the queue is an open item.

## Live view

The live view lets someone look through the camera without disturbing detection, and costs nothing while nobody is watching.

- `scripts/stream_server.py` is a small Flask server on port 8080, started by the container next to `main.py`. While a viewer is connected, it keeps a lock file (`/dev/shm/viewer_active.lock`) fresh.
- While that lock file exists and is less than `lock_stale_timeout_seconds` old, the motion detector writes each frame as a JPEG (quality `jpeg_quality`, with the capture time in its EXIF data) to `/dev/shm/live_frame.jpg`. Without the lock file it writes nothing.
- The server streams those JPEGs to the browser at up to `target_fps`, lowers the rate when the network is slow, admits at most `max_viewers` viewers, and pauses each stream after `stream_timeout_seconds` (60) so a forgotten browser tab cannot use up a cellular data plan.

`scripts/view_live_local.py` shows the same frames in a window on the device itself. All settings are in the `live_view` section of `config/config.yaml`.

## Settings

| Setting (`motion_detector` section) | Default | Meaning |
|---|---|---|
| `video_source` | `${VIDEO_SOURCE}` | Camera index or video file |
| `camera_width`, `camera_height`, `camera_fps` | from `.env` | Requested camera mode; `v4l2-ctl -d /dev/video0 --list-formats-ext` lists the supported ones |
| `motion_frame_width` | 640 | Width of the copy used for motion analysis |
| `blur_kernel_size` | [21, 21] | Gaussian blur before background subtraction |
| `min_motion_contour_area` | 500 | Smallest moving region, in pixels of the 640-pixel copy |
| `motion_cooldown_seconds` | 20.0 | Time without motion that ends an event |
