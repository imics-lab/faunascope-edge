### Module B: Animal Analysis and Data Packaging

#### 1. Overview

The `AnimalAnalyzer` receives the frames of each motion event from the Motion Detector (Module A), decides whether the event shows an animal or a person, and, if it does, saves the event as an MP4 video and a JSON metadata file in `data/pending_upload/`, where the uploader (Module C) finds them. Events that show neither are discarded, or moved to `data/temp_debug/` when `debug_mode` is on.

#### 2. Interface with Module A

The analyzer reads a `multiprocessing.Queue`. A motion event arrives as a sequence of BGR frames (`numpy.ndarray`, `uint8`, shape `(height, width, 3)`) followed by `None`, which ends the event.

#### 3. Detection

`megadetector.py` runs MegaDetector V6 (`MDV6-mit-yolov9-c`), the MegaDetector variant trained with MultimediaTechLab's MIT-licensed YOLO code, in the ONNX form that `scripts/export_megadetector.py` makes from the published checkpoint. It finds three classes, `animal`, `person`, and `vehicle`, and does not name species. A frame is scaled to fit 640x640 and padded with gray, as in Pytorch-Wildlife; the 640x480 camera frames are only padded. Detections below `confidence_threshold` are dropped, and boxes of one class that overlap by an IoU above 0.5 are reduced to the most confident one. On a 640x480 frame the results match Pytorch-Wildlife's own implementation.

ONNX Runtime runs the model with the first available of its TensorRT (in FP16), CUDA, and CPU execution providers. With TensorRT, the first start builds an engine from the model, which takes about 8 minutes on the Jetson Orin Nano, and keeps it in `detector_engine_cache_dir`; later starts load it in about 3 seconds. On the Orin Nano a detection takes about 29 ms (13 ms of it inference) and a tracker step about 2 ms, and the FP16 engine gives the same boxes as the FP32 model on 120 archive frames, with confidences within 0.04.

#### 4. Tracking and event significance

MegaDetector runs on the first frame of an event and on every `detection_interval_frames`-th frame after it. `tracker.py` connects its boxes into tracks and follows them through the frames in between:

1. On a detection frame, each detection continues the track of the same class whose box it overlaps most, if their IoU is at least `track_match_iou`; every other detection starts a new track. A track that no detection continues is hidden, and it ends after more than `track_max_missed_detections` such detection frames in a row.
2. Between detection frames, the box of each visible track moves with the median Lucas-Kanade optical flow of corner points inside it. When too few points can be followed, the box stays where it is.

A track is confirmed once it has been detected in `confirmation_detections` detection frames, so a single false detection never keeps an event. The event is significant once a track of one of the `trigger_classes` is confirmed. Only confirmed tracks are written to the metadata, with their boxes from their first detection onward.

#### 5. Memory

Frames are held in RAM until `ram_frame_limit` is reached; the buffer is then encoded to a temporary video part in `output_temp_dir`. A significant event's parts are joined and re-encoded with FFmpeg (H.264) into the final MP4.

#### 6. JSON metadata

```json
{
  "eventId": "event_20261108-120000_1a2b3c4d",
  "deviceId": "IMICS-Jetson",
  "location": {"latitude": 29.9, "longitude": -97.9, "timezone": "America/Chicago"},
  "timestamp_start_utc": "2026-11-08T18:00:00.123456+00:00",
  "timestamp_end_utc": "2026-11-08T18:00:41.654321+00:00",
  "local_video_path": "data/pending_upload/event_20261108-120000_1a2b3c4d.mp4",
  "video_duration_seconds": 41.5,
  "event_summary": {
    "species_list": ["animal"],       // classes of the confirmed tracks
    "primary_species": "animal",      // the class with the most boxes
    "max_confidence": 0.93
  },
  "detector": {
    "model": "MDV6-mit-yolov9-c.onnx",
    "model_sha256": "…",
    "confidence_threshold": 0.2,
    "detection_interval_frames": 5,
    "confirmation_detections": 2
  },
  "detections": [                     // one entry per confirmed track per frame
    {
      "frame_index": 5,               // frame number in the event's video
      "track_id": 1,
      "label": "animal",
      "confidence": 0.91,             // the track's most recent detection
      "detected": true,               // false when the box was moved by the tracker
      "box_xywh": [320, 240, 80, 60]  // center x, center y, width, height in pixels
    }
  ]
}
```
