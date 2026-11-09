### Module B: Animal Analysis and Data Packaging

#### 1. Overview

The `AnimalAnalyzer` receives the frames of each motion event from the Motion Detector (Module A), decides whether the event shows an animal or a person, names the species, and saves the event as an MP4 video and a JSON metadata file in `data/pending_upload/`, where the uploader (Module C) finds them. Events that show neither are discarded, or moved to `data/temp_debug/` when `debug_mode` is on, and events whose animals SpeciesNet finds to be empty scenes are saved in `data/held_blank/`, which is not uploaded.

#### 2. Interface with Module A

The analyzer reads a `multiprocessing.Queue`. A motion event arrives as a sequence of BGR frames (`numpy.ndarray`, `uint8`, shape `(height, width, 3)`) followed by `None`, which ends the event.

#### 3. Detection

`megadetector.py` runs MegaDetector V6 (`MDV6-mit-yolov9-c`), the MegaDetector variant trained with MultimediaTechLab's MIT-licensed YOLO code, in the ONNX form that `scripts/export_megadetector.py` makes from the published checkpoint. It finds three classes, `animal`, `person`, and `vehicle`, and does not name species. A frame is scaled to fit 640x640 and padded with gray, as in Pytorch-Wildlife; the 640x480 camera frames are only padded. Detections below `confidence_threshold` are dropped, and boxes of one class that overlap by an IoU above 0.5 are reduced to the most confident one. On a 640x480 frame the results match Pytorch-Wildlife's own implementation.

ONNX Runtime runs the model with the first available of its TensorRT (in FP16), CUDA, and CPU execution providers. With TensorRT, the first start builds an engine from the model, which takes about 8 minutes on the Jetson Orin Nano, and keeps it in `engine_cache_dir`; later starts load it in about 3 seconds. On the Orin Nano a detection takes about 29 ms (13 ms of it inference) and a tracker step about 2 ms, and the FP16 engine gives the same boxes as the FP32 model on 120 archive frames, with confidences within 0.04.

#### 4. Tracking and event significance

MegaDetector runs on the first frame of an event and on every `detection_interval_frames`-th frame after it. `tracker.py` connects its boxes into tracks and follows them through the frames in between:

1. On a detection frame, each detection continues the track of the same class whose box it overlaps most, if their IoU is at least `track_match_iou`; every other detection starts a new track. A track that no detection continues is hidden, and it ends after more than `track_max_missed_detections` such detection frames in a row.
2. Between detection frames, the box of each visible track moves with the median Lucas-Kanade optical flow of corner points inside it. When too few points can be followed, the box stays where it is.

A track is confirmed once it has been detected in `confirmation_detections` detection frames, so a single false detection never keeps an event. The event is significant once a track of one of the `trigger_classes` is confirmed. Only confirmed tracks are written to the metadata, with their boxes from their first detection onward.

#### 5. Species classification

When a significant event ends, `species_classifier.py` classifies each confirmed track with SpeciesNet 4.0.3a, Google's camera-trap classifier for 2,498 labels (species, higher taxa, blank, human, vehicle), in the ONNX form that `scripts/export_speciesnet.py` makes from the PyTorch release. During the event, the analyzer keeps a 480x480 crop of each track's `classification_crops` most confident detections, cut and resized as SpeciesNet's own preprocessing does; on 130 archive crops the classifier's top label matched SpeciesNet's preprocessing in 129 cases. The classifier's probabilities are averaged over a track's crops, and the five most probable labels go through the SpeciesNet ensemble rules, copied unchanged from the `speciesnet` package into `speciesnet/`, together with the track's MegaDetector class and highest confidence. These rules accept a label only above their score thresholds and where the geofence places it (`geofence_country`, `geofence_admin1_region`), and otherwise roll the probabilities up to genus, family, order, class, or kingdom. With MegaDetector's confidence at 0.2 or more, they call a track blank only when its averaged probability of blank exceeds 0.99.

If every confirmed track of the `trigger_classes` is blank, the event goes to `output_held_dir` instead of `output_pending_dir`.

#### 6. Memory

Frames are held in RAM until `ram_frame_limit` is reached; the buffer is then encoded to a temporary video part in `output_temp_dir`. A significant event's parts are joined and re-encoded with FFmpeg (H.264) into the final MP4.

#### 7. JSON metadata

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
    "species_list": ["white-tailed deer"],  // SpeciesNet's names for the tracks it does not call blank, the track with the most boxes first
    "primary_species": "white-tailed deer", // the first of them, or "blank"
    "max_confidence": 0.93                  // MegaDetector's highest confidence
  },
  "detector": {
    "model": "MDV6-mit-yolov9-c.onnx",
    "model_sha256": "…",
    "confidence_threshold": 0.2,
    "detection_interval_frames": 5,
    "confirmation_detections": 2
  },
  "classifier": {
    "model": "SpeciesNet 4.0.3a",
    "model_sha256": "…",
    "crops_per_track": 5,
    "geofence": "USA-TX"
  },
  "classifications": [                // one entry per confirmed track
    {
      "track_id": 1,
      "detector_label": "animal",
      "crops": 5,
      "prediction": "white-tailed deer",
      "taxonomy": "5c7ce479-…;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer",
      "score": 0.98,
      "source": "classifier",         // or e.g. "classifier+rollup_to_family", "detector"
      "blank": false,
      "top_classes": [{"label": "white-tailed deer", "score": 0.98}, …]
    }
  ],
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
