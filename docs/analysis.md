# Animal analysis

The animal analyzer (`src/animal_analyzer/analyzer.py`, class `AnimalAnalyzer`) receives the frames of each motion event, decides whether the event shows an animal or a person, names the species, and packages the event as an MP4 video and a JSON metadata file. It uses two models in sequence: MegaDetector, which finds animals, people, and vehicles while the event is running, and SpeciesNet, which names the species of each animal once the event has ended. [Models](models.md) describes where both come from and how they were checked.

## Input

The analyzer reads the queue filled by the motion detector. An event arrives as a sequence of BGR frames, each with the time it was read from the camera and whether it showed motion, followed by `None` (see [Motion detection](motion-detection.md#events)). Frame `i` of the queue is frame `i` of the event's video, which is how the boxes in the metadata are tied to the video.

## Detection: MegaDetector

`megadetector.py` runs MegaDetector V6 (`MDV6-mit-yolov9-c`), a camera-trap detector with three classes, `animal`, `person`, and `vehicle`. It runs on the first frame of an event and on every `detection_interval_frames`-th frame after it (every fifth frame by default, six times a second at 30 frames per second).

- **Preprocessing**, as in Pytorch-Wildlife: the frame is scaled to fit 640 x 640 and centered on a gray (114) background. A 640 x 480 camera frame is only padded, 80 rows above and below; a larger frame is shrunk with area averaging.
- **Postprocessing**: detections below `confidence_threshold` (0.2) are dropped, and among boxes of the same class that overlap by an IoU above 0.5 only the most confident is kept. Boxes of different classes never suppress each other, so a person next to a dog keeps both boxes.

On native 640 x 480 frames the detections match Pytorch-Wildlife's own implementation exactly.

## Tracking

`tracker.py` (class `BoxTracker`, adapted from FaunaCodec's ROI tracker) connects MegaDetector's boxes into tracks, one per animal, and follows them through the frames between detector runs.

1. **On a detection frame**, each detection continues the track of the same class whose box it overlaps most, if their IoU is at least `track_match_iou` (0.3). Every other detection starts a new track. A track that no detection continues is hidden; it ends after more than `track_max_missed_detections` (2) such detection frames in a row, that is, after about half a second unseen at the default settings.
2. **Between detection frames**, the box of each visible track moves with the median Lucas-Kanade optical flow of up to 80 corner points found inside it. When fewer than 10 points can be followed, the box stays where it is.

Each box written to the metadata says whether it came from the detector in that frame (`detected: true`) or was moved by the tracker.

Known limits, all of which show most in night footage:

- MegaDetector does not distinguish species, so two animals that overlap by more than half can be merged into one box and one track.
- Matching uses box overlap only, so two animals that cross can swap track IDs.
- An animal unseen for more than about half a second, behind vegetation or in weak infrared light, comes back as a new track. The number of tracks is therefore not the number of individuals; the largest number of tracks of one species visible in a single frame is the defensible count.

## Event significance

A track is **confirmed** once it has been detected in `confirmation_detections` (2) detection frames, so a single false detection never keeps an event. The event is **significant** once a confirmed track belongs to one of the `trigger_classes` (`animal` and `person` by default). Only confirmed tracks are written to the metadata, each with its boxes from its first detection onward. Confirmed vehicle tracks are recorded but do not by themselves keep an event.

An event without a significant track is discarded.

## Species classification: SpeciesNet

When a significant event ends, `species_classifier.py` classifies every confirmed track with SpeciesNet 4.0.3a, Google's camera-trap classifier for 2,498 labels: species, higher taxa (genus, family, order, class), and the special labels blank, animal, human, and vehicle.

1. **Crops.** During the event, the analyzer keeps a 480 x 480 crop of each track's `classification_crops` (5) most confident detections. Crops are cut and resized as SpeciesNet's own preprocessing does; on 130 archive crops the classifier's top label was the same as with SpeciesNet's preprocessing in 129 cases.
2. **Averaging.** The classifier's probabilities are averaged over a track's crops, which steadies the answer for an animal seen from several angles.
3. **Ensemble rules.** The five most probable labels, the track's MegaDetector class, and its highest detection confidence go through SpeciesNet's own rules for combining detector and classifier, copied unchanged from Google's `speciesnet` package into `src/animal_analyzer/speciesnet/`. A species is accepted only above the rules' score thresholds and only where the geofence places it (`geofence_country`, `geofence_admin1_region`, Texas by default). Otherwise the probabilities are summed up the taxonomy until a genus, family, order, class, or kingdom passes 0.65 (for example "cervidae family" or "bird"). If nothing passes, an animal track keeps the detector's "animal" (when its confidence is above 0.5) or gets "no cv result".
4. **People.** A person track with a detector confidence above 0.7 is "human" without consulting the classifier, which is SpeciesNet's own rule.
5. **Blanks.** Because every confirmed track has a detector confidence of at least 0.2, the rules call a track blank only when its averaged probability of blank exceeds 0.99.

The result for each track (label, score, the rule that produced it, the top five labels, and the frames of the crops) is written to the metadata, together with a summary per species: its tracks and the most of them visible in one frame.

## Held events

If SpeciesNet calls every confirmed track of the `trigger_classes` blank, the event is a detector false positive with high probability (a stain, a rock, a branch), and it goes to `output_held_dir` (`data/held_blank/`) instead of `data/pending_upload/`. Held events are kept with their metadata but not uploaded. Moving an event's two files into `data/pending_upload/` uploads it. The held folder has no size limit yet.

## Memory and video encoding

Frames are held in RAM until `ram_frame_limit` (700, about 650 MB at 640 x 480) is reached; the buffer is then encoded to a temporary video part in `output_temp_dir`, a RAM disk in the container. When a significant event ends, the remaining frames are written as a last part, and FFmpeg joins the parts and re-encodes them as H.264 (`-preset slow -crf 24`, `+faststart` so browsers can start playing before the download ends) into the final MP4. The parts are written at a nominal `output_fps`; FFmpeg retimes the frames to the average rate at which the camera delivered them, so the video plays at real speed whatever the camera's rate (which falls in low light). The metadata is written to a temporary name and renamed, so the uploader never reads a half-written file.

For an event that is not significant, the parts already written to disk are deleted, or moved to `data/temp_debug/` when `debug_mode` is on; frames still in RAM are dropped either way.

## Speed and memory on the Jetson Orin Nano

Measured on the Orin Nano 8 GB in its 15 W mode, with ONNX Runtime's TensorRT execution provider in FP16:

| Step | Time |
|---|---|
| MegaDetector, one frame (13 ms of it inference) | 29 ms |
| Tracker step between detections | 1.7 ms |
| SpeciesNet, one track from 5 crops | 95 ms |
| Start-up with cached TensorRT engines | 3 s (MegaDetector) + 4 s (SpeciesNet) |
| First start, building the engines | about 8 min + 11 min |

With a detection on every fifth frame, analysis needs about 8 ms per frame on average, against 33 ms between frames at 30 frames per second. The analyzer process with both models uses about 2.6 GB, plus the frames held for the current event. The FP16 engines give the same boxes as the FP32 model on 120 archive frames, with confidences within 0.04; SpeciesNet calls close to its thresholds can still differ between the Jetson and a desktop GPU.

## Settings

All in the `animal_analyzer` section of `config/config.yaml`:

| Setting | Default | Meaning |
|---|---|---|
| `detector_model_path` | `model_weight/MDV6-mit-yolov9-c.onnx` | MegaDetector in ONNX form |
| `trigger_classes` | `[animal, person]` | Classes whose confirmed tracks keep an event |
| `confidence_threshold` | 0.2 | Lowest MegaDetector confidence that counts |
| `detection_interval_frames` | 5 | MegaDetector runs on every N-th frame |
| `confirmation_detections` | 2 | Detection frames that confirm a track |
| `track_match_iou` | 0.3 | Overlap needed to continue a track |
| `track_max_missed_detections` | 2 | Missed detection frames after which a track ends |
| `classifier_model_dir` | `model_weight/speciesnet/` | SpeciesNet files |
| `classification_crops` | 5 | Crops per track sent to SpeciesNet |
| `geofence_country`, `geofence_admin1_region` | `USA`, `TX` | Where SpeciesNet's geofence is applied |
| `engine_cache_dir` | `data/engine_cache/` | TensorRT engines built on the first start |
| `ram_frame_limit` | 700 | Frames held in RAM before a part is written to disk |
| `output_pending_dir`, `output_held_dir` | `data/pending_upload/`, `data/held_blank/` | Where finished events go |
| `debug_mode`, `debug_temp_dir` | `true`, `data/temp_debug/` | Keep the parts of discarded events |
| `output_temp_dir` | `/tmp/FaunaScope_edge_temp/` | Temporary video parts |
