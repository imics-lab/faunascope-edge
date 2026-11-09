# Event metadata

Every kept event is a pair of files with the same name: `event_<YYYYMMDD-HHMMSS>_<8 hex digits>.mp4` and `.json`. The date and time in the name are the device's local time when the analyzer received the event's first frame; the hex digits make the name unique. The JSON file is the event's record: the server builds its database entry from it, and nothing else about the event travels with the video.

## Example

```json
{
  "eventId": "event_20261108-120000_1a2b3c4d",
  "deviceId": "IMICS-Jetson",
  "location": {"latitude": 29.9, "longitude": -97.9, "timezone": "America/Chicago"},
  "timestamp_start_utc": "2026-11-08T18:00:00.123456+00:00",
  "timestamp_end_utc": "2026-11-08T18:01:02.654321+00:00",
  "local_video_path": "data/pending_upload/event_20261108-120000_1a2b3c4d.mp4",
  "video_duration_seconds": 41.5,
  "event_summary": {
    "species_list": ["white-tailed deer", "wild turkey"],
    "primary_species": "white-tailed deer",
    "max_confidence": 0.93
  },
  "detector": {
    "model": "MDV6-mit-yolov9-c.onnx",
    "model_sha256": "4cd3c0ef…",
    "confidence_threshold": 0.2,
    "detection_interval_frames": 5,
    "confirmation_detections": 2
  },
  "classifier": {
    "model": "SpeciesNet 4.0.3a",
    "model_sha256": "1f600bb5…",
    "crops_per_track": 5,
    "geofence": "USA-TX"
  },
  "classifications": [
    {
      "track_id": 1,
      "detector_label": "animal",
      "crops": 5,
      "prediction": "white-tailed deer",
      "taxonomy": "5c7ce479-8a45-40b3-ae21-7c97dfae22f5;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer",
      "score": 0.9717,
      "source": "classifier",
      "blank": false,
      "top_classes": [{"label": "white-tailed deer", "score": 0.9717}, {"label": "mule deer", "score": 0.0103}]
    }
  ],
  "detections": [
    {"frame_index": 5, "track_id": 1, "label": "animal", "confidence": 0.91, "detected": true, "box_xywh": [320, 240, 80, 60]},
    {"frame_index": 6, "track_id": 1, "label": "animal", "confidence": 0.91, "detected": false, "box_xywh": [322, 240, 80, 60]}
  ]
}
```

## Fields

**Identity and time**

| Field | Meaning |
|---|---|
| `eventId` | The event's name, the same as the file names |
| `deviceId` | `DEVICE_ID` from `.env`; it must match the device the upload token was issued to |
| `location` | `DEVICE_LATITUDE`, `DEVICE_LONGITUDE`, and `DEVICE_TIMEZONE` from `.env` (an IANA zone such as `America/Chicago`) |
| `timestamp_start_utc` | When the analyzer received the event's first frame, in UTC (ISO 8601) |
| `timestamp_end_utc` | When the analyzer received the end of the event, after the motion cooldown, in UTC |
| `local_video_path` | Where the video was written on the device |
| `video_duration_seconds` | Length of the encoded video |

The times come from the device's clock. The Jetson has no battery-backed clock, so after a power loss it runs from the last saved time until it reaches a time server; events recorded in that window carry wrong times. Recording whether the clock was synchronized is planned.

**Summary**

| Field | Meaning |
|---|---|
| `event_summary.species_list` | SpeciesNet's names for the tracks it did not call blank, without repeats, the track with the most boxes first. Empty for a held event |
| `event_summary.primary_species` | The first entry of `species_list`, or `"blank"` |
| `event_summary.max_confidence` | MegaDetector's highest confidence among the recorded boxes |

The names are SpeciesNet's common names: a species (`"white-tailed deer"`), a higher taxon when the classifier was not sure enough (`"cervidae family"`, `"bird"`), or one of `"human"`, `"animal"` (the detector saw an animal the classifier could not place), and `"no cv result"` (neither could say).

**Models**

`detector` and `classifier` record which models and settings produced the event: the model file, the SHA-256 of its weights, and the settings that decide what is kept. Two events with the same hashes and settings were judged by the same pipeline.

**Classifications** (one entry per confirmed track)

| Field | Meaning |
|---|---|
| `track_id` | The track, as in `detections` |
| `detector_label` | MegaDetector's class for the track: `animal`, `person`, or `vehicle` |
| `crops` | How many crops SpeciesNet saw (up to `crops_per_track`) |
| `prediction` | SpeciesNet's answer, as a common name |
| `taxonomy` | The full SpeciesNet label: an identifier, then class, order, family, genus, species, and common name |
| `score` | The score of `prediction` |
| `source` | Which rule produced it: `classifier` (the classifier's top label), `classifier+rollup_to_<level>` (probabilities summed up the taxonomy), or `detector` (MegaDetector's class, when the classifier could not decide) |
| `blank` | True if SpeciesNet called the track an empty scene |
| `top_classes` | The classifier's five most probable labels, averaged over the crops |

**Detections** (one entry per confirmed track per frame in which it is visible)

| Field | Meaning |
|---|---|
| `frame_index` | Frame number in the event's video, starting at 0 |
| `track_id` | The track the box belongs to |
| `label` | MegaDetector's class |
| `confidence` | The confidence of the track's most recent detection |
| `detected` | True if MegaDetector produced this box in this frame; false if the tracker moved it from an earlier frame |
| `box_xywh` | Center x, center y, width, and height in pixels of the full-resolution frame |

Counting individuals: the number of distinct `track_id`s overstates the number of animals, because an animal that disappears briefly gets a new track (see [Animal analysis](analysis.md#tracking)). The largest number of tracks of one species in a single `frame_index` is a minimum count.

## Added by the server

When the server accepts an event, it adds `received_at` (UTC) and `video_sha256` to its copy of the JSON. Everything the server adds later (species and behaviors from its AI model, corrections made by people) stays on the server; the device's files never change after they are written.

## Earlier versions

Events recorded before November 2026 used a species YOLO model on the device. Their files have the same top-level fields, but `species_list` and `primary_species` hold that model's class names (`White-Tailed_Deer`, `Vulture`, ...), the detections have no `track_id` or `detected` fields, and there are no `detector`, `classifier`, or `classifications` blocks.
