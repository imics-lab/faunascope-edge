# Event metadata and heartbeats

Every kept event is a pair of files with the same name: `event_<YYYYMMDD-HHMMSS>_<8 hex digits>.mp4` and `.json`. The JSON file is the event's record: the server builds its database entry from it, and nothing else about the event travels with the video. Besides events, the device sends an hourly heartbeat that reports what it did not keep and whether it was running at all.

This document is the contract between devices and the server. The current event format is version 2 (November 2026); the server reads every version it has ever accepted, choosing the parser by `format_version`, and refuses versions it does not know. The format follows five requirements:

1. **Every time is real.** Frame times are the times the frames were read from the camera, so durations, behavior bouts, and the video's playback speed match what happened. (In version 1, the median video played in 78% of its event's elapsed time, because videos were encoded at a fixed 30 frames per second.)
2. **Every time can be trusted or corrected.** The file says whether the device clock was synchronized, and carries enough to correct it later if it was not.
3. **Every result can be reproduced.** The file names the exact code, models, and settings that produced it.
4. **The track is the unit.** Each animal the device followed is one object, with its class, its species, and its boxes.
5. **The device reports; the server decides.** Location, species, and counts in the file are what the device measured. Authoritative values (a deployment's location, a reviewed species) live on the server.

Both files are written compactly (no indentation) and sent gzipped (see [Uploading](uploading.md)).

## Example

```json
{
  "format_version": 2,
  "event_id": "event_20261108-120000_1a2b3c4d",
  "device_id": "IMICS-Jetson",
  "edge_decision": "upload",

  "software": {"source_sha256": "8f2c…", "commit": "5288d77"},
  "models": {
    "detector": {"name": "MDV6-mit-yolov9-c", "sha256": "4cd3c0ef…", "runtime": "TensorRT FP16"},
    "classifier": {"name": "SpeciesNet", "version": "4.0.3a", "sha256": "1f600bb5…", "runtime": "TensorRT FP16"}
  },
  "settings": {
    "motion": {"frame_width": 640, "blur_kernel": [21, 21], "min_contour_area": 500, "cooldown_seconds": 20.0},
    "detector": {"confidence_threshold": 0.2, "iou_threshold": 0.5, "interval_frames": 5},
    "tracker": {"match_iou": 0.3, "max_missed_detections": 2, "confirmation_detections": 2},
    "classifier": {"crops_per_track": 5, "geofence": "USA-TX"},
    "trigger_classes": ["animal", "person"]
  },

  "clock": {"synchronized": true, "boot_id": "3f1c9d2e-…", "boot_seconds_at_start": 86412.337},
  "time": {"start_utc": "2026-11-08T18:00:00.123Z", "end_utc": "2026-11-08T18:01:02.654Z", "timezone": "America/Chicago"},
  "location": {"latitude": 29.9, "longitude": -97.9, "source": "device configuration"},

  "video": {
    "file": "event_20261108-120000_1a2b3c4d.mp4",
    "width": 640, "height": 480,
    "frames": 1562, "frames_dropped": 0,
    "frame_rate": 24.981, "duration_seconds": 62.527,
    "frame_offsets_ms": [0, 40, 79, 119, "…one per frame…"]
  },
  "motion": {"last_motion_frame": 1060},

  "summary": {
    "species": [{"taxon": "5c7ce479-…;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer",
                 "name": "white-tailed deer", "tracks": 2, "max_count": 2}],
    "primary_species": "white-tailed deer"
  },

  "tracks": [
    {
      "track_id": 1,
      "class": "animal",
      "first_frame": 5, "last_frame": 412, "confirmed_frame": 10,
      "species": {
        "taxon": "5c7ce479-…;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer",
        "name": "white-tailed deer", "score": 0.9717, "source": "classifier", "blank": false,
        "crop_frames": [35, 40, 120, 125, 130],
        "top": [{"taxon": "…;white-tailed deer", "score": 0.9717}, {"taxon": "…;mule deer", "score": 0.0103}]
      },
      "boxes": {
        "frame": [5, 6, 7, "…"],
        "xyxy": [[280, 210, 360, 270], [282, 210, 362, 270], "…"],
        "detected": [true, false, false, "…"],
        "confidence": [0.91, null, null, "…"]
      }
    }
  ]
}
```

## Event fields

### Identity and decision

| Field | Type | Meaning |
|---|---|---|
| `format_version` | integer | `2`. Files without this field are version 1 |
| `event_id` | string | `event_<YYYYMMDD-HHMMSS>_<8 hex digits>`: the device's local time at the first frame and a random suffix; also the file names |
| `device_id` | string | `DEVICE_ID` from `.env`; it must match the device the upload token was issued to |
| `edge_decision` | string | `upload`, or `held_blank` (SpeciesNet called every animal and person track blank). A held event that a person later moves to the pending folder keeps `held_blank`, so the server knows it was released by hand |

### Provenance

| Field | Meaning |
|---|---|
| `software.source_sha256` | SHA-256 over the device's Python sources (`main.py` and `src/`, each with its path), computed at start. It identifies the code exactly, even without git |
| `software.commit` | The commit the code comes from: the `FAUNASCOPE_COMMIT` build argument of the Docker image (see [Deployment](deployment.md)), or the checkout's HEAD when run outside Docker; absent if neither is known |
| `models.detector`, `models.classifier` | Model name (and version), SHA-256 of the file loaded, and the runtime that executed it (`TensorRT FP16`, `CUDA FP32`, or `CPU FP32`), because borderline results differ between runtimes |
| `settings` | Every configuration setting that affects what is recorded, kept, or labeled. Two events with the same `software`, `models`, and `settings` were judged by the same pipeline |

### Time

| Field | Meaning |
|---|---|
| `clock.synchronized` | Whether the kernel considered the system clock synchronized (by NTP) when the event started, as `adjtimex` reports it; `null` if it could not be asked |
| `clock.boot_id` | The kernel's boot ID (`/proc/sys/kernel/random/boot_id`), unique per boot |
| `clock.boot_seconds_at_start` | Seconds since boot (`CLOCK_BOOTTIME`) when the first frame was read |
| `time.start_utc`, `time.end_utc` | When the first and the last frame were read from the camera, in UTC, ISO 8601 with milliseconds. The end is the start plus the last frame's offset, so a clock change during the event does not distort it |
| `time.timezone` | `DEVICE_TIMEZONE`, the device's IANA time zone, for local-time displays and analyses |

The Jetson has no battery-backed clock: after a power loss it runs from the last saved time until it reaches a time server. If `clock.synchronized` is false, the server can correct the event later: a heartbeat sent after synchronization during the same boot (same `boot_id`) gives `boot_seconds_at_end` with a true `end_utc`, and so the true time of every moment of that boot.

### Video and motion

| Field | Meaning |
|---|---|
| `video.file` | The MP4's file name |
| `video.width`, `video.height` | Frame size in pixels; box coordinates refer to it |
| `video.frames` | Frames in the video |
| `video.frames_dropped` | Frames read from the camera but not passed to the analyzer. Always 0 for now, because the queue to the analyzer is unbounded (audit H5); frames the camera driver drops show as gaps in `frame_offsets_ms` |
| `video.frame_rate` | The average rate at which the frames were read, `(frames - 1) / (last offset / 1000)`. The MP4 is encoded at this rate, so it plays at real speed |
| `video.duration_seconds` | `frames / frame_rate` |
| `video.frame_offsets_ms` | When each frame was read, in milliseconds after `time.start_utc`, measured on `CLOCK_BOOTTIME`. The camera's rate varies (it falls in low light), so a frame's true time is its offset, not its index divided by the rate |
| `motion.last_motion_frame` | The last frame in which the motion detector saw motion; the frames after it are the cooldown. The first frame is always one with motion |

### Summary

Derived from `tracks`, for convenience.

| Field | Meaning |
|---|---|
| `summary.species` | One entry per species SpeciesNet named (blank excluded): the SpeciesNet `taxon` and common `name`, the number of `tracks`, and `max_count`, the largest number of that species' tracks visible in one frame. The species with the most boxes comes first, except that `"animal"` and `"no cv result"` (animals SpeciesNet could not place) come after every named taxon |
| `summary.primary_species` | The `name` of the first entry, or `"blank"` |

Counting individuals: `tracks` overstates the number of animals, because an animal that disappears briefly gets a new track (see [Animal analysis](analysis.md#tracking)); `max_count` is a minimum count.

The names are SpeciesNet's common names: a species (`"white-tailed deer"`), a higher taxon when the classifier was not sure enough (`"cervidae family"`, `"bird"`), or one of `"human"`, `"vehicle"`, `"animal"` (the detector saw an animal the classifier could not place), and `"no cv result"` (neither could say).

### Tracks

One entry per confirmed track (detected in at least `confirmation_detections` detection frames), in track order. Unconfirmed tracks are single false detections in most cases and are not recorded; the video is on the server for any re-analysis.

| Field | Meaning |
|---|---|
| `track_id` | Unique within the event |
| `class` | MegaDetector's class: `animal`, `person`, or `vehicle` |
| `first_frame`, `last_frame` | The frames of the track's first and last box |
| `confirmed_frame` | The frame in which the track was confirmed |
| `species.taxon` | The full SpeciesNet label: identifier, class, order, family, genus, species, and common name |
| `species.name` | The common name |
| `species.score`, `species.source` | The label's score, and the rule that produced it: `classifier` (the classifier's top label), `classifier+rollup_to_<level>` (probabilities summed up the taxonomy), or `detector` (MegaDetector's class, when the classifier could not decide) |
| `species.blank` | True if SpeciesNet called the track an empty scene |
| `species.crop_frames` | The frames whose crops were classified, so the server can show or re-classify the same crops |
| `species.top` | The classifier's five most probable labels, averaged over the crops, as full labels with scores |
| `boxes` | Parallel arrays with one element per frame in which the track is visible: `frame` (index into the video), `xyxy` (corners in pixels, integers), `detected` (true if MegaDetector produced this box, false if the tracker moved it), and `confidence` (MegaDetector's confidence for detected boxes, `null` for tracked ones) |

`species` is `null` for a track without a crop large enough to classify.

## Added by the server

When the server accepts an event, it adds `received_at` (UTC) and `video_sha256` to its copy of the JSON. Everything the server adds later (species and behaviors from its AI model, corrections made by people) stays on the server; the device's files never change after they are written. The server takes a deployment's authoritative location from its own records rather than from `location`.

## Heartbeat, format version 1

Event files describe what the device kept. What it did not keep, and whether it was running at all, is needed for ecology (sampling effort) and for evaluating the decisions on the device. The main process (`src/heartbeat.py`) therefore writes a heartbeat once an hour, and once more when the application shuts down, as `heartbeat_<YYYYMMDD-HHMMSS>.json` in the pending folder. The uploader sends heartbeats before events, oldest first, and deletes each once the server confirms it, so none is lost while the device is offline.

```json
{
  "format_version": 1,
  "device_id": "IMICS-Jetson",
  "period": {"start_utc": "2026-11-08T18:00:00Z", "end_utc": "2026-11-08T19:00:00Z"},
  "software": {"source_sha256": "8f2c…", "commit": "5288d77"},
  "clock": {"synchronized": true, "boot_id": "3f1c9d2e-…", "boot_seconds_at_end": 90012.551},
  "operating_seconds": {"camera": 3600, "analyzer": 3600},
  "events": {"motion": 14, "discarded": 9, "held_blank": 1, "kept": 4},
  "queue": {"pending": 3, "held_blank": 27, "failed": 0, "uploaded_bytes": 41943040},
  "storage": {"free_bytes": 738197504000},
  "temperature_c": {"cpu": 48.5, "gpu": 47.0}
}
```

| Field | Meaning |
|---|---|
| `period` | The time covered: from the previous heartbeat (or the start of the application) to this one |
| `clock` | As in events; `boot_seconds_at_end` with `period.end_utc` gives the correction for unsynchronized events of the same boot |
| `operating_seconds` | Seconds of the period during which the camera delivered frames (`camera`) and the analyzer was running with its models loaded (`analyzer`). This is the sampling effort |
| `events` | Motion events that ended in the period, and what became of them: discarded (no confirmed animal or person), held as blank, or kept for upload. `motion` also counts an event whose video could not be encoded |
| `queue` | At the end of the period: events waiting for upload, events held as blank, events marked `.failed`, and the bytes the uploader sent during the period |
| `storage.free_bytes` | Free space on the data volume |
| `temperature_c` | Each thermal zone's temperature, by zone (`cpu`, `gpu`, ... on the Jetson) |

A heartbeat the server refuses is renamed `.failed` and kept. Battery and solar figures will be added where the hardware reports them.

## Version 1

Events recorded before format version 2 have no `format_version`; the whole archive is in this format. Their fields are camelCase identifiers (`eventId`, `deviceId`), times taken when the analyzer received the first frame and the end of the event (`timestamp_start_utc`, `timestamp_end_utc`), a video encoded at a fixed 30 frames per second (`video_duration_seconds`), a `location` that includes the time zone, an `event_summary` with `species_list`, `primary_species`, and `max_confidence`, and one entry per box in `detections` (`frame_index`, `label`, `confidence`, and `box_xywh` as center and size). The labels are the class names of the species YOLO model the devices ran then (`White-Tailed_Deer`, `Vulture`, ...).
