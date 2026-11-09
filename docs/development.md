# Development

## Tests

The unit tests in `tests/` run without a camera, a GPU, a model, or a server. They need `ffmpeg` on the PATH and the packages in `requirements-dev.txt`:

```sh
pip install -r requirements-dev.txt
pytest
```

| File | What it checks |
|---|---|
| `test_config.py` | Loading `config.yaml` with values from `.env`; the committed configuration parses. One test is an expected failure that documents an open problem: an unset variable stays in the configuration as literal text |
| `test_motion_detector.py` | Resizing and grayscale conversion for motion analysis, the minimum area of a moving region, and the frames, read times, and motion flags an event sends to the analyzer |
| `test_megadetector.py` | MegaDetector's padding, scaling of boxes back to the frame, and per-class suppression, with a stand-in for the model |
| `test_tracker.py` | Track creation, matching by class and overlap, missed detections, and following a textured box with optical flow |
| `test_species_classifier.py` | Averaging over crops, batching, geofencing, taxonomic roll-up, the blank threshold, and the person rule, with a stand-in model and a small label set |
| `test_analyzer_event.py` | A whole event through the analyzer with scripted detections and species: which frames are checked, which tracks are written, the video (at the measured frame rate) and metadata produced, species counts, and held events |
| `test_heartbeat.py` | Operating time from the camera's and the analyzer's state, the heartbeat file and its counts, and the clock and commit reported |
| `test_uploader.py` | The uploader against an in-process fake server that can drop connections, refuse requests, or misreport offsets: resuming, back-off, deferral, `.failed` markers, the daily budget, the rule that files move only after confirmation, and heartbeats |

## Running on recorded video

Three ways to run the pipeline on video files instead of the camera:

- **The whole application**: `python main.py --video <file>` runs motion detection, analysis, and uploading on the file at its own frame rate, as if it were the camera.
- **The analyzer alone**: `python scripts/test_analyzer.py --video <file>` feeds every frame of the file to the analyzer as one motion event and writes the event to `data/pending_upload/` or `data/held_blank/`. The script does not exit on its own; stop it once the event is written.
- **Many recordings at once**: `scripts/replay_archive.py` runs the analyzer over every `.mp4` in a folder, each as one event, without re-encoding the videos, in several worker processes:
  ```sh
  python scripts/replay_archive.py --videos <folder> --output <results folder> --workers 8
  ```
  It writes one JSON file per video with the outcome (`uploaded`, `held`, or `discarded`), the metadata the analyzer would write (with frame times taken from the video's frame rate), and every MegaDetector detection down to `--raw-threshold` (0.1) on the detection frames, so other confidence thresholds and tracking settings can be tried later without running the detector again. Existing results are skipped, so an interrupted run can be restarted. This is how the edge models are compared with the server's archive.

## Helper scripts

| Script | Purpose |
|---|---|
| `scripts/test_camera.py` | Open the camera the way the motion detector does and report what it delivers |
| `scripts/test_detector.py` | Run the motion detector on a video file with a debug window showing the motion mask |
| `scripts/stream_server.py` | The live view server (started by the container) |
| `scripts/view_live_local.py` | Show the live view in a window on the device |
| `scripts/export_megadetector.py`, `scripts/export_speciesnet.py` | Convert the published models to ONNX (see [Models](models.md)) |
| `scripts/configure-camera-exposure.sh` | Set the USB camera's exposure through `v4l2-ctl` |
| `scripts/toggle-auto-updates.sh` | Turn the Jetson's automatic system updates on or off |

## Conventions

- `config/config.yaml` holds algorithm settings and is committed; `.env` holds per-device values and secrets and is not. A setting that differs between devices goes in `.env` and is referenced from `config.yaml` as `${NAME}`.
- Each module runs in its own process, started with `spawn`, so the models are loaded inside the analyzer's process, in `AnimalAnalyzer.start()`.
- Files that another process reads are written under a temporary name and renamed, so a reader never sees a half-written file.
- The edge never deletes an event that has not been confirmed by the server.
