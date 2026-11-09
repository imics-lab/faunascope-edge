# Deployment

FaunaScope Edge runs in a Docker container on an NVIDIA Jetson (the field nodes are Jetson Orin Nano 8 GB developer kits with JetPack 6) or on a Linux machine with an NVIDIA GPU. The container holds the Python environment, GPU access, and the camera; the configuration, models, data, and logs stay on the host and are mounted into it.

## Prerequisites

- **Docker** 20.10 or later, with your user in the `docker` group.
- **NVIDIA Container Toolkit**, so containers can use the GPU:
  ```sh
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  ```
- **On a Jetson**: JetPack installed on the host.
- **A camera**: a USB webcam (MJPEG) or a CSI camera.
- **A device token** from the FaunaScope server's administrator, to upload (see [Uploading](uploading.md#credentials)).

## Two compose files

| File | For | Base image |
|---|---|---|
| `docker-compose-jetson.yml` | Jetson devices (ARM64) | `dustynv/l4t-ml:r36.4.0`, which matches JetPack 6 and includes ONNX Runtime with TensorRT |
| `docker-compose-gpu.yml` | x86-64 machines with an NVIDIA GPU | `nvcr.io/nvidia/pytorch:22.12-py3`; the Dockerfile adds `onnxruntime-gpu` |

Both build the image `faunascope-edge-gpu:latest` and name the container `FaunaScope-edge-gpu`, so only one can run at a time. The container runs in privileged mode with the camera devices (`/dev/video0` to `/dev/video3`, `/dev/media1`) mapped in; edit the `devices` list for other cameras.

## Setup

1. **Get the code.**
   ```sh
   git clone https://github.com/imics-lab/faunascope-edge.git
   cd faunascope-edge
   ```

2. **Device settings.** Copy `env.example` to `.env` and fill it in. `.env` is never committed.

   | Variable | Meaning |
   |---|---|
   | `VIDEO_SOURCE` | Camera index (`0`) or a video file |
   | `CAMERA_WIDTH`, `CAMERA_HEIGHT`, `CAMERA_FPS` | Camera mode; the field nodes use 640 x 480 at 30 fps |
   | `DEVICE_ID` | The device's name, as registered on the server |
   | `DEVICE_LATITUDE`, `DEVICE_LONGITUDE`, `DEVICE_TIMEZONE` | Where the device is, and its IANA time zone |
   | `UPLOADER_ENABLED` | `true` to upload |
   | `SERVER_URL`, `DEVICE_TOKEN` | The server and this device's token |

   The container's user (uid and gid 999) must be able to read `.env`. On a Jetson, gid 999 is the host's `gpio` group, so `chgrp gpio .env && chmod 640 .env` keeps the file private and readable by the container.

3. **Algorithm settings** are in `config/config.yaml`, which is committed: motion thresholds, the detector's confidence threshold and trigger classes, upload limits, and the live view. [Motion detection](motion-detection.md), [Animal analysis](analysis.md), and [Uploading](uploading.md) explain each setting.

4. <a id="model-weights"></a>**Model weights.** MegaDetector, `model_weight/MDV6-mit-yolov9-c.onnx`, comes with the repository. SpeciesNet is downloaded from the repository's releases:
   ```sh
   mkdir -p model_weight/speciesnet
   for f in speciesnet.onnx labels.txt taxonomy.txt geofence.json version.txt; do
       curl -L -o model_weight/speciesnet/$f https://github.com/imics-lab/faunascope-edge/releases/download/speciesnet-v4.0.3a/$f
   done
   ```

5. **Build and start.**
   ```sh
   docker compose -f docker-compose-jetson.yml up --build -d
   ```
   The first start converts both models into TensorRT engines, which takes about 20 minutes on an Orin Nano; nothing is recorded until it finishes. The engines are kept in `data/engine_cache/`, and later starts take seconds. The container restarts automatically after a crash or a reboot (`restart: unless-stopped`).

## What runs in the container

`docker-entrypoint.sh` checks that `.env` and the models are present, gives the container user (`FaunaScope`) ownership of the data and log folders and membership in the groups that own the GPU and camera device nodes, starts a virtual display (Xvfb), and then starts two processes as that user:

- `main.py`, which runs the motion detector, the animal analyzer, and the uploader as three processes;
- `scripts/stream_server.py`, the live view server on port 8080.

If either process exits, the container stops and Docker restarts it.

The host folders mounted into the container are `config/config.yaml` and `.env` (read-only), `model_weight/` (read-only), `data/pending_upload/`, `data/uploaded/`, `data/held_blank/`, `data/temp_debug/`, `data/engine_cache/`, and `logs/`. Temporary video parts go to a 1 GB RAM disk.

## Daily operation

```sh
docker compose -f docker-compose-jetson.yml logs -f        # follow the logs
docker compose -f docker-compose-jetson.yml down           # stop
docker compose -f docker-compose-jetson.yml up --build -d  # rebuild after a code change
ls data/pending_upload/ data/held_blank/                   # what is waiting, and what was held
cat data/pending_upload/.upload_state.json                 # the uploader's counters
```

Each module also writes its own log in `logs/` (`motion_detector.log`, `animal_analyzer.log`, `data_uploader.log`). To empty Docker's log of the container: `sudo truncate -s 0 $(docker inspect --format='{{.LogPath}}' FaunaScope-edge-gpu)`.

## Live view

With `live_view.enabled: true`, open `http://<device address>:8080` in a browser on the same network. The stream pauses after `stream_timeout_seconds` (60 by default) to protect the data plan; reload the page to continue. From outside the device's network, reach port 8080 through a VPN or an SSH tunnel. On the device's own desktop, `python scripts/view_live_local.py` shows the same feed in a window. [Motion detection](motion-detection.md#live-view) explains how the live view stays idle while nobody watches.

## Troubleshooting

| Problem | Check |
|---|---|
| The GPU is not found | `docker info \| grep -i runtime` on the host should list `nvidia`; `docker exec -it FaunaScope-edge-gpu nvidia-smi` (on a desktop GPU) |
| The camera is not found | `ls -la /dev/video*` on the host, and that the device is mapped in the compose file; `python scripts/test_camera.py` tests the camera directly |
| `Permission denied` on `config.yaml` or `.env` | The files must be readable by the container user (see Setup, step 2) |
| Nothing is recorded after a fresh start | The TensorRT engines are still being built; `logs/animal_analyzer.log` shows when the models have loaded |
| Events are not uploaded | `logs/data_uploader.log`; the token check in [Uploading](uploading.md#credentials); `.failed` markers in `data/pending_upload/` |
| A shell inside the container | `docker exec -it FaunaScope-edge-gpu /bin/bash` |
