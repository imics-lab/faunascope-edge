# FaunaScope-Edge

An intelligent, field-deployable animal monitoring system. FaunaScope-Edge uses a multi-stage AI pipeline to efficiently detect animal activity, capture high-quality video, and reliably upload significant events to a central database for scientific research.

## Overview

This repository contains the source code for the edge device component of the FaunaScope project. The system is designed to run on a power-constrained device (like a Jetson Nano Orin) deployed in the wild. It employs a three-module architecture to maximize efficiency and reliability:

1.  **Motion Detection:** A lightweight, low-power process continuously monitors a low-resolution video stream for any motion.
2.  **Animal Analysis:** When motion is detected, high-resolution frames are passed to a more powerful YOLO-based model to identify specific animal species.
3.  **Data Upload:** If an animal of interest is confirmed, the event (video and metadata) is packaged and handed to a resilient uploader that sends the data to a cloud server.

This approach ensures that the power-hungry GPU and network components are only activated when there is a high probability of a significant biological event, dramatically extending battery life.

## Key Features

-   **Efficient Motion Detection:** Uses adaptive background subtraction to filter out environmental noise like wind and changing shadows.
-   **AI-Powered Species Identification:** Employs a YOLO model to analyze motion events and confirm the presence of target animal species.
-   **Robust Data Packaging:** Confirmed events are saved as a pair: a high-quality MP4 video and a detailed JSON metadata file.
-   **Resilient Data Upload:** A background process reliably uploads data to a remote server using SFTP and records metadata in a PostgreSQL database, automatically retrying on network failure.
-   **Modular Architecture:** The system is built with three independent processes that communicate safely, ensuring stability and responsiveness.

## Live View Feature (Optional)

The system includes an optional, on-demand live viewing feature. To use it, you must first set `enabled: true` in the `live_view` section of your `config.yaml`.

**Data Usage Note:** To prevent accidental high data usage on LTE connections, the live stream is limited by a timer (default 60 seconds). When the timer expires, the stream pauses and must be manually refreshed. You can adjust this limit via `stream_timeout_seconds` in `config.yaml`.

The main `FaunaScope-edge` application must be running for this feature to work.

### Local Viewing (On the Jetson Desktop)

This is useful for direct debugging. It reads the camera feed from the RAM disk.

1.  Start the main application in one terminal: `python main.py`
2.  Open a second terminal and run:
    ```sh
    python scripts/view_live_local.py
    ```
    A window will appear showing the live feed. Press 'q' to close it.

### Remote Viewing (Over the Network)

This runs a web server that streams the feed, allowing you to view it from a web browser on another computer.

1.  Start the main application in one terminal: `python main.py`
2.  Open a second terminal and run the stream server:
    ```sh
    python scripts/stream_server.py
    ```
3.  On any computer on the same network, open a web browser and navigate to `http://<JETSON_IP_ADDRESS>:8080`. For remote access from outside the local network, see documentation on setting up a VPN or a reverse SSH tunnel.

## System Architecture

The application operates as a pipeline of three independent modules communicating through a data queue and the local filesystem.

```mermaid
graph TD
    A[Camera] --> B(Module A: MotionDetector);
    B -- High-res Frames --> C{Queue};
    C --> D(Module B: AnimalAnalyzer);
    D -- Video + JSON --> E["/data/pending_upload/"];
    E --> F(Module C: DataUploader);
    F -- SFTP --> G[Cloud File Server];
    F -- SQL --> H[PostgreSQL Database];
```

## Getting Started

FaunaScope-Edge is primarily intended to run in a Docker container on a CUDA-capable edge device. The Docker setup provides the expected runtime environment, GPU access, camera device mappings, persistent data directories, and the live-view service. See [`DOCKER.md`](DOCKER.md) for the full deployment guide.

### Prerequisites

-   An NVIDIA Jetson device (or a Linux PC with a CUDA-enabled GPU). **Note:** This application is computationally intensive and requires a CUDA-enabled GPU to run effectively. CPU-only execution is not supported due to poor performance.
-   A connected camera (USB webcam or CSI camera)
-   Docker and NVIDIA Container Toolkit installed on the target device
-   Access to a remote server with PostgreSQL and an SFTP user

### Docker Deployment

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/FaunaScope-edge.git
    cd FaunaScope-edge
    ```

2.  **Configure the application:**
    
    FaunaScope-Edge uses a two-layer configuration approach:
    
    -   **Hardware and credentials** (`.env` file):
        ```sh
        cp env.example .env
        nano .env  # Fill in your camera settings, database credentials, etc.
        ```
        The `.env` file is git-ignored and contains deployment-specific settings like camera resolution, database passwords, and device location.
    
    -   **Algorithm parameters** (`config/config.yaml`):
        Edit this file to adjust motion detection thresholds, species of interest, and other algorithm settings. This file is safe to commit to version control.

3.  **Place model weights:**
    Put the trained YOLO model at `model_weight/best.pt`.

4.  **Run with Docker Compose:**
    ```sh
    docker compose -f docker-compose-jetson.yml up --build -d
    ```
    Use `docker-compose-jetson.yml` for Jetson devices and `docker-compose-gpu.yml` for generic NVIDIA GPU machines. See [`DOCKER.md`](DOCKER.md) for platform-specific prerequisites, volume mappings, logs, troubleshooting, and live-view details.

## Usage

For normal operation, run the application through Docker Compose on the target device:

```sh
docker compose -f docker-compose-jetson.yml up -d
```

The container starts the main application and stream server together. To view logs:

```sh
docker compose -f docker-compose-jetson.yml logs -f
```

To stop the application:

```sh
docker compose -f docker-compose-jetson.yml down
```

Running `python main.py` directly is mainly useful for local development or debugging outside the container, and requires manually installing the Python and system dependencies described by the Docker image.

## Project Structure

-   **`main.py`**: The main entry point that initializes and runs all modules.
-   **`config/`**: Contains the `config.yaml` file for all system settings.
-   **`src/`**: Contains the core source code for the application.
    -   `motion_detector/`: Module A for detecting motion.
    -   `animal_analyzer/`: Module B for YOLO analysis and data packaging.
    -   `data_uploader/`: Module C for handling cloud uploads.
-   **`data/`**: Default location for generated data.
    -   `pending_upload/`: Staging area for video/JSON pairs waiting to be uploaded.
    -   `uploaded/`: Archive for successfully uploaded files.
-   **`scripts/`**: A collection of useful helper scripts for testing and debugging individual components.
-   **`tests/`**: Unit tests, run with `pytest` (see Development and Testing).

## Development and Testing

The `scripts/` directory contains tools to help test parts of the system in isolation:
-   `test_camera.py`: Verify that your camera is detected and working correctly.
-   `test_yolo.py`: Run the YOLO model on a test image or video to confirm it's functional.
-   `test_db_connection.py`: Check if the application can connect to your remote PostgreSQL database with the provided credentials.
-   `test_detector.py`: Test motion detection module functionality using a video file.

The unit tests in `tests/` run without a camera, a GPU, or a server. They need `ffmpeg` on the PATH and the Python packages in `requirements-dev.txt`:

```sh
pip install -r requirements-dev.txt
pytest
```
