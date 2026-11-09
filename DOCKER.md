# FaunaScope-Edge Docker Deployment Guide

This guide explains how to build and run FaunaScope-Edge using Docker Compose on systems with NVIDIA GPU support.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Prerequisites](#prerequisites)
- [Quick Start - Generic GPU](#quick-start---generic-gpu)
- [Quick Start - Jetson Devices](#quick-start---jetson-devices)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Deployment Options

FaunaScope-Edge provides two Docker Compose configurations to support different hardware platforms:

### 1. **docker-compose-gpu.yml** - Generic GPU Machines
- **Use case:** Desktop/server machines with NVIDIA GPUs (RTX, Tesla, Pascal/Volta/Turing/Ampere)
- **Base image:** `nvcr.io/nvidia/pytorch:22.12-py3` (PyTorch 1.13.1, CUDA 11.x, cuDNN 8)
- **Requirements:** NVIDIA driver supporting CUDA 11.x (R520+)
- **Optimizations:** Standard CUDA memory management, larger shared memory (2GB)
- **Camera support:** Standard USB cameras via V4L2
- **Video acceleration:** CPU-based decoding (software fallback)

### 2. **docker-compose-jetson.yml** - NVIDIA Jetson Devices
- **Use case:** NVIDIA Jetson Nano, Jetson Orin, and other Jetson family devices
- **Base image:** `dustynv/l4t-ml:r36.4.0` (JetPack-optimized with L4T)
- **Requirements:** JetPack SDK installed on host
- **Optimizations:** `cudaMallocAsync` backend for Jetson memory management, privileged mode for V4L2
- **Camera support:** USB cameras and CSI cameras with full media controller access
- **Video acceleration:** Hardware-accelerated format conversion via nvvidconv

**Which one should I use?**
- Use `docker-compose-gpu.yml` for x86_64 machines with standard NVIDIA GPUs
- Use `docker-compose-jetson.yml` for ARM64-based Jetson devices

## Prerequisites

### Common Requirements (All Platforms)

1.  **Docker** (version 20.10 or later)
    ```bash
    # Install Docker on Ubuntu/Debian
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    # Add your user to docker group (logout/login required)
    sudo usermod -aG docker $USER
    ```

2.  **NVIDIA Container Toolkit** (Required for GPU access in Docker)
    
    This is **required** for both generic GPU machines and Jetson devices to allow Docker containers to access the GPU.
    
    ```bash
    # Add the GPG key
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    
    # Add the repository (works for Ubuntu/Debian)
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Update and install
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    
    # Restart Docker
    sudo systemctl restart docker
    
    # Verify installation (should show your GPU)
    docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
    ```

### Platform-Specific Requirements

**For Generic GPU Machines:**
- NVIDIA GPU driver supporting CUDA 11.x
- x86_64 architecture

**For Jetson Devices:**
- JetPack SDK installed on host
- ARM64 architecture

## Quick Start - Generic GPU

For desktop/server machines with standard NVIDIA GPUs (RTX, Tesla, A100, etc.):

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/your-username/FaunaScope-edge.git
    cd FaunaScope-edge
    ```

2.  **Prepare configuration**:
    Ensure `config/config.yaml` exists and is properly configured for your camera and device settings. See the main `README.md` for more details.

3.  **Ensure model weights are present**:
    The detector, `model_weight/MDV6-mit-yolov9-c.onnx`, comes with the repository; download the species classifier into `model_weight/speciesnet/` as described in the main `README.md`.

4.  **Build and run**:
    ```bash
    docker compose -f docker-compose-gpu.yml up --build -d
    ```

5.  **View logs**:
    ```bash
    docker compose -f docker-compose-gpu.yml logs -f
    ```
6. **Clean logs**:
    ```bash
    sudo truncate -s 0 $(docker inspect --format='{{.LogPath}}' FaunaScope-edge-gpu)
    ```

7.  **Access Live View**:
    Once the container is running, the live view is available at `http://<YOUR_DEVICE_IP>:8080`.

8.  **Stop the application**:
    ```bash
    docker compose -f docker-compose-gpu.yml down
    ```

## Quick Start - Jetson Devices

For NVIDIA Jetson devices (Jetson Nano, Jetson Orin, etc.):

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/your-username/FaunaScope-edge.git
    cd FaunaScope-edge
    ```

2.  **Prepare configuration**:
    Ensure `config/config.yaml` exists and is properly configured for your camera and device settings. See the main `README.md` for more details.

3.  **Ensure model weights are present**:
    The detector, `model_weight/MDV6-mit-yolov9-c.onnx`, comes with the repository; download the species classifier into `model_weight/speciesnet/` as described in the main `README.md`.

4.  **Build and run**:
    ```bash
    docker compose -f docker-compose-jetson.yml up --build -d
    ```

5.  **View logs**:
    ```bash
    docker compose -f docker-compose-jetson.yml logs -f
    ```
6. **Clean logs**:
    ```bash
    sudo truncate -s 0 $(docker inspect --format='{{.LogPath}}' FaunaScope-edge-gpu)
    ```
7.  **Access Live View**:
    Once the container is running, the live view is available at `http://<YOUR_JETSON_IP>:8080`.

8.  **Stop the application**:
    ```bash
    docker compose -f docker-compose-jetson.yml down
    ```

## Configuration

### Docker Compose
If you need to modify the Docker setup (e.g., change the camera device from `/dev/video0`), edit the appropriate compose file:
- `docker-compose-gpu.yml` for generic GPU machines
- `docker-compose-jetson.yml` for Jetson devices

### Application
All application-level settings (camera resolution, model confidence, uploader settings, etc.) are in `config/config.yaml`.

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA runtime is installed on the host
docker info | grep -i runtime

# Test GPU access inside the running container
docker exec -it FaunaScope-edge-gpu nvidia-smi
```

### Camera Not Found
```bash
# Verify camera is connected on the host
ls -la /dev/video*

# Ensure the correct device is mapped in your docker-compose file
# (docker-compose-gpu.yml or docker-compose-jetson.yml)
```

### Start an interactive shell inside the container
```bash
docker exec -it FaunaScope-edge-gpu /bin/bash
```

### Switching Between Configurations
If you need to switch from one configuration to another (e.g., from Jetson to generic GPU):

1. Stop the current container:
   ```bash
   docker compose -f docker-compose-jetson.yml down
   ```

2. Build and start with the new configuration:
   ```bash
   docker compose -f docker-compose-gpu.yml up --build -d
   ```

Note: The two configurations use the same container name, so they cannot run simultaneously.