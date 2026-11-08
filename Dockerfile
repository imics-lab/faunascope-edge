# FaunaScope-Edge Dockerfile
# This Dockerfile supports both Jetson and generic GPU (NVIDIA CUDA) deployments
# For Jetson: use docker-compose-jetson.yml (BASE_IMAGE=dustynv/l4t-ml:r36.2.0)
# For generic GPU: use docker-compose-gpu.yml (BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3)

# -----------------------------------------------------------------------------
# Base Image Selection
# -----------------------------------------------------------------------------
# Default: Generic GPU with NVIDIA PyTorch image
# Override with docker-compose build args for different configurations
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}

# -----------------------------------------------------------------------------
# Build Arguments and Environment Variables
# -----------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------------------------------------------------------
# Install System Dependencies
# -----------------------------------------------------------------------------
# Both base images (dustynv/l4t-ml and nvcr.io/nvidia/pytorch) contain most
# necessary system dependencies. We install ffmpeg and GStreamer plugins
# for video processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# Install gosu, a lightweight tool for dropping root privileges
RUN apt-get update && apt-get install -y --no-install-recommends gosu xvfb && rm -rf /var/lib/apt/lists/*

# Create a symbolic link so 'python' can be used to run 'python3' (if not already present)
RUN ln -sf /usr/bin/python3 /usr/bin/python || true

# -----------------------------------------------------------------------------
# Create Application User (security best practice)
# -----------------------------------------------------------------------------
# Create the user and add them to the 'video' group to allow camera access
RUN groupadd -r FaunaScope \
    && (getent group render >/dev/null || groupadd -r render) \
    && useradd -r -g FaunaScope -G video,render -m FaunaScope

RUN mkdir -p /home/FaunaScope/.config
RUN chown -R FaunaScope:FaunaScope /home/FaunaScope

# -----------------------------------------------------------------------------
# Set Working Directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Install Python Dependencies
# -----------------------------------------------------------------------------
# Both base images (dustynv/l4t-ml and nvcr.io/nvidia/pytorch) come with
# torch and torchvision pre-installed. We install ultralytics separately
# to ensure it uses the system-provided torch.
# --no-dependencies prevents pip from replacing the pre-compiled packages
# in the base image (like numpy, torch, etc.)
RUN python -m pip install --index-url https://pypi.org/simple --no-dependencies ultralytics
RUN python -m pip install --index-url https://pypi.org/simple matplotlib

# Networking
RUN python -m pip install --index-url https://pypi.org/simple requests
# Utilities
RUN python -m pip install --index-url https://pypi.org/simple PyYAML

# Web server for the optional remote live stream
RUN python -m pip install --index-url https://pypi.org/simple Flask==2.2.5

# For EXIF timestamping in live view
RUN python -m pip install --index-url https://pypi.org/simple piexif

# -----------------------------------------------------------------------------
# Copy Application Code
# -----------------------------------------------------------------------------
COPY main.py .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docker-entrypoint.sh .

# -----------------------------------------------------------------------------
# Make Entrypoint Script Executable
# -----------------------------------------------------------------------------
RUN chmod +x /app/docker-entrypoint.sh

# -----------------------------------------------------------------------------
# Create Required Directories
# -----------------------------------------------------------------------------
RUN mkdir -p \
    /app/config \
    /app/data/pending_upload \
    /app/data/uploaded \
    /app/model_weight \
    /app/logs \
    /tmp/FaunaScope_edge_temp \
    && chown -R FaunaScope:FaunaScope /app /tmp/FaunaScope_edge_temp

# -----------------------------------------------------------------------------
# Set Ownership
# -----------------------------------------------------------------------------
# Change ownership of all application files to the FaunaScope user
RUN chown -R FaunaScope:FaunaScope /app

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
# Simple health check to verify the container is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep -f "python.*main.py" || exit 1

# -----------------------------------------------------------------------------
# Expose Ports (if using live view web server)
# -----------------------------------------------------------------------------
# Port 8080 is used by the optional stream_server.py for remote viewing
EXPOSE 8080

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
# Default command runs the main application
# Can be overridden in docker-compose.yml or with docker run
CMD ["python", "main.py"]
