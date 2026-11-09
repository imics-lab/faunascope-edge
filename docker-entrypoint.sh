#!/bin/bash
# FaunaScope-Edge Docker Entrypoint Script
# This script starts both the main application and the stream server

set -e

echo "==================================================================="
echo "FaunaScope-Edge Docker Container Starting"
echo "==================================================================="

# Start a virtual X server in the background to satisfy EGL requirements.
# The resolution and color depth don't matter, but are required syntax.
# Clean up any stale lock files to ensure Xvfb can start after a crash.
rm -f /tmp/.X0-lock
export DISPLAY=:0
Xvfb :0 -screen 0 1280x720x24 &
XVFB_PID=$!
echo "Virtual X server (Xvfb) started with PID: $XVFB_PID"

# Function to handle shutdown gracefully
shutdown() {
    echo ""
    echo "==================================================================="
    echo "Shutdown signal received. Stopping all processes..."
    echo "==================================================================="

    # Kill the stream server if it's running
    if [ -n "$STREAM_PID" ]; then
        echo "Stopping stream server (PID: $STREAM_PID)..."
        kill -TERM "$STREAM_PID" 2>/dev/null || true
        wait "$STREAM_PID" 2>/dev/null || true
    fi

    # Kill the main application if it's running
    if [ -n "$MAIN_PID" ]; then
        echo "Stopping main application (PID: $MAIN_PID)..."
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        wait "$MAIN_PID" 2>/dev/null || true
    fi
    
    # Kill the virtual X server
    if [ -n "$XVFB_PID" ]; then
        echo "Stopping virtual X server (PID: $XVFB_PID)..."
        kill -TERM "$XVFB_PID" 2>/dev/null || true
    fi

    echo "All processes stopped. Goodbye!"
    exit 0
}

# Trap SIGTERM and SIGINT signals and call shutdown function
trap shutdown SIGTERM SIGINT

# Check if config file exists
if [ ! -f "/app/config/config.yaml" ]; then
    echo "ERROR: Configuration file not found at /app/config/config.yaml"
    echo "Please mount your config file as a volume."
    exit 1
fi

# Check if .env file exists
if [ ! -f "/app/.env" ]; then
    echo "ERROR: Environment file not found at /app/.env"
    echo "Please create .env file from env.example and mount it as a volume."
    echo "Run: cp env.example .env"
    exit 1
fi

# Check if model weights exist
if [ ! -f "/app/model_weight/MDV6-mit-yolov9-c.onnx" ]; then
    echo "WARNING: Model weights not found at /app/model_weight/MDV6-mit-yolov9-c.onnx"
    echo "The application may fail if the MegaDetector model is not available."
fi
if [ ! -f "/app/model_weight/speciesnet/speciesnet.onnx" ]; then
    echo "WARNING: SpeciesNet not found in /app/model_weight/speciesnet/ (see docs/deployment.md, Model weights)"
fi

# Fix permissions on mounted volumes (must run as root)
echo ""
echo "Fixing permissions on mounted volumes..."
chown -R FaunaScope:FaunaScope /app/logs /app/data /tmp/FaunaScope_edge_temp 2>/dev/null || true

# Give the application user access to host-mounted Jetson device nodes.
for DEVICE_PATH in /dev/nvhost* /dev/nvmap /dev/nvidia* /dev/dri/* /dev/video* /dev/media*; do
    if [ -e "$DEVICE_PATH" ]; then
        DEVICE_GID=$(stat -c '%g' "$DEVICE_PATH")
        DEVICE_GROUP=$(getent group "$DEVICE_GID" | cut -d: -f1)
        if [ -z "$DEVICE_GROUP" ]; then
            DEVICE_GROUP="hostdev_$DEVICE_GID"
            groupadd -g "$DEVICE_GID" "$DEVICE_GROUP" 2>/dev/null || true
        fi
        usermod -aG "$DEVICE_GROUP" FaunaScope 2>/dev/null || true
    fi
done

echo ""
echo "Starting FaunaScope-Edge Main Application..."
echo "-------------------------------------------------------------------"

# Start the main application in the background as FaunaScope user
gosu FaunaScope python main.py &
MAIN_PID=$!
echo "Main application started with PID: $MAIN_PID"

# Give the main app a moment to initialize
sleep 3

echo ""
echo "Starting Stream Server..."
echo "-------------------------------------------------------------------"

# Start the stream server in the background as FaunaScope user
gosu FaunaScope python scripts/stream_server.py &
STREAM_PID=$!
echo "Stream server started with PID: $STREAM_PID"

echo ""
echo "==================================================================="
echo "FaunaScope-Edge is now running!"
echo "==================================================================="
echo "Main Application PID: $MAIN_PID"
echo "Stream Server PID:    $STREAM_PID"
echo "Live View URL:        http://localhost:8080"
echo ""
echo "To view logs: docker compose logs -f"
echo "To stop:      docker compose down"
echo "==================================================================="
echo ""

# Wait for both processes
# This keeps the container running and allows for graceful shutdown
wait -n

# If we get here, one of the processes has exited
EXIT_CODE=$?

echo ""
echo "==================================================================="
echo "A process has exited with code: $EXIT_CODE"
echo "==================================================================="

# Check which process exited
if ! kill -0 "$MAIN_PID" 2>/dev/null; then
    echo "Main application has stopped."
fi

if ! kill -0 "$STREAM_PID" 2>/dev/null; then
    echo "Stream server has stopped."
fi

# Trigger shutdown of remaining processes
shutdown
