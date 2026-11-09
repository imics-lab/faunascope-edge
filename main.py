# main.py

import yaml
import time
import multiprocessing
from multiprocessing import Process, Queue
import sys
import os
import argparse

# Import the main classes from the source directory
from src.motion_detector.detector import MotionDetector
from src.animal_analyzer.analyzer import AnimalAnalyzer
from src.data_uploader.uploader import DataUploader
from src.device_info import StatusCounters
from src.heartbeat import Heartbeat

def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file with environment variable substitution.
    
    Supports ${VAR_NAME} syntax in config.yaml. Values are loaded from .env file
    if present, allowing sensitive credentials to be kept out of version control.
    
    Args:
        config_path (str): The path to the configuration file.
        
    Returns:
        dict: The configuration dictionary with environment variables substituted.
    """
    # Load .env file if it exists
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    try:
        with open(config_path, 'r') as f:
            config_text = f.read()
            # Expand ${VAR} and $VAR to their values from environment variables
            config_text = os.path.expandvars(config_text)
            return yaml.safe_load(config_text)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def main():
    """
    The main entry point for the FaunaScope-Edge application.
    """
    parser = argparse.ArgumentParser(description="Run the FaunaScope-Edge application.")
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help="Path to a video file to use as input. Overrides config file setting."
    )
    args = parser.parse_args()

    print("--- FaunaScope-Edge Application Starting ---")

    # 1. Load configuration from the YAML file
    config = load_config()

    # 2. Determine video source: command-line arg overrides config
    if args.video is not None:
        video_source = args.video
    else:
        video_source = config.get('motion_detector', {}).get('video_source', 0)
    
    print(f"Using video source: {video_source}")

    # 3. Create the shared queue for communication between the
    #    MotionDetector and the AnimalAnalyzer.
    frame_queue = multiprocessing.Queue()
    # Counters the modules share with the heartbeat, which this process writes once an hour.
    status = StatusCounters()

    # 4. Instantiate the main module classes
    try:
        motion_detector = MotionDetector(config, video_source=video_source, status=status)
        animal_analyzer = AnimalAnalyzer(frame_queue, config, status)
        heartbeat = Heartbeat(config, status)
    except KeyError as e:
        print(f"Error: A required key is missing from the configuration file: {e}")
        sys.exit(1)

    # 5. Create a process for each module.
    #    The 'target' is the 'start' method of each class instance.
    analyzer_process = Process(target=animal_analyzer.start)
    processes = [
        Process(target=motion_detector.start, args=(frame_queue,)),
        analyzer_process,
    ]

    uploader_enabled = config.get('uploader', {}).get('enabled', True)
    if uploader_enabled:
        data_uploader = DataUploader(config, status)
        processes.append(Process(target=data_uploader.start))
    else:
        print("Data uploader is disabled. Skipping uploader process.")

    try:
        # 6. Start all processes
        print("Starting all modules in separate processes...")
        for p in processes:
            p.start()
        
        # Monitor the child processes and initiate a clean shutdown if any of them
        # terminate unexpectedly. This prevents zombie processes.
        while True:
            time.sleep(5)
            heartbeat.tick(analyzer_process.is_alive())
            alive_processes = [p for p in processes if p.is_alive()]
            if len(alive_processes) < len(processes):
                print("\n--- A subprocess has terminated unexpectedly ---")
                for p in processes:
                    if not p.is_alive():
                        print(f"Process {p.name} (PID: {p.pid}) exited with code: {p.exitcode}")
                break # Exit the loop to trigger the finally block for shutdown.

    except KeyboardInterrupt:
        print("\n--- Shutdown signal received (Ctrl+C) ---")
    
    finally:
        # 8. Graceful shutdown procedure
        print("Initiating graceful shutdown of all modules...")

        # The order of shutdown can matter. It's often best to stop
        # the producer (MotionDetector) first. However, a forceful
        # termination is simpler and generally effective here.
        for p in processes:
            print(f"Terminating process: {p.name} (PID: {p.pid})")
            if p.is_alive():
                p.terminate()  # Sends a SIGTERM signal
                p.join(timeout=5)  # Wait for the process to exit
        
        print("All processes have been terminated.")
        # The heartbeat of the period so far, so its operating time is not lost.
        print(f"Wrote {heartbeat.write()}")
        print("--- FaunaScope-Edge Application Shut Down ---")

if __name__ == "__main__":
    # Set the start method to 'spawn' right at the beginning.
    # This is crucial for CUDA compatibility in child processes on platforms
    # like the Jetson Nano, and must be done before any multiprocessing
    # objects (like Queues or Processes) are instantiated.
    multiprocessing.set_start_method('spawn', force=True)
    main()