import cv2
import time
import yaml
import argparse
import sys
from multiprocessing import Process, Queue
import os

# Ensure that the project root is on sys.path, no matter where we run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
from src.animal_analyzer.analyzer import AnimalAnalyzer

def load_config(config_path="config/config.yaml"):
    """Loads the YAML configuration file from the project root."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        print("Please ensure you are running this script from the project's root directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def simulate_motion_detector(video_path: str, frame_queue: Queue):
    """
    Reads a video file and pushes its frames to a queue, simulating the
    output of the MotionDetector module during a single, continuous event.

    Args:
        video_path (str): The path to the input video file.
        frame_queue (Queue): The queue to put frames into.
    """
    print(f"[Simulator] Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Simulator] Error: Could not open video file.")
        frame_queue.put(None) # Signal end even on error
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break
        
        # Put the frame onto the queue for the analyzer to process
        frame_queue.put(frame)
        frame_count += 1
        print(f"[Simulator] Fed frame #{frame_count}", end='\r')
        
        # Optional: Uncomment to simulate a real-time frame rate (e.g., 15 FPS)
        time.sleep(1/15)

    # After the video is finished, send the end-of-event signal
    frame_queue.put(None)
    print(f"\n[Simulator] Finished feeding {frame_count} frames. Sent end-of-event signal.")
    cap.release()

def main():
    """
    Main function to set up and run the test.
    """
    parser = argparse.ArgumentParser(
        description="Test the AnimalAnalyzer module with a video file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help="Path to the input video file to be processed."
    )
    parser.add_argument(
        '--config',
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file (default: config/config.yaml)."
    )
    args = parser.parse_args()

    # --- Setup ---
    print("--- Starting AnimalAnalyzer Test Harness ---")
    config = load_config(args.config)
    frame_queue = Queue()

    # --- Initialize and Start Analyzer ---
    # The analyzer will start and immediately block, waiting for frames on the queue.
    analyzer = AnimalAnalyzer(frame_queue, config)
    analyzer_process = Process(target=analyzer.start, name="AnimalAnalyzerProcess")
    
    analyzer_process.start()
    print("[Harness] AnimalAnalyzer process started. Now simulating MotionDetector...")
    
    # --- Start the Simulation ---
    # The main process will now act as the motion detector, feeding the queue.
    try:
        simulate_motion_detector(args.video, frame_queue)

        # --- Wait for Analyzer to Finish ---
        # The join() method will wait here until the analyzer process has
        # finished all its work (which it will after receiving the 'None' signal).
        print("[Harness] Waiting for AnimalAnalyzer to finish processing...")
        analyzer_process.join()
        print("[Harness] AnimalAnalyzer process has finished.")

    except KeyboardInterrupt:
        print("\n[Harness] Interruption detected. Shutting down...")
        if analyzer_process.is_alive():
            analyzer_process.terminate()
            analyzer_process.join()
    
    finally:
        if analyzer_process.is_alive():
            analyzer_process.terminate()
        print("--- Test Harness Finished ---")

# Run as: python scripts/test_analyzer.py --video video_file.mp4
if __name__ == "__main__":
    main()