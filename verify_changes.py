import cv2
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.getcwd())

from video_analysis.core.inference import SequentialDualYOLO
from video_analysis.utils.model_loader import get_models
from video_analysis.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verification")

def verify():
    # Load Models
    print("Loading models...")
    models = get_models()
    
    # Initialize implementation
    print("Initializing SequentialDualYOLO...")
    dual_yolo = SequentialDualYOLO(
        model_accidents=models['accidents'],
        model_coco=models['coco'],
        accident_confidence=config.inference.get("accident_confidence"),
        object_confidence=config.inference.get("object_confidence"),
        correlation_distance=config.inference.get("correlation_distance")
    )

    # Open video
    video_path = 'videos/moiz3.mp4'
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video")
        return

    print("Processing 10 frames...")
    try:
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = dual_yolo.process_frame(frame)
            
            # Check results structure
            accidents = results['accidents']
            print(f"Frame {i}: Found {len(accidents)} confirmed accidents")
            
    except Exception as e:
        print(f"FAILED with error: {e}")
        raise e
    finally:
        cap.release()
        print("Verification finished.")

if __name__ == "__main__":
    verify()
