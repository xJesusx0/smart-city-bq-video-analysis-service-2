"""
Test script for Sequential Dual-YOLO System

This script tests the sequential inference system without requiring a camera.
It creates a synthetic test frame and verifies the system works correctly.
"""

import cv2
import numpy as np
from sequential_inference import SequentialDualYOLO
import utils

def create_test_frame():
    """Create a synthetic test frame"""
    # Create a blank frame (1280x720, BGR)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), -1)  # Blue
    cv2.rectangle(frame, (400, 200), (600, 400), (0, 255, 0), -1)  # Green
    cv2.rectangle(frame, (800, 300), (1000, 500), (0, 0, 255), -1)  # Red
    
    # Add some text
    cv2.putText(frame, "Test Frame", (500, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def test_sequential_inference():
    """Test the sequential dual-YOLO system"""
    print("=" * 60)
    print("Testing Sequential Dual-YOLO System")
    print("=" * 60)
    
    # Load models
    print("\n[1/4] Loading models...")
    try:
        models = utils.get_models()
        print("✓ Models loaded successfully")
        print(f"  - Accidents model: {type(models['accidents'])}")
        print(f"  - COCO model: {type(models['coco'])}")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False
    
    # Initialize system
    print("\n[2/4] Initializing Sequential Dual-YOLO system...")
    try:
        dual_yolo = SequentialDualYOLO(
            model_accidents=models['accidents'],
            model_coco=models['coco'],
            accident_confidence=0.5,
            object_confidence=0.4,
            correlation_distance=100.0
        )
        print("✓ System initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing system: {e}")
        return False
    
    # Create test frame
    print("\n[3/4] Creating test frame...")
    frame = create_test_frame()
    print(f"✓ Test frame created: {frame.shape}")
    
    # Process frame
    print("\n[4/4] Processing frame with dual-YOLO...")
    try:
        results = dual_yolo.process_frame(frame)
        print("✓ Frame processed successfully")
        
        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        print(f"\nAccidents detected: {len(results['accidents'])}")
        for i, accident in enumerate(results['accidents']):
            print(f"  Accident {i+1}:")
            print(f"    - Confidence: {accident['confidence']:.2f}")
            print(f"    - BBox: {accident['bbox']}")
            print(f"    - Involved objects: {len(accident['involved_objects'])}")
        
        print(f"\nObjects detected: {len(results['objects'])}")
        object_counts = {}
        for obj in results['objects']:
            cls = obj['class']
            object_counts[cls] = object_counts.get(cls, 0) + 1
        
        for cls, count in object_counts.items():
            print(f"  - {cls}: {count}")
        
        print(f"\nPerformance:")
        print(f"  - FPS: {results['fps_info']['fps']:.2f}")
        print(f"  - Total latency: {results['fps_info']['total_time']*1000:.2f}ms")
        print(f"  - Accidents inference: {results['fps_info']['accidents_inference_time']*1000:.2f}ms")
        print(f"  - COCO inference: {results['fps_info']['coco_inference_time']*1000:.2f}ms")
        print(f"  - Post-processing: {results['fps_info']['postprocess_time']*1000:.2f}ms")
        
        print("\n" + "=" * 60)
        print("✓ TEST PASSED")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sequential_inference()
    exit(0 if success else 1)
