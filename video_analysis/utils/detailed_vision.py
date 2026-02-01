import cv2
import numpy as np

def resize_frame(frame: np.ndarray, target_width: int = 640, fast_mode: bool = True) -> np.ndarray:
    """
    Optimized frame resizing for better FPS.
    """
    h, w = frame.shape[:2]
    
    # Skip resize if already at target width
    if w == target_width:
        return frame
    
    # Calculate target height maintaining aspect ratio
    scale = target_width / w
    target_height = int(h * scale)
    
    # Choose interpolation method based on mode
    interpolation = cv2.INTER_NEAREST if fast_mode else cv2.INTER_LINEAR
    
    frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
    return frame
