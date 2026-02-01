import cv2
from ultralytics import YOLO

def resize_frame(frame):
    target_width = 1280
    h, w, _ = frame.shape
    scale = target_width / w
    target_height = int(h * scale)

    frame = cv2.resize(frame, (target_width, target_height))
    return frame

def get_models():
    """
    Load both YOLO models for dual inference.
    
    Returns:
        dict: Dictionary containing both models:
            - 'accidents': YOLO model for accident detection
            - 'coco': YOLO model for object detection
    """
    model_accidents = YOLO('models/yolov8m-accidents.pt')
    model_coco = YOLO('models/yolov8n.pt')
    
    return {
        'accidents': model_accidents,
        'coco': model_coco
    }
