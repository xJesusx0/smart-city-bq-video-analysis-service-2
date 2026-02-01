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
    model = YOLO('models/best-versi-1.pt')
    return model
