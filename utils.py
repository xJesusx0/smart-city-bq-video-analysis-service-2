import cv2
from ultralytics import YOLO

def resize_frame(frame):
    target_width = 1280
    h, w, _ = frame.shape
    scale = target_width / w
    target_height = int(h * scale)

    frame = cv2.resize(frame, (target_width, target_height))
    return frame

def get_model():
    yolo_model = 'yolo26n.pt'

    model = YOLO(yolo_model)
    model.task = 'detect'
    model.conf = 0.8
    return model
