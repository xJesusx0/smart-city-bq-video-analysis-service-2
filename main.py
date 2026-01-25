import cv2
from ultralytics import YOLO


yolo_model = 'yolo26n.pt'

model = YOLO(yolo_model)
print(yolo_model)
