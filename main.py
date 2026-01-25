import cv2
from ultralytics import YOLO
import utils

classes = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign"    
}

classes_id = [ key for key, value in classes.items() ]

model = utils.get_model()
cap = cv2.VideoCapture('videos/salida_720p.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = utils.resize_frame(frame)
    if not ret:
        break

    results = model(frame, stream=True, classes = classes_id)
    for result in results: 
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls] 

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    cv2.imshow("YOLO Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

