import cv2
from ultralytics import YOLO
import numpy as np
import utils

# ---------------------------
# Clases de interés
# ---------------------------
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

classes_id = [key for key in classes.keys()]

# ---------------------------
# ROI (zona de interés)
# Ajusta estos puntos a tu video
# ---------------------------
roi_points = [
    (0, 0),
    (500, 0),
    (500, 500),
    (0, 500),
]

roi_pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))

# ---------------------------
# Modelo y video
# ---------------------------
model = utils.get_model()
cap = cv2.VideoCapture('videos/salida_720p.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = utils.resize_frame(frame)

    # Dibujar ROI
    cv2.polylines(frame, [roi_pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Inferencia YOLO
    results = model(frame, stream=True, classes=classes_id)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Centro del bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ¿Está dentro del ROI?
            inside = cv2.pointPolygonTest(roi_pts, (cx, cy), False)

            if inside < 0:
                continue  # Ignorar objetos fuera del ROI

            # Dibujar detección válida
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            cv2.putText(
                frame,
                f'{label} {conf:.2f}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
