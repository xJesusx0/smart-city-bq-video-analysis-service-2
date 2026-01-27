import cv2
from ultralytics import YOLO
import numpy as np
import utils
from services.exporter import ConsoleExportService
from services.reporter import ReportManager

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
}

classes_id = [key for key in classes.keys()]

# ---------------------------
# Configuración de ROIs
# ---------------------------
# Definimos dos zonas con colores distintos
rois = [
    {
        "name": "Zona A (Arriba-Izq)",
        "points": np.array([
            (0, 0),
            (600, 0),
            (600, 600),
            (0, 600)
        ], np.int32),
        "color": (0, 0, 255), # Rojo
        "counts": {}
    },
    {
        "name": "Zona B (Abajo-Der)",
        "points": np.array([
            (680, 360),
            (1280, 360),
            (1280, 720),
            (680, 720)
        ], np.int32),
        "color": (255, 0, 0), # Azul
        "counts": {}
    }
]

# ---------------------------
# Modelo y video
# ---------------------------
model = utils.get_model()
cap = cv2.VideoCapture('videos/salida_720p.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# ---------------------------
# Servicios de Reporte
# ---------------------------
exporter = ConsoleExportService()
reporter = ReportManager(exporter, interval_seconds=5.0)

current_video_time = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calcular tiempo actual del video
    current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    frame = utils.resize_frame(frame)

    # Reiniciar contadores para este frame
    for roi in rois:
        roi["counts"] = {name: 0 for name in classes.values()}

    # Inferencia YOLO
    results = model(frame, stream=True, classes=classes_id, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = classes.get(cls, model.names[cls])

            # Centro del bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Verificar en qué ROI está
            for roi in rois:
                inside = cv2.pointPolygonTest(roi["points"], (cx, cy), False)
                if inside >= 0:
                    roi["counts"][label] += 1
                    
                    # Dibujar bounding box del color de la ROI
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, roi["color"], -1)
                    cv2.putText(
                        frame,
                        f'{label} {conf:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

    # ---------------------------
    # Reporte Dinámico
    # ---------------------------
    # Preparamos los datos para el reporte
    report_data = {
        "timestamp": current_video_time,
        "rois": {roi["name"]: roi["counts"].copy() for roi in rois}
    }
    reporter.update(current_video_time, report_data)


    # ---------------------------
    # Visualización
    # ---------------------------
    overlay = frame.copy()
    
    # Dibujar Polígonos y Panel de Estadísticas
    ui_y_offset = 30
    
    for roi in rois:
        # Dibujar polígono
        cv2.polylines(frame, [roi["points"]], isClosed=True, color=roi["color"], thickness=2)
        
        # Dibujar fondo semitransparente para stats
        cv2.rectangle(overlay, (10, ui_y_offset - 25), (300, ui_y_offset + 120), (0, 0, 0), -1)
        
        # Título de la zona
        cv2.putText(frame, f"{roi['name']}:", (20, ui_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi["color"], 2)
        ui_y_offset += 25
        
        # Contadores por tipo
        for cls_name, count in roi["counts"].items():
            if count > 0:
                text = f"  {cls_name.capitalize()}: {count}"
                cv2.putText(frame, text, (20, ui_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                ui_y_offset += 25
        
        ui_y_offset += 10 # Espacio extra entre zonas

    # Aplicar transparencia al panel
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.imshow("Multi-Zone Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
