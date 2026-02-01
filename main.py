import cv2
from ultralytics import YOLO
import numpy as np
import utils
from services.exporter import ConsoleExportService
from services.reporter import ReportManager
from sequential_inference import SequentialDualYOLO

# ---------------------------
# Clases de interés
# ---------------------------
# Clases para modelo COCO (objetos)
coco_classes = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# ---------------------------
# Configuración de ROIs (Basadas en Porcentajes)
# ---------------------------
# Las ROIs ahora se definen con coordenadas porcentuales (0.0-1.0)
# Esto las hace independientes de la resolución del video

# Convertir coordenadas originales (1280x720) a porcentajes:
# Zona A: (0, 320) → (0%, 44.4%), (600, 320) → (46.9%, 44.4%)
# Zona B: (680, 360) → (53.1%, 50%), (1280, 360) → (100%, 50%)

rois = [
    utils.create_roi_from_percentage(
        name="Zona A (Arriba-Izq)",
        percentage_points=[
            (0.0, 0.44),    # Top-left: 0%, 44%
            (0.47, 0.44),   # Top-right: 47%, 44%
            (0.47, 1.0),    # Bottom-right: 47%, 100%
            (0.0, 1.0)      # Bottom-left: 0%, 100%
        ],
        color=(0, 0, 255)  # Rojo
    ),
    utils.create_roi_from_percentage(
        name="Zona B (Abajo-Der)",
        percentage_points=[
            (0.53, 0.50),   # Top-left: 53%, 50%
            (1.0, 0.50),    # Top-right: 100%, 50%
            (1.0, 1.0),     # Bottom-right: 100%, 100%
            (0.53, 1.0)     # Bottom-left: 53%, 100%
        ],
        color=(255, 0, 0)  # Azul
    )
]

# ---------------------------
# Modelos y Sistema Dual
# ---------------------------
models = utils.get_models()
dual_yolo = SequentialDualYOLO(
    model_accidents=models['accidents'],
    model_coco=models['coco'],
    accident_confidence=0.2,
    object_confidence=0.4,
    correlation_distance=100.0
)

# ---------------------------
# Video Source
# ---------------------------
#cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap = cv2.VideoCapture('videos/moiz3.mp4')
# ---------------------------
# Servicios de Reporte
# ---------------------------
exporter = ConsoleExportService()
reporter = ReportManager(exporter, interval_seconds=5.0)

# ---------------------------
# Frame Skipping Configuration
# ---------------------------
# Procesar solo 1 de cada N frames para aumentar FPS
# PROCESS_EVERY_N_FRAMES = 1: Sin skipping (procesa todos)
# PROCESS_EVERY_N_FRAMES = 2: Procesa 1 de cada 2 (2x FPS)
# PROCESS_EVERY_N_FRAMES = 3: Procesa 1 de cada 3 (3x FPS)
PROCESS_EVERY_N_FRAMES = 2

frame_count = 0
last_results = None
current_video_time = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calcular tiempo actual del video
    current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    frame = utils.resize_frame(frame)
    
    # Actualizar coordenadas de ROIs según el tamaño del frame actual
    frame_height, frame_width = frame.shape[:2]
    utils.update_roi_pixels(rois, frame_width, frame_height)
    
    frame_count += 1
    
    # ---------------------------
    # Frame Skipping Logic
    # ---------------------------
    # Solo procesar inferencia cada N frames
    should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)
    
    if should_process:
        # Reiniciar contadores para este frame
        for roi in rois:
            roi["counts"] = {name: 0 for name in coco_classes.values()}
            roi["counts"]["accident"] = 0

        # Inferencia Dual YOLO (Secuencial) - Solo en frames seleccionados
        results = dual_yolo.process_frame(frame, rois=rois)
        last_results = results  # Guardar para frames saltados
        
        # Actualizar contadores de ROI desde los resultados
        for roi in rois:
            roi_name = roi["name"]
            if roi_name in results['roi_stats']:
                roi["counts"] = results['roi_stats'][roi_name]
    else:
        # Frame saltado: reutilizar resultados anteriores
        if last_results is not None:
            results = last_results
            # Actualizar contadores de ROI desde los resultados guardados
            for roi in rois:
                roi_name = roi["name"]
                if roi_name in results['roi_stats']:
                    roi["counts"] = results['roi_stats'][roi_name]
        else:
            # Primer frame, procesar de todas formas
            for roi in rois:
                roi["counts"] = {name: 0 for name in coco_classes.values()}
                roi["counts"]["accident"] = 0
            results = dual_yolo.process_frame(frame, rois=rois)
            last_results = results

    # ---------------------------
    # Visualización de Detecciones
    # ---------------------------
    
    # Dibujar objetos detectados (solo los que están dentro de ROIs)
    for obj in results['objects']:
        # Solo dibujar si el objeto está dentro de una ROI
        if 'roi' not in obj:
            continue
            
        x1, y1, x2, y2 = obj['bbox']
        conf = obj['confidence']
        label = obj['class']
        
        # Color según si está involucrado en accidente
        if obj['involved_in_accident']:
            color = (0, 255, 255)  # Amarillo para objetos involucrados
            thickness = 3
        else:
            color = (0, 255, 0)  # Verde para objetos normales
            thickness = 2
        
        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Dibujar centro
        cx, cy = obj['center']
        cv2.circle(frame, (cx, cy), 5, color, -1)
        
        # Label
        label_text = f'{label} {conf:.2f}'
        if obj['involved_in_accident']:
            label_text += ' [INVOLVED]'
        
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    
    # Dibujar accidentes detectados (solo los que están dentro de ROIs)
    for accident in results['accidents']:
        # Solo dibujar si el accidente está dentro de una ROI
        if 'roi' not in accident:
            continue
            
        x1, y1, x2, y2 = accident['bbox']
        conf = accident['confidence']
        
        # Dibujar bounding box rojo grueso para accidentes
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        # Dibujar centro
        cx, cy = accident['center']
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        
        # Label de accidente
        cv2.putText(
            frame,
            f'ACCIDENT {conf:.2f}',
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )
        
        # Mostrar objetos involucrados
        num_involved = len(accident['involved_objects'])
        if num_involved > 0:
            cv2.putText(
                frame,
                f'Involved: {num_involved} objects',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2
            )

    # ---------------------------
    # Reporte Dinámico
    # ---------------------------
    report_data = {
        "timestamp": current_video_time,
        "rois": {roi["name"]: roi["counts"].copy() for roi in rois},
        "accidents": len(results['accidents']),
        "total_objects": len(results['objects']),
        "fps": results['fps_info']['fps']
    }
    reporter.update(current_video_time, report_data)

    # ---------------------------
    # Visualización de UI
    # ---------------------------
    overlay = frame.copy()
    
    # Dibujar Polígonos y Panel de Estadísticas
    ui_y_offset = 30
    
    # Panel de FPS y Performance
    cv2.rectangle(overlay, (10, ui_y_offset - 25), (400, ui_y_offset + 75), (0, 0, 0), -1)
    
    # Mostrar FPS efectivo (considerando frame skipping)
    effective_fps = results['fps_info']['fps'] * PROCESS_EVERY_N_FRAMES if 'fps_info' in results else 0
    cv2.putText(frame, f"FPS: {effective_fps:.1f}", (20, ui_y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ui_y_offset += 25
    
    # Mostrar latencia de inferencia
    if 'fps_info' in results:
        cv2.putText(frame, f"Latency: {results['fps_info']['total_time']*1000:.1f}ms", (20, ui_y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    ui_y_offset += 25
    
    # Mostrar frame skipping info
    skip_text = f"Skip: 1/{PROCESS_EVERY_N_FRAMES}" if PROCESS_EVERY_N_FRAMES > 1 else "Skip: OFF"
    cv2.putText(frame, skip_text, (20, ui_y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    ui_y_offset += 25
    cv2.putText(frame, f"Accidents: {len(results['accidents'])}", (20, ui_y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    ui_y_offset += 40
    
    # Panel de ROIs
    for roi in rois:
        # Dibujar polígono
        cv2.polylines(frame, [roi["points"]], isClosed=True, color=roi["color"], thickness=2)
        
        # Dibujar fondo semitransparente para stats
        cv2.rectangle(overlay, (10, ui_y_offset - 25), (350, ui_y_offset + 150), (0, 0, 0), -1)
        
        # Título de la zona
        cv2.putText(frame, f"{roi['name']}:", (20, ui_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi["color"], 2)
        ui_y_offset += 25
        
        # Contadores por tipo
        for cls_name, count in roi["counts"].items():
            if count > 0:
                text = f"  {cls_name.capitalize()}: {count}"
                text_color = (0, 0, 255) if cls_name == "accident" else (255, 255, 255)
                cv2.putText(frame, text, (20, ui_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                ui_y_offset += 25
        
        ui_y_offset += 10 # Espacio extra entre zonas

    # Aplicar transparencia al panel
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.imshow("Sequential Dual-YOLO: Accident Detection + Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

