import cv2
import numpy as np
import logging
from typing import Dict, List

# Local imports from new package structure
from .config import config
from .core.inference import SequentialDualYOLO
from .services.exporter import ConsoleExportService
from .services.reporter import ReportManager
from .services.notification import ConsoleNotificationService, AccidentNotifier
from .utils import resize_frame, create_roi_from_percentage, update_roi_pixels, get_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # ---------------------------
    # Load Models & Configuration
    # ---------------------------
    models = get_models()
    
    dual_yolo = SequentialDualYOLO(
        model_accidents=models['accidents'],
        model_coco=models['coco'],
        accident_confidence=config.inference.get("accident_confidence"),
        object_confidence=config.inference.get("object_confidence"),
        correlation_distance=config.inference.get("correlation_distance")
    )

    # ---------------------------
    # Setup ROIs
    # ---------------------------
    rois = []
    for roi_config in config.rois:
        rois.append(create_roi_from_percentage(
            name=roi_config['name'],
            percentage_points=roi_config['percentage_points'],
            color=tuple(roi_config['color'])
        ))
    
    logger.info(f"Loaded {len(rois)} ROIs")

    # ---------------------------
    # Services Setup
    # ---------------------------
    exporter = ConsoleExportService()
    reporter = ReportManager(
        exporter, 
        interval_seconds=config.reporting.get("interval_seconds", 5.0)
    )

    notification_service = ConsoleNotificationService(verbose=True)
    accident_notifier = AccidentNotifier(
        notification_service=notification_service,
        cooldown_seconds=config.notifications.get("cooldown_seconds", 10.0),
        min_confidence=config.notifications.get("min_confidence", 0.3)
    )

    # ---------------------------
    # Video Source Verification
    # ---------------------------
    video_source = 0 # Default to webcam
    # You might want to add video source to config if it's not dynamic
    # For now we'll stick to a default or argument parsing could be added here
    # Example: video_source = config.video.get("source", 0)
    # Using the hardcoded path from original main.py for now as per user context
    video_source = 'videos/moiz3.mp4' 
    
    logger.info(f"Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        return

    # Frame skipping settings
    process_every_n = config.video.get("process_every_n_frames", 2)
    target_width = config.video.get("target_width", 640)
    fast_resize = config.video.get("fast_resize", True)

    frame_count = 0
    last_results = None
    coco_classes = SequentialDualYOLO.COCO_CLASSES

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Resize and update ROIs
            frame = resize_frame(frame, target_width=target_width, fast_mode=fast_resize)
            h, w = frame.shape[:2]
            update_roi_pixels(rois, w, h)
            
            frame_count += 1
            should_process = (frame_count % process_every_n == 0)

            # ---------------------------
            # Inference Logic
            # ---------------------------
            if should_process:
                # Reset counters
                for roi in rois:
                    roi["counts"] = {name: 0 for name in coco_classes.values()}
                    roi["counts"]["accident"] = 0

                results = dual_yolo.process_frame(frame, rois=rois)
                last_results = results
                
                # Update ROI stats from results
                for roi in rois:
                    if roi["name"] in results['roi_stats']:
                        roi["counts"] = results['roi_stats'][roi["name"]]
            else:
                # Re-use last results
                if last_results:
                    results = last_results
                    for roi in rois:
                        if roi["name"] in results['roi_stats']:
                            roi["counts"] = results['roi_stats'][roi["name"]]
                else:
                    # First frame fallback
                     results = dual_yolo.process_frame(frame, rois=rois)
                     last_results = results

            # ---------------------------
            # Visualization
            # ---------------------------
            _draw_results(frame, results, rois)
            _draw_ui(frame, results, rois, process_every_n)

            # ---------------------------
            # Reporting & Notifications
            # ---------------------------
            report_data = {
                "timestamp": current_video_time,
                "rois": {r["name"]: r["counts"].copy() for r in rois},
                "accidents": len(results['accidents']),
                "total_objects": len(results['objects']),
                "fps": results['fps_info']['fps']
            }
            reporter.update(current_video_time, report_data)

            if should_process and results['accidents']:
                accident_notifier.process_accidents(
                    results['accidents'], 
                    current_time=current_video_time
                )

            cv2.imshow("Video Analysis Service 2.0", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Application shut down")

def _draw_results(frame, results, rois):
    """Helper to draw detection results"""
    # Draw objects in ROIs
    for obj in results['objects']:
        if 'roi' not in obj: continue
        
        x1, y1, x2, y2 = obj['bbox']
        color = (0, 255, 255) if obj['involved_in_accident'] else (0, 255, 0)
        thickness = 3 if obj['involved_in_accident'] else 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cx, cy = obj['center']
        cv2.circle(frame, (cx, cy), 5, color, -1)
        
        label = f"{obj['class']} {obj['confidence']:.2f}"
        if obj['involved_in_accident']: label += " [INVOLVED]"
        
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw accidents
    for acc in results['accidents']:
        if 'roi' not in acc: continue
        
        x1, y1, x2, y2 = acc['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cx, cy = acc['center']
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        
        cv2.putText(frame, f"ACCIDENT {acc['confidence']:.2f}", (x1, y1-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

def _draw_ui(frame, results, rois, process_every_n):
    """Helper to draw UI overlay"""
    overlay = frame.copy()
    ui_y = 30
    
    # Background for FPS
    cv2.rectangle(overlay, (10, ui_y-25), (400, ui_y+75), (0,0,0), -1)
    
    # FPS Info
    eff_fps = results['fps_info']['fps'] * process_every_n
    cv2.putText(frame, f"FPS: {eff_fps:.1f}", (20, ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    ui_y += 25
    
    if 'fps_info' in results:
        latency = results['fps_info']['total_time']*1000
        cv2.putText(frame, f"Latency: {latency:.1f}ms", (20, ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    ui_y += 25
    
    cv2.putText(frame, f"Skip: 1/{process_every_n}", (20, ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
    ui_y += 40

    # Draw ROI polygons and stats
    for roi in rois:
        cv2.polylines(frame, [roi["points"]], True, roi["color"], 2)
        
        # Stats background
        cv2.rectangle(overlay, (10, ui_y-25), (350, ui_y+150), (0,0,0), -1)
        
        cv2.putText(frame, f"{roi['name']}:", (20, ui_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi["color"], 2)
        ui_y += 25
        
        for cls, count in roi["counts"].items():
            if count > 0:
                color = (0,0,255) if cls == "accident" else (255,255,255)
                cv2.putText(frame, f"  {cls.capitalize()}: {count}", (20, ui_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                ui_y += 25
        ui_y += 10
        
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

if __name__ == "__main__":
    main()
