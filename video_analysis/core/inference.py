"""
Sequential Dual-YOLO Inference System

This module implements a sequential architecture for running two YOLO models:
1. YOLOv8m-accidents: Detects accidents
2. YOLOv8 COCO: Detects objects (person, car, motorcycle, bus, truck)

The models are executed sequentially on the same frame to detect accidents
and identify the objects involved.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Any
import time
import logging
from ..config import config

logger = logging.getLogger(__name__)


class AccidentTracker:
    """
    Tracks accident candidates over multiple frames to filter out false positives.
    An accident is only confirmed if it persists for 'persistence_threshold' consecutive frames.
    """
    def __init__(self, persistence_threshold: int = 3, iou_threshold: float = 0.3):
        self.persistence_threshold = persistence_threshold
        self.iou_threshold = iou_threshold
        # List of tracked accidents: {'bbox': [x1, y1, x2, y2], 'frames_seen': int, 'confirmed': bool, 'data': dict}
        self.tracked_accidents = []

    def update(self, current_detections: List[Dict]) -> List[Dict]:
        """
        Update tracked accidents with current frame detections.
        
        Args:
            current_detections: List of accident dictionaries from current frame
            
        Returns:
            List of CONFIRMED accident dictionaries to be reported
        """
        # Match current detections with tracked accidents using IoU
        # Simple greedy matching
        matched_indices = set()
        updated_tracks = []

        for track in self.tracked_accidents:
            best_iou = 0
            best_idx = -1
            
            for i, det in enumerate(current_detections):
                if i in matched_indices:
                    continue
                
                iou = self._calculate_iou(track['bbox'], det['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_idx != -1:
                # Match found, update track
                matched_indices.add(best_idx)
                det = current_detections[best_idx]
                track['bbox'] = det['bbox']
                track['data'] = det # Update with latest confidence/info
                track['frames_seen'] += 1
                updated_tracks.append(track)
            else:
                # Track lost - we could implement a "patience" logic here, 
                # but for strict filtering we drop it if lost in one frame
                # Or we can just drop it. Let's drop it for now to be strict.
                pass
        
        # Add new detections as new tracks
        for i, det in enumerate(current_detections):
            if i not in matched_indices:
                updated_tracks.append({
                    'bbox': det['bbox'],
                    'frames_seen': 1,
                    'confirmed': False,
                    'data': det
                })
        
        self.tracked_accidents = updated_tracks
        
        # Collect confirmed accidents
        confirmed_accidents = []
        for track in self.tracked_accidents:
            if track['frames_seen'] >= self.persistence_threshold:
                track['confirmed'] = True
                confirmed_accidents.append(track['data'])
                
        return confirmed_accidents

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return intersection_area / float(box1_area + box2_area - intersection_area)


class SequentialDualYOLO:
    """
    Sequential dual-YOLO inference system for accident detection and object tracking.
    """
    
    # COCO classes of interest
    COCO_CLASSES = {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
    COCO_CLASS_IDS = list(COCO_CLASSES.keys())
    
    def __init__(
        self,
        model_accidents: YOLO,
        model_coco: YOLO,
        accident_confidence: float = None,
        object_confidence: float = None,
        correlation_distance: float = None
    ):
        """
        Initialize the sequential dual-YOLO system.
        
        Args:
            model_accidents: YOLO model for accident detection
            model_coco: YOLO model for object detection (COCO)
            accident_confidence: Minimum confidence for accident detection
            object_confidence: Minimum confidence for object detection
            correlation_distance: Maximum distance (pixels) to correlate objects with accidents
        """
        self.model_accidents = model_accidents
        self.model_coco = model_coco
        
        # Use config values if not provided
        self.accident_confidence = accident_confidence or config.inference.get("accident_confidence", 0.5)
        self.object_confidence = object_confidence or config.inference.get("object_confidence", 0.4)
        self.correlation_distance = correlation_distance or config.inference.get("correlation_distance", 100.0)
        
        # Initialize Accident Tracker
        temporal_threshold = config.inference.get("temporal_threshold", 3)
        self.tracker = AccidentTracker(persistence_threshold=temporal_threshold)
        
        logger.info(f"Initialized SequentialDualYOLO with conf_acc={self.accident_confidence}, conf_obj={self.object_confidence}, temporal_threshold={temporal_threshold}")
        
    def process_frame(self, frame: np.ndarray, rois: List[Dict] = None) -> Dict[str, Any]:
        """
        Process a frame with both YOLO models sequentially.
        
        Args:
            frame: Input frame (BGR format)
            rois: Optional list of regions of interest
            
        Returns:
            Dictionary containing:
                - timestamp: Processing timestamp
                - accidents: List of accident detections
                - objects: List of object detections
                - roi_stats: Statistics per ROI (if provided)
                - fps_info: Processing time information
        """
        start_time = time.time()
        
        # Preprocess frame (shared preprocessing)
        preprocessed = self._preprocess_frame(frame)
        preprocess_time = time.time()
        
        # Step 1: Detect accidents
        results_accidents = self.model_accidents(
            preprocessed,
            verbose=False,
            conf=self.accident_confidence
        )
        accidents_time = time.time()
        
        # Step 2: Detect objects
        results_coco = self.model_coco(
            preprocessed,
            verbose=False,
            classes=self.COCO_CLASS_IDS,
            conf=self.object_confidence
        )
        coco_time = time.time()
        
        # Step 3: Combine and correlate results
        combined_results = self._combine_results(
            results_accidents,
            results_coco,
            frame,
            rois
        )
        combine_time = time.time()
        
        # Add timing information
        combined_results['fps_info'] = {
            'total_time': combine_time - start_time,
            'preprocess_time': preprocess_time - start_time,
            'accidents_inference_time': accidents_time - preprocess_time,
            'coco_inference_time': coco_time - accidents_time,
            'postprocess_time': combine_time - coco_time,
            'fps': 1.0 / (combine_time - start_time) if (combine_time - start_time) > 0 else 0
        }
        
        return combined_results
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLO inference.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # YOLO models handle preprocessing internally, but we can do
        # any custom preprocessing here if needed
        return frame
    
    def _combine_results(
        self,
        results_accidents,
        results_coco,
        frame: np.ndarray,
        rois: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Combine results from both models and correlate accidents with objects.
        
        Args:
            results_accidents: Results from accident detection model
            results_coco: Results from COCO object detection model
            frame: Original frame
            rois: Optional regions of interest
            
        Returns:
            Combined results dictionary
        """
        combined = {
            'timestamp': time.time(),
            'accidents': [],
            'objects': [],
            'roi_stats': {}
        }
        
        # Initialize ROI stats if provided
        if rois:
            for roi in rois:
                roi_name = roi['name']
                combined['roi_stats'][roi_name] = {
                    cls_name: 0 for cls_name in self.COCO_CLASSES.values()
                }
                combined['roi_stats'][roi_name]['accident'] = 0
        
        # Process raw accident detections
        raw_accent_detections = []
        for result in results_accidents:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                accident_data = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'involved_objects': [] # Will be populated if confirmed
                }
                raw_accent_detections.append(accident_data)
        
        # FILTER: Apply Temporal Consistency Check
        confirmed_accidents = self.tracker.update(raw_accent_detections)
        
        # Process confirmed accidents for ROI and stats
        for accident_data in confirmed_accidents:
            # Check ROI membership
            if rois:
                for roi in rois:
                    inside = cv2.pointPolygonTest(
                        roi['points'],
                        accident_data['center'],
                        False
                    )
                    if inside >= 0:
                        accident_data['roi'] = roi['name']
                        combined['roi_stats'][roi['name']]['accident'] += 1
                        break
        
        # Process object detections
        object_boxes = []
        for result in results_coco:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.COCO_CLASSES.get(cls_id, f"class_{cls_id}")
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                object_data = {
                    'class': cls_name,
                    'class_id': cls_id,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'center': center,
                    'involved_in_accident': False
                }
                
                # Check ROI membership
                if rois:
                    for roi in rois:
                        inside = cv2.pointPolygonTest(
                            roi['points'],
                            center,
                            False
                        )
                        if inside >= 0:
                            object_data['roi'] = roi['name']
                            combined['roi_stats'][roi['name']][cls_name] += 1
                            break
                
                object_boxes.append(object_data)
        
        # Correlate confirmed accidents with nearby objects
        for accident in confirmed_accidents:
            for obj in object_boxes:
                distance = self._calculate_distance(
                    accident['center'],
                    obj['center']
                )
                
                if distance <= self.correlation_distance:
                    # Object is involved in accident
                    obj['involved_in_accident'] = True
                    accident['involved_objects'].append({
                        'class': obj['class'],
                        'class_id': obj['class_id'],
                        'bbox': obj['bbox'],
                        'confidence': obj['confidence'],
                        'distance_to_accident': distance
                    })
        
        combined['accidents'] = confirmed_accidents
        combined['objects'] = object_boxes
        
        return combined
    
    def _calculate_distance(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(
            (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
        )
    
    def get_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract statistics from processing results.
        
        Args:
            results: Results from process_frame()
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_accidents': len(results['accidents']),
            'total_objects': len(results['objects']),
            'objects_involved_in_accidents': sum(
                1 for obj in results['objects'] if obj['involved_in_accident']
            ),
            'objects_by_class': {},
            'fps': results['fps_info']['fps'],
            'latency_ms': results['fps_info']['total_time'] * 1000
        }
        
        # Count objects by class
        for obj in results['objects']:
            cls = obj['class']
            stats['objects_by_class'][cls] = stats['objects_by_class'].get(cls, 0) + 1
        
        return stats
