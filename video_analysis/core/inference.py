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
        print(model_accidents.names)
        self.model_coco = model_coco
        
        # Use config values if not provided
        self.accident_confidence = accident_confidence or config.inference.get("accident_confidence", 0.5)
        self.object_confidence = object_confidence or config.inference.get("object_confidence", 0.4)
        self.correlation_distance = correlation_distance or config.inference.get("correlation_distance", 100.0)
        
        logger.info(f"Initialized SequentialDualYOLO with conf_acc={self.accident_confidence}, conf_obj={self.object_confidence}")
        
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
        
        # Process accident detections
        accident_boxes = []
        for result in results_accidents:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                accident_data = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'involved_objects': []
                }
                
                accident_boxes.append(accident_data)
                
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
        
        # Correlate accidents with nearby objects
        for accident in accident_boxes:
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
        
        combined['accidents'] = accident_boxes
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
