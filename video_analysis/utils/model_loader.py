from ultralytics import YOLO
from typing import Dict
from ..config import config
import logging
import os

logger = logging.getLogger(__name__)

def get_models() -> Dict[str, YOLO]:
    """
    Load both YOLO models for dual inference using paths from config.
    """
    accidents_path = config.model.get("accidents_path", "models/yolov8m-accidents.pt")
    coco_path = config.model.get("coco_path", "models/yolov8n.pt")
    
    # Validate paths
    if not os.path.exists(accidents_path) and not accidents_path.startswith("http"):
         logger.warning(f"Accident model not found at {accidents_path}")

    logger.info(f"Loading accidents model from {accidents_path}")
    model_accidents = YOLO(accidents_path)
    
    logger.info(f"Loading COCO model from {coco_path}")
    model_coco = YOLO(coco_path)
    
    return {
        'accidents': model_accidents,
        'coco': model_coco
    }
