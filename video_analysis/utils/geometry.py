import numpy as np
from typing import List, Tuple, Dict, Any

def percentage_to_pixels(percentage_points: List[Tuple[float, float]], frame_width: int, frame_height: int) -> np.ndarray:
    """
    Convert percentage-based ROI coordinates to pixel coordinates.
    """
    pixel_points = []
    for x_percent, y_percent in percentage_points:
        x_pixel = int(x_percent * frame_width)
        y_pixel = int(y_percent * frame_height)
        pixel_points.append((x_pixel, y_pixel))
    
    return np.array(pixel_points, np.int32)

def create_roi_from_percentage(name: str, percentage_points: List[Tuple[float, float]], color: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Create a ROI definition using percentage-based coordinates.
    """
    return {
        "name": name,
        "percentage_points": percentage_points,
        "points": None,  # Will be calculated dynamically
        "color": color,
        "counts": {}
    }

def update_roi_pixels(rois: List[Dict], frame_width: int, frame_height: int):
    """
    Update pixel coordinates for all ROIs based on current frame size.
    """
    for roi in rois:
        roi["points"] = percentage_to_pixels(
            roi["percentage_points"],
            frame_width,
            frame_height
        )
