"""
FastAPI module for exposing ROI state via HTTP.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import threading
import uvicorn
import json

from .time import traffic_light_algorithm

app = FastAPI(title="Video Analysis Service API")

# Shared state - will be updated by the video processing loop
_roi_state: Dict[str, Dict[str, int]] = {}
_roi_state_lock = threading.Lock()

# Vehicle categories to count (excluding person and accident)
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]


def update_roi_state(rois_data: Dict[str, Dict[str, int]]):
    """
    Update the shared ROI state from the video processing loop.
    
    Args:
        rois_data: Dictionary mapping ROI names to their counts
    """
    global _roi_state
    with _roi_state_lock:
        _roi_state = rois_data.copy()


def get_roi_state() -> Dict[str, Dict[str, int]]:
    """
    Get the current ROI state (thread-safe).
    """
    with _roi_state_lock:
        return _roi_state.copy()


def _count_vehicles(roi_counts: Dict[str, int]) -> int:
    """Sum all vehicle counts (car, motorcycle, bus, truck) for a ROI."""
    return sum(roi_counts.get(cls, 0) for cls in VEHICLE_CLASSES)


@app.get("/roi-state")
def roi_state_endpoint() -> Dict[str, Any]:
    """
    Returns the current state of all ROIs.
    """
    return get_roi_state()


@app.get("/traffic-light")
def traffic_light_endpoint():
    """
    Returns traffic light timing based on vehicle counts in each ROI.
    Maps ROI A -> street_A, ROI B -> street_B.
    """
    state = get_roi_state()
    roi_names = list(state.keys())
    
    # Get vehicle counts for each ROI (default to 0 if not enough ROIs)
    count_a = _count_vehicles(state.get(roi_names[0], {})) if len(roi_names) > 0 else 0
    count_b = _count_vehicles(state.get(roi_names[1], {})) if len(roi_names) > 1 else 0
    
    # Call the traffic light algorithm
    result_json = traffic_light_algorithm(count_a, count_b)
    
    # Parse and return as proper JSON response
    return JSONResponse(content=json.loads(result_json))


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the FastAPI server in a background thread.
    """
    def run():
        uvicorn.run(app, host=host, port=port, log_level="warning")
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread
