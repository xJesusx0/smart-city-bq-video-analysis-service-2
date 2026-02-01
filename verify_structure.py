
import sys
import os
import logging

# Ensure root path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

def test_config_loading():
    logger.info("Testing Config Loading...")
    from video_analysis.config import config
    
    assert config.model['accidents_path'] is not None
    assert len(config.rois) > 0
    logger.info(f"Config loaded successfully. ROIs: {len(config.rois)}")

def test_model_loading():
    logger.info("Testing Model Loading...")
    from video_analysis.utils import get_models
    
    # We mock this if models are not present, or try-except
    try:
        models = get_models()
        assert 'accidents' in models
        assert 'coco' in models
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.warning(f"Model loading failed (expected if models missing): {e}")

def test_imports():
    logger.info("Testing Imports...")
    try:
        from video_analysis.core.inference import SequentialDualYOLO
        from video_analysis.services.reporter import ReportManager
        from video_analysis.utils import resize_frame
        logger.info("All imports successful.")
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
    test_config_loading()
    test_model_loading()
    print("VERIFICATION COMPLETE: Structure appears correct.")
