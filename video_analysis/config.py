import yaml
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class AppConfig:
    _instance = None
    _config = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self, config_path: str = "config/config.yaml"):
        path = Path(config_path)
        if not path.exists():
            # Fallback to default relative path if running from root
            path = Path.cwd() / "config" / "config.yaml"
            
        if not path.exists():
            logger.warning(f"Config file not found at {path}, using defaults")
            return

        try:
            with open(path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    @property
    def model(self) -> Dict[str, str]:
        return self._config.get("model", {})

    @property
    def inference(self) -> Dict[str, float]:
        return self._config.get("inference", {})

    @property
    def video(self) -> Dict[str, Any]:
        return self._config.get("video", {})
    
    @property
    def rois(self) -> List[Dict[str, Any]]:
        return self._config.get("rois", [])

    @property
    def reporting(self) -> Dict[str, float]:
        return self._config.get("reporting", {})
    
    @property
    def notifications(self) -> Dict[str, float]:
        return self._config.get("notifications", {})

config = AppConfig()
