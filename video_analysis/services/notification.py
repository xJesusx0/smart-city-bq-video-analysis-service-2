"""
Notification Service - Abstract Base and Implementations

This module provides an abstract notification service for accident alerts
with debouncing to prevent duplicate notifications for the same accident.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import time


class NotificationService(ABC):
    """
    Abstract base class for notification services.
    
    Implementations can send notifications via different channels:
    - Console (for development/testing)
    - Email
    - SMS
    - Webhook
    - Telegram/Slack
    """
    
    @abstractmethod
    def send_accident_alert(self, accident_data: Dict[str, Any]) -> bool:
        """
        Send an accident alert notification.
        
        Args:
            accident_data: Dictionary containing:
                - timestamp: Time of detection
                - confidence: Detection confidence
                - location: ROI name where detected
                - involved_objects: List of objects involved
                - bbox: Bounding box coordinates
                
        Returns:
            bool: True if notification sent successfully
        """
        pass
    
    @abstractmethod
    def send_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        Send a summary notification (e.g., daily report).
        
        Args:
            summary_data: Dictionary with summary information
            
        Returns:
            bool: True if notification sent successfully
        """
        pass


class ConsoleNotificationService(NotificationService):
    """
    Console-based notification service for development and testing.
    Prints formatted notifications to the console.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize console notification service.
        
        Args:
            verbose: If True, print detailed information
        """
        self.verbose = verbose
    
    def send_accident_alert(self, accident_data: Dict[str, Any]) -> bool:
        """Send accident alert to console."""
        try:
            print("\n" + "=" * 70)
            print("🚨 ACCIDENT DETECTED 🚨")
            print("=" * 70)
            
            # Timestamp
            timestamp = accident_data.get('timestamp', time.time())
            dt = datetime.fromtimestamp(timestamp)
            print(f"⏰ Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Confidence
            confidence = accident_data.get('confidence', 0.0)
            print(f"📊 Confidence: {confidence:.2%}")
            
            # Location
            location = accident_data.get('location', 'Unknown')
            print(f"📍 Location: {location}")
            
            # Involved objects
            involved = accident_data.get('involved_objects', [])
            if involved:
                print(f"🚗 Involved Objects ({len(involved)}):")
                for obj in involved:
                    obj_class = obj.get('class', 'unknown')
                    obj_conf = obj.get('confidence', 0.0)
                    distance = obj.get('distance_to_accident', 0.0)
                    print(f"   - {obj_class.capitalize()} (conf: {obj_conf:.2f}, dist: {distance:.0f}px)")
            else:
                print(f"🚗 Involved Objects: None detected")
            
            # Bounding box (if verbose)
            if self.verbose:
                bbox = accident_data.get('bbox', [])
                if bbox:
                    print(f"📦 Bounding Box: {bbox}")
            
            print("=" * 70 + "\n")
            return True
            
        except Exception as e:
            print(f"❌ Error sending console notification: {e}")
            return False
    
    def send_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send summary to console."""
        try:
            print("\n" + "=" * 70)
            print("📋 ACCIDENT SUMMARY")
            print("=" * 70)
            
            total = summary_data.get('total_accidents', 0)
            period = summary_data.get('period', 'Unknown')
            
            print(f"📊 Total Accidents: {total}")
            print(f"⏱️  Period: {period}")
            
            # By location
            by_location = summary_data.get('by_location', {})
            if by_location:
                print(f"\n📍 By Location:")
                for location, count in by_location.items():
                    print(f"   - {location}: {count}")
            
            print("=" * 70 + "\n")
            return True
            
        except Exception as e:
            print(f"❌ Error sending summary: {e}")
            return False


class AccidentNotifier:
    """
    Manages accident notifications with debouncing to prevent duplicates.
    
    Debouncing ensures that multiple detections of the same accident
    (across consecutive frames) only trigger one notification.
    """
    
    def __init__(
        self,
        notification_service: NotificationService,
        cooldown_seconds: float = 10.0,
        min_confidence: float = 0.3
    ):
        """
        Initialize accident notifier.
        
        Args:
            notification_service: Service to use for notifications
            cooldown_seconds: Seconds to wait before allowing another notification
            min_confidence: Minimum confidence to trigger notification
        """
        self.notification_service = notification_service
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        
        # Track last notification time per ROI
        self.last_notification_time: Dict[str, float] = {}
        
        # Statistics
        self.total_detections = 0
        self.total_notifications = 0
        self.suppressed_notifications = 0
    
    def process_accidents(
        self,
        accidents: list,
        current_time: Optional[float] = None
    ) -> int:
        """
        Process accident detections and send notifications if needed.
        
        Args:
            accidents: List of accident detections from dual-YOLO
            current_time: Current timestamp (uses time.time() if None)
            
        Returns:
            int: Number of notifications sent
        """
        if current_time is None:
            current_time = time.time()
        
        notifications_sent = 0
        
        for accident in accidents:
            self.total_detections += 1
            
            # Check confidence threshold
            confidence = accident.get('confidence', 0.0)
            if confidence < self.min_confidence:
                continue
            
            # Get location (ROI)
            location = accident.get('roi', 'Unknown')
            
            # Check cooldown
            if self._is_in_cooldown(location, current_time):
                self.suppressed_notifications += 1
                continue
            
            # Send notification
            accident_data = {
                'timestamp': current_time,
                'confidence': confidence,
                'location': location,
                'involved_objects': accident.get('involved_objects', []),
                'bbox': accident.get('bbox', [])
            }
            
            if self.notification_service.send_accident_alert(accident_data):
                self.last_notification_time[location] = current_time
                self.total_notifications += 1
                notifications_sent += 1
        
        return notifications_sent
    
    def _is_in_cooldown(self, location: str, current_time: float) -> bool:
        """
        Check if a location is still in cooldown period.
        
        Args:
            location: ROI name
            current_time: Current timestamp
            
        Returns:
            bool: True if in cooldown, False otherwise
        """
        if location not in self.last_notification_time:
            return False
        
        time_since_last = current_time - self.last_notification_time[location]
        return time_since_last < self.cooldown_seconds
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get notification statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_detections': self.total_detections,
            'total_notifications': self.total_notifications,
            'suppressed_notifications': self.suppressed_notifications,
            'suppression_rate': (
                self.suppressed_notifications / self.total_detections
                if self.total_detections > 0 else 0.0
            )
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self.total_detections = 0
        self.total_notifications = 0
        self.suppressed_notifications = 0
