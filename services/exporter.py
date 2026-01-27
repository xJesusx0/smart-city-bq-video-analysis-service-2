from abc import ABC, abstractmethod
import json

class ExportService(ABC):
    @abstractmethod
    def export(self, data: dict):
        pass

class ConsoleExportService(ExportService):
    def export(self, data: dict):
        """
        Prints the reporting data to the console in a readable format.
        Expected data format:
        {
            "timestamp": float,
            "rois": {
                "Zone Name": {"Class": count, ...},
                ...
            }
        }
        """
        timestamp = data.get("timestamp", 0.0)
        print(f"--- Report at {timestamp:.2f}s ---")
        for roi_name, counts in data.get("rois", {}).items():
            # Filter out zero counts for cleaner output if desired, 
            # or just print meaningful ones.
            active_counts = {k: v for k, v in counts.items() if v > 0}
            if active_counts:
                # Format: "Car: 2, Person: 1"
                count_str = ", ".join([f"{k}: {v}" for k, v in active_counts.items()])
                print(f"[{roi_name}] {count_str}")
            else:
                print(f"[{roi_name}] No activity")
        print("-----------------------------")
