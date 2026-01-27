from .exporter import ExportService

class ReportManager:
    def __init__(self, exporter: ExportService, interval_seconds: float = 5.0):
        self.exporter = exporter
        self.interval_seconds = interval_seconds
        self.last_report_time = 0.0

    def set_interval(self, seconds: float):
        """Dynamically update the reporting interval."""
        self.interval_seconds = seconds

    def update(self, current_time: float, data: dict):
        """
        Checks if enough time has passed since the last report.
        If so, triggers the export.
        """
        # Initialize last_report_time on first valid update if needed
        # (Assuming video time starts at 0, or we track relative diffs)
        
        if current_time - self.last_report_time >= self.interval_seconds:
            self.force_report(data)
            self.last_report_time = current_time

    def force_report(self, data: dict):
        """Manually trigger a report with the provided data."""
        # Inject timestamp into data if strictly needed by exporter, 
        # but usually data already has it or exporter handles it.
        self.exporter.export(data)
