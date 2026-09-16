from __future__ import annotations

from dataclasses import dataclass, field

from sensorqa.ingestion.dataset import SensorQADataset
from sensorqa.services import (
    AnalysisServiceResult,
    CSVPreviewResult,
    CalibrationServiceResult,
)


@dataclass
class DesktopPreferences:
    """User-facing preferences for the current desktop session."""

    run_qualification: bool = True
    open_report_after_export: bool = True
    calibration_validation_percent: float = 20.0
    calibration_split_strategy: str = "sequential"


@dataclass
class DesktopSession:
    """Mutable state shared by the desktop pages for one app session."""

    preview: CSVPreviewResult | None = None
    dataset: SensorQADataset | None = None
    analysis: AnalysisServiceResult | None = None
    calibration: CalibrationServiceResult | None = None
    preferences: DesktopPreferences = field(default_factory=DesktopPreferences)

    def clear_analysis(self) -> None:
        """Clear analysis and calibration results while keeping the dataset.
        
        Returns:
            None.
        """
        self.dataset = None
        self.analysis = None
        self.calibration = None

    def clear_all(self) -> None:
        """Clear the current dataset, analysis, and calibration state.
        
        Returns:
            None.
        """
        self.preview = None
        self.clear_analysis()
