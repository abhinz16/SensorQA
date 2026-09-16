#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from sensorqa.calibration import CalibrationRunResult
from sensorqa.presentation import WorkflowSummaryView
from sensorqa.services.analysis_service import AnalysisServiceResult
from sensorqa.services.calibration_service import CalibrationServiceResult


@dataclass(frozen=True)
class ExportResult:
    """Result returned by an export operation."""
    success: bool
    file_path: str | None
    errors: tuple[str, ...] = ()


class ExportService:
    """Lightweight JSON/CSV export service for application results."""

    def export_analysis_json(
        self,
        result: AnalysisServiceResult | WorkflowSummaryView,
        destination: str | Path,
        *,
        indent: int = 2,
    ) -> ExportResult:
        """Export analysis json.
        
        Args:
            result: Result object to process.
            destination: Destination used by this function.
            indent: Indent used by this function.
        
        Returns:
            ExportResult returned by this function.
        """
        if isinstance(result, AnalysisServiceResult):
            payload = result.summary.to_dict()
        elif isinstance(result, WorkflowSummaryView):
            payload = result.to_dict()
        else:
            raise TypeError(
                "result must be AnalysisServiceResult or WorkflowSummaryView."
            )
        return self._write_json(payload, destination, indent=indent)

    def export_calibration_json(
        self,
        result: CalibrationServiceResult | CalibrationRunResult,
        destination: str | Path,
        *,
        indent: int = 2,
    ) -> ExportResult:
        """Export calibration json.
        
        Args:
            result: Result object to process.
            destination: Destination used by this function.
            indent: Indent used by this function.
        
        Returns:
            ExportResult returned by this function.
        """
        if isinstance(result, CalibrationServiceResult):
            if not result.success or result.summary is None:
                return ExportResult(
                    success=False,
                    file_path=None,
                    errors=("Cannot export an unsuccessful calibration result.",),
                )
            payload = result.summary
        elif isinstance(result, CalibrationRunResult):
            payload = result.summary()
        else:
            raise TypeError(
                "result must be CalibrationServiceResult or CalibrationRunResult."
            )
        return self._write_json(payload, destination, indent=indent)

    def export_calibration_validation_csv(
        self,
        result: CalibrationServiceResult | CalibrationRunResult,
        destination: str | Path,
    ) -> ExportResult:
        """Export calibration validation csv.
        
        Args:
            result: Result object to process.
            destination: Destination used by this function.
        
        Returns:
            ExportResult returned by this function.
        """
        if isinstance(result, CalibrationServiceResult):
            data = result.evaluated_data
            if not result.success or data is None:
                return ExportResult(
                    success=False,
                    file_path=None,
                    errors=("Cannot export an unsuccessful calibration result.",),
                )
        elif isinstance(result, CalibrationRunResult):
            data = result.evaluated_validation_data
        else:
            raise TypeError(
                "result must be CalibrationServiceResult or CalibrationRunResult."
            )
        return self.export_dataframe_csv(data, destination)

    def export_dataframe_csv(
        self,
        data: pd.DataFrame,
        destination: str | Path,
        *,
        index: bool = True,
    ) -> ExportResult:
        """Export dataframe csv.
        
        Args:
            data: Input data to process.
            destination: Destination used by this function.
            index: Item index.
        
        Returns:
            ExportResult returned by this function.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        path = self._prepare_destination(destination)
        try:
            data.to_csv(path, index=index)
        except Exception as exc:
            return ExportResult(
                success=False,
                file_path=None,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
        return ExportResult(success=True, file_path=str(path))

    def _write_json(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
        destination: str | Path,
        *,
        indent: int,
    ) -> ExportResult:
        """Write json.
        
        Args:
            payload: Payload used by this function.
            destination: Destination used by this function.
            indent: Indent used by this function.
        
        Returns:
            ExportResult returned by this function.
        """
        path = self._prepare_destination(destination)
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    indent=indent,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                handle.write("\n")
        except Exception as exc:
            return ExportResult(
                success=False,
                file_path=None,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
        return ExportResult(success=True, file_path=str(path))

    @staticmethod
    def _prepare_destination(destination: str | Path) -> Path:
        """Return prepare destination.
        
        Args:
            destination: Destination used by this function.
        
        Returns:
            Resolved path.
        """
        if not isinstance(destination, (str, Path)):
            raise TypeError("destination must be a string path or pathlib.Path.")
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
