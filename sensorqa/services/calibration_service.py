#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from sensorqa.calibration import (
    CalibrationManager,
    CalibrationRunConfig,
    CalibrationRunResult,
)
from sensorqa.ingestion.dataset import SensorQADataset


@dataclass(frozen=True)
class CalibrationRequest:
    """Request for one held-out calibration run."""

    data: pd.DataFrame | SensorQADataset
    measurement_column: str = "measurement"
    reference_column: str = "reference"
    config: CalibrationRunConfig | None = None
    raise_on_error: bool = False

    def __post_init__(self) -> None:
        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(self.data, (pd.DataFrame, SensorQADataset)):
            raise TypeError("data must be a DataFrame or SensorQADataset.")
        if not isinstance(self.measurement_column, str) or not self.measurement_column.strip():
            raise ValueError("measurement_column must be a non-empty string.")
        if not isinstance(self.reference_column, str) or not self.reference_column.strip():
            raise ValueError("reference_column must be a non-empty string.")
        if self.config is not None and not isinstance(self.config, CalibrationRunConfig):
            raise TypeError("config must be a CalibrationRunConfig or None.")
        if not isinstance(self.raise_on_error, bool):
            raise TypeError("raise_on_error must be a boolean.")


@dataclass(frozen=True)
class CalibrationServiceResult:
    """Structured result suitable for a calibration results screen."""

    success: bool
    run_result: CalibrationRunResult | None
    summary: dict[str, Any] | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def evaluated_data(self) -> pd.DataFrame | None:
        """Return a copy of the row-level calibration evaluation data.
        
        Returns:
            DataFrame containing the requested data.
        """
        if self.run_result is None:
            return None
        return self.run_result.evaluated_validation_data


class CalibrationService:
    """Service wrapper around SensorQA's held-out calibration manager."""

    def __init__(self, manager: CalibrationManager | None = None) -> None:
        """Initialize the calibration service.
        
        Args:
            manager: Dependency manager fixture.
        
        Returns:
            None.
        """
        self.manager = manager or CalibrationManager()
        if not isinstance(self.manager, CalibrationManager):
            raise TypeError("manager must be a CalibrationManager.")

    def run(self, request: CalibrationRequest) -> CalibrationServiceResult:
        """Run run.
        
        Args:
            request: Request object containing the user inputs.
        
        Returns:
            CalibrationServiceResult returned by this function.
        """
        if not isinstance(request, CalibrationRequest):
            raise TypeError("request must be a CalibrationRequest.")

        try:
            result = self.manager.run(
                request.data,
                measurement_column=request.measurement_column,
                reference_column=request.reference_column,
                config=request.config,
            )
        except Exception as exc:
            if request.raise_on_error:
                raise
            return CalibrationServiceResult(
                success=False,
                run_result=None,
                summary=None,
                warnings=(),
                errors=(f"{type(exc).__name__}: {exc}",),
            )

        return CalibrationServiceResult(
            success=True,
            run_result=result,
            summary=result.summary(),
            warnings=tuple(result.warnings),
            errors=(),
        )
