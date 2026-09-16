#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sensorqa.calibration import CalibrationRunResult
from sensorqa.presentation import WorkflowSummaryView


@dataclass(frozen=True)
class ReportMetadata:
    """Human-readable metadata attached to one generated report."""

    title: str = "SensorQA Characterization Report"
    subtitle: str | None = None
    prepared_for: str | None = None
    prepared_by: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate fields after initialization.
        
        Returns:
            None.
        """
        for name in ("title", "subtitle", "prepared_for", "prepared_by", "notes"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None.")
        if not self.title.strip():
            raise ValueError("title must not be empty.")


@dataclass(frozen=True)
class ReportBundle:
    """Stable report input independent of any output format."""

    metadata: ReportMetadata
    workflow: WorkflowSummaryView
    calibration_summary: dict[str, Any] | None = None
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_file: str | None = None

    @property
    def has_calibration(self) -> bool:
        """Return whether calibration results are attached.
        
        Returns:
            True when the condition is met; otherwise False.
        """
        return self.calibration_summary is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            "report": {
                "title": self.metadata.title,
                "subtitle": self.metadata.subtitle,
                "prepared_for": self.metadata.prepared_for,
                "prepared_by": self.metadata.prepared_by,
                "notes": self.metadata.notes,
                "generated_at_utc": self.generated_at_utc,
                "source_file": self.source_file,
            },
            "workflow": self.workflow.to_dict(),
            "calibration": self.calibration_summary,
        }


class ReportBuilder:
    """Build output-format-neutral report data from presentation results."""

    def build(
        self,
        workflow: WorkflowSummaryView,
        *,
        calibration: CalibrationRunResult | Mapping[str, Any] | None = None,
        metadata: ReportMetadata | None = None,
        source_file: str | Path | None = None,
    ) -> ReportBundle:
        """Build build.
        
        Args:
            workflow: Completed SensorQA workflow.
            calibration: Calibration used by this function.
            metadata: Metadata associated with the dataset or tool.
            source_file: Source file associated with the result.
        
        Returns:
            ReportBundle returned by this function.
        """
        if not isinstance(workflow, WorkflowSummaryView):
            raise TypeError("workflow must be a WorkflowSummaryView.")

        calibration_summary: dict[str, Any] | None = None
        if calibration is not None:
            if isinstance(calibration, CalibrationRunResult):
                calibration_summary = calibration.summary()
            elif isinstance(calibration, Mapping):
                calibration_summary = dict(calibration)
            else:
                raise TypeError(
                    "calibration must be CalibrationRunResult, a mapping, or None."
                )

        report_metadata = metadata or ReportMetadata()
        if not isinstance(report_metadata, ReportMetadata):
            raise TypeError("metadata must be ReportMetadata or None.")

        resolved_source = source_file
        if resolved_source is None and workflow.dataset is not None:
            resolved_source = workflow.dataset.source_file_name

        return ReportBundle(
            metadata=report_metadata,
            workflow=workflow,
            calibration_summary=calibration_summary,
            source_file=(str(Path(resolved_source)) if resolved_source else None),
        )


def build_report(
    workflow: WorkflowSummaryView,
    *,
    calibration: CalibrationRunResult | Mapping[str, Any] | None = None,
    metadata: ReportMetadata | None = None,
    source_file: str | Path | None = None,
) -> ReportBundle:
    """Convenience wrapper for building one report bundle.
    
    Args:
        workflow: Completed SensorQA workflow.
        calibration: Value for `calibration`.
        metadata: Metadata associated with the dataset or tool.
        source_file: Source file associated with the result.
    
    Returns:
        ReportBundle returned by the function.
    """

    return ReportBuilder().build(
        workflow,
        calibration=calibration,
        metadata=metadata,
        source_file=source_file,
    )
