#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sensorqa.calibration import CalibrationRunResult
from sensorqa.presentation import WorkflowSummaryView
from sensorqa.reporting import (
    HTMLReportOptions,
    HTMLReportRenderer,
    ReportBuilder,
    ReportMetadata,
)
from sensorqa.services.analysis_service import AnalysisServiceResult
from sensorqa.services.calibration_service import CalibrationServiceResult


@dataclass(frozen=True)
class ReportExportResult:
    """Result returned when an HTML report is exported."""
    success: bool
    file_path: str | None
    errors: tuple[str, ...] = ()


class ReportService:
    """Application-facing report generation service."""

    def __init__(
        self,
        *,
        builder: ReportBuilder | None = None,
        html_renderer: HTMLReportRenderer | None = None,
    ) -> None:
        """Initialize the report service.
        
        Args:
            builder: Builder used by this function.
            html_renderer: Html renderer used by this function.
        
        Returns:
            None.
        """
        self.builder = builder or ReportBuilder()
        self.html_renderer = html_renderer or HTMLReportRenderer()
        if not isinstance(self.builder, ReportBuilder):
            raise TypeError("builder must be a ReportBuilder.")
        if not isinstance(self.html_renderer, HTMLReportRenderer):
            raise TypeError("html_renderer must be an HTMLReportRenderer.")

    def export_html(
        self,
        analysis: AnalysisServiceResult | WorkflowSummaryView,
        destination: str | Path,
        *,
        calibration: CalibrationServiceResult | CalibrationRunResult | None = None,
        metadata: ReportMetadata | None = None,
        options: HTMLReportOptions | None = None,
        raise_on_error: bool = False,
    ) -> ReportExportResult:
        """Export html.
        
        Args:
            analysis: Analysis used by this function.
            destination: Destination used by this function.
            calibration: Calibration used by this function.
            metadata: Metadata associated with the dataset or tool.
            options: Options used by this function.
            raise_on_error: Controls whether to raise on error.
        
        Returns:
            ReportExportResult returned by this function.
        """
        try:
            if isinstance(analysis, AnalysisServiceResult):
                workflow = analysis.summary
                source_file = analysis.source_file
            elif isinstance(analysis, WorkflowSummaryView):
                workflow = analysis
                source_file = None
            else:
                raise TypeError(
                    "analysis must be AnalysisServiceResult or WorkflowSummaryView."
                )

            calibration_payload = None
            if isinstance(calibration, CalibrationServiceResult):
                if not calibration.success or calibration.summary is None:
                    raise ValueError(
                        "Cannot include an unsuccessful calibration result in a report."
                    )
                calibration_payload = calibration.summary
            elif isinstance(calibration, CalibrationRunResult):
                calibration_payload = calibration
            elif calibration is not None:
                raise TypeError(
                    "calibration must be CalibrationServiceResult, "
                    "CalibrationRunResult, or None."
                )

            report = self.builder.build(
                workflow,
                calibration=calibration_payload,
                metadata=metadata,
                source_file=source_file,
            )
            path = self.html_renderer.write(
                report,
                destination,
                options=options,
            )
        except Exception as exc:
            if raise_on_error:
                raise
            return ReportExportResult(
                success=False,
                file_path=None,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
        return ReportExportResult(success=True, file_path=str(path))
