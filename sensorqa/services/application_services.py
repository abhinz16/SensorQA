#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

from sensorqa.bootstrap import SensorQABootstrapResult
from sensorqa.services.analysis_service import AnalysisService
from sensorqa.services.calibration_service import CalibrationService
from sensorqa.services.export_service import ExportService
from sensorqa.services.ingestion_service import IngestionService
from sensorqa.services.report_service import ReportService
from sensorqa.services.tool_service import ToolService


@dataclass(frozen=True)
class SensorQAServices:
    """All application SensorQA services used by the desktop UI."""

    ingestion: IngestionService
    analysis: AnalysisService
    calibration: CalibrationService
    tools: ToolService
    export: ExportService
    reports: ReportService


def create_services(
    bootstrap: SensorQABootstrapResult,
) -> SensorQAServices:
    """Create the complete service layer from one initialized bootstrap.
    
    Args:
        bootstrap: Value for `bootstrap`.
    
    Returns:
        SensorQAServices returned by the function.
    """

    if not isinstance(bootstrap, SensorQABootstrapResult):
        raise TypeError("bootstrap must be a SensorQABootstrapResult.")

    if not bootstrap.initialized:
        raise RuntimeError(
            "create_services() requires a successfully initialized SensorQA bootstrap."
        )

    application = bootstrap.application

    return SensorQAServices(
        ingestion=IngestionService(
            csv_loader=application.dataset_builder.csv_loader,
            column_mapper=application.dataset_builder.column_mapper,
            unit_manager=application.dataset_builder.unit_manager,
            dataset_builder=application.dataset_builder,
        ),
        analysis=AnalysisService(application),
        calibration=CalibrationService(),
        tools=ToolService(
            registry=bootstrap.registry,
            runtime_config=application.runtime_config,
            loader=bootstrap.tool_loader,
        ),
        export=ExportService(),
        reports=ReportService(),
    )
