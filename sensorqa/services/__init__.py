#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public application-service API for SensorQA.

The services package contains application facades that coordinate
SensorQA's backend workflows for presentation surfaces such as the future
desktop application.

UI code should prefer importing from ``sensorqa.services`` rather than from
individual service modules. This keeps the user interface decoupled from the
internal organization of the service layer.
"""

from sensorqa.services.analysis_service import (
    AnalysisService,
    AnalysisServiceResult,
    CSVAnalysisRequest,
)
from sensorqa.services.application_services import (
    SensorQAServices,
    create_services,
)
from sensorqa.services.calibration_service import (
    CalibrationRequest,
    CalibrationService,
    CalibrationServiceResult,
)
from sensorqa.services.export_service import (
    ExportResult,
    ExportService,
)
from sensorqa.services.ingestion_service import (
    CSVPreviewRequest,
    CSVPreviewResult,
    DatasetBuildRequest,
    DatasetBuildServiceResult,
    IngestionService,
)
from sensorqa.services.report_service import (
    ReportExportResult,
    ReportService,
)
from sensorqa.services.tool_service import (
    ToolHealthView,
    ToolParameterView,
    ToolService,
    ToolView,
)

__all__ = [
    "CSVAnalysisRequest",
    "AnalysisServiceResult",
    "AnalysisService",
    "CSVPreviewRequest",
    "CSVPreviewResult",
    "DatasetBuildRequest",
    "DatasetBuildServiceResult",
    "IngestionService",
    "CalibrationRequest",
    "CalibrationServiceResult",
    "CalibrationService",
    "ToolParameterView",
    "ToolView",
    "ToolHealthView",
    "ToolService",
    "ExportResult",
    "ExportService",
    "ReportExportResult",
    "ReportService",
    "SensorQAServices",
    "create_services",
]
