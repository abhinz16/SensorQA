#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any


class SensorQAError(Exception):
    """
    Base exception for expected SensorQA application errors.

    SensorQAError provides:

        - a human-readable message
        - a stable error code
        - optional structured details

    The structured representation will later be useful for:

        - API responses
        - GUI error dialogs
        - logs
        - JSON reports

    Unexpected programming errors should generally NOT be converted
    into SensorQAError automatically, because doing so could hide
    actual software defects.
    """

    default_code = "sensorqa_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:

        """Initialize the sensor qaerror.
        
        Args:
            message: Message text.
            code: Code used by this function.
            details: Details used by this function.
        
        Returns:
            None.
        """
        if not isinstance(
            message,
            str,
        ) or not message.strip():

            raise ValueError(
                "SensorQA exception message must be a "
                "non-empty string."
            )

        super().__init__(
            message
        )

        self.message = message.strip()

        self.code = (
            code
            if code is not None
            else self.default_code
        )

        self.details = (
            dict(
                details
            )
            if details is not None
            else {}
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return a JSON-friendly representation.
        
        Returns:
            Dictionary containing the result values.
        """

        return {
            "error_type":
                self.__class__.__name__,

            "code":
                self.code,

            "message":
                self.message,

            "details":
                self.details,
        }

    def __str__(
        self,
    ) -> str:

        """Return a readable string representation.
        
        Returns:
            Requested text value.
        """
        return self.message


# Configuration


class ConfigurationError(
    SensorQAError
):
    """
    Base error for invalid SensorQA configuration.
    """

    default_code = "configuration_error"


class RuntimeConfigurationError(
    ConfigurationError
):
    """
    Raised when sensorqa.toml configuration cannot be used.
    """

    default_code = "runtime_configuration_error"


class RequirementsConfigurationError(
    ConfigurationError
):
    """
    Raised when requirements configuration cannot be used.
    """

    default_code = "requirements_configuration_error"


# Tool system


class ToolError(
    SensorQAError
):
    """
    Base exception for SensorQA analysis-tool problems.
    """

    default_code = "tool_error"


class ToolLoadingError(
    ToolError
):
    """
    Raised when a Python analysis tool cannot be loaded.

    Named ToolLoadingError rather than ToolLoadError because
    tool_loader.py already uses ToolLoadError as a report
    dataclass.
    """

    default_code = "tool_loading_error"


class ToolValidationError(
    ToolError
):
    """
    Raised when an analysis-tool definition is invalid.
    """

    default_code = "tool_validation_error"


class ToolNotFoundError(
    ToolError
):
    """
    Raised when a requested tool is not registered.
    """

    default_code = "tool_not_found"


class ToolExecutionError(
    ToolError
):
    """
    Raised by application-level code when a tool cannot complete
    execution and an exception-based interface is appropriate.

    Individual BaseAnalysisTool execution normally converts
    expected tool failures into AnalysisResult(ERROR), so this
    exception should primarily be used at orchestration boundaries.
    """

    default_code = "tool_execution_error"


# Dependency system


class DependencyError(
    SensorQAError
):
    """
    Base exception for tool dependency problems.
    """

    default_code = "dependency_error"


class MissingDependencyError(
    DependencyError
):
    """
    Raised when a required tool dependency is unavailable or
    omitted from a strict selection.
    """

    default_code = "missing_dependency"


class DependencyCycleError(
    DependencyError
):
    """
    Raised when tool dependencies contain a cycle.
    """

    default_code = "dependency_cycle"


# Dataset ingestion


class IngestionError(
    SensorQAError
):
    """
    Base exception for loading or preparing input data.
    """

    default_code = "ingestion_error"


class DatasetLoadError(
    IngestionError
):
    """
    Raised when an input dataset cannot be loaded.
    """

    default_code = "dataset_load_error"


class ColumnMappingError(
    IngestionError
):
    """
    Raised when physical dataset columns cannot be mapped into
    the requested SensorQA standard fields.
    """

    default_code = "column_mapping_error"


class UnitConversionError(
    IngestionError
):
    """
    Raised when required sensor-unit normalization cannot be
    performed.
    """

    default_code = "unit_conversion_error"


class DatasetValidationError(
    IngestionError
):
    """
    Raised when the dataset violates conditions required by an
    exception-based workflow.

    DatasetValidator itself should normally continue returning its
    structured DatasetValidationResult instead of raising.
    """

    default_code = "dataset_validation_error"


class DatasetBuildError(
    IngestionError
):
    """
    Raised when the canonical SensorQADataset cannot be built.
    """

    default_code = "dataset_build_error"


# Analysis pipeline


class PipelineError(
    SensorQAError
):
    """
    Base exception for analysis-pipeline orchestration problems.
    """

    default_code = "pipeline_error"


class PipelineConfigurationError(
    PipelineError
):
    """
    Raised when the requested pipeline configuration cannot be
    executed.
    """

    default_code = "pipeline_configuration_error"


class PipelineExecutionError(
    PipelineError
):
    """
    Raised when the overall analysis workflow cannot execute.
    """

    default_code = "pipeline_execution_error"


# Requirements and qualification


class QualificationError(
    SensorQAError
):
    """
    Base exception for qualification workflow failures.
    """

    default_code = "qualification_error"


class RequirementEvaluationError(
    QualificationError
):
    """
    Raised when a requirement cannot be evaluated because of an
    application-level problem rather than a normal
    NOT_EVALUATED condition.

    Missing metrics and unsupported experiments should normally
    produce NOT_EVALUATED instead of raising this exception.
    """

    default_code = "requirement_evaluation_error"


# Calibration


class CalibrationError(
    SensorQAError
):
    """
    Base exception for calibration-model workflow problems.
    """

    default_code = "calibration_error"


class CalibrationDataError(
    CalibrationError
):
    """
    Raised when calibration data are structurally insufficient
    for the requested calibration operation.
    """

    default_code = "calibration_data_error"


class CalibrationValidationError(
    CalibrationError
):
    """
    Raised when a calibration model cannot be evaluated using the
    requested independent validation data.
    """

    default_code = "calibration_validation_error"


# Reporting / export


class ReportingError(
    SensorQAError
):
    """
    Base exception for report construction and result export.
    """

    default_code = "reporting_error"


class ResultExportError(
    ReportingError
):
    """
    Raised when SensorQA cannot export requested results.
    """

    default_code = "result_export_error"


class PlotExportError(
    ReportingError
):
    """
    Raised when a requested plot cannot be generated or saved.
    """

    default_code = "plot_export_error"
