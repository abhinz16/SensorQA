#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from sensorqa.core.application import SensorQAWorkflowResult
from sensorqa.core.requirements_engine import QualificationSummary
from sensorqa.core.result_schema import (
    AnalysisResult,
    Evidence,
    MetricResult,
    PlotResult,
    QualificationStatus,
    RequirementCheck,
)
from sensorqa.ingestion.dataset import SensorQADataset
from sensorqa.ingestion.dataset_validator import DatasetIssue


# JSON/UI-safe conversion


def _plain_value(value: Any) -> Any:
    """Convert common SensorQA/Python values to JSON/UI-safe values.
    
    Args:
        value: Value to process.
    
    Returns:
        Any returned by the function.
    """

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): _plain_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _plain_value(item)
            for item in value
        ]

    return value


# Dataset view models


@dataclass(frozen=True)
class DatasetIssueView:
    """Display model for a dataset validation issue."""
    issue_id: str
    severity: str
    message: str
    column: str | None = None
    affected_rows: int | None = None
    affected_fraction: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingSummaryView:
    """Display model for dataset sampling information."""
    available: bool
    sample_count: int
    duration_seconds: float | None
    median_interval_seconds: float | None
    mean_interval_seconds: float | None
    std_interval_seconds: float | None
    estimated_sampling_rate_hz: float | None
    duplicate_timestamp_count: int
    backward_timestamp_count: int
    large_gap_count: int
    largest_gap_seconds: float | None
    coefficient_of_variation: float | None


@dataclass(frozen=True)
class DatasetSummaryView:
    """Display model for the dataset summary shown in the app and reports."""
    source_file_name: str | None
    sensor_type: str | None
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    units: dict[str, str]
    valid: bool | None
    error_count: int
    warning_count: int
    information_count: int
    issues: tuple[DatasetIssueView, ...]
    sampling: SamplingSummaryView | None
    metadata: dict[str, Any]


# Analysis view models


@dataclass(frozen=True)
class MetricView:
    """Display model for one analysis metric."""
    name: str
    value: float | int | None
    unit: str | None
    description: str | None


@dataclass(frozen=True)
class EvidenceView:
    """Display model for one evidence item."""
    statement: str
    strength: str
    supporting_metrics: tuple[str, ...]


@dataclass(frozen=True)
class PlotView:
    """Display model for one plot definition."""
    plot_id: str
    title: str
    plot_type: str
    description: str | None
    file_path: str | None


@dataclass(frozen=True)
class AnalysisSummaryView:
    """Display model for one analysis result."""
    tool_id: str
    tool_name: str
    tool_version: str
    execution_status: str
    qualification_status: str
    metrics: tuple[MetricView, ...]
    evidence: tuple[EvidenceView, ...]
    plots: tuple[PlotView, ...]
    warnings: tuple[str, ...]
    messages: tuple[str, ...]
    metadata: dict[str, Any]


# Qualification view models


@dataclass(frozen=True)
class RequirementCheckView:
    """Display model for one requirement check."""
    tool_id: str
    tool_name: str
    metric_name: str
    measured_value: float | int | None
    unit: str | None
    requirement_description: str
    status: str
    limit_value: float | int | None
    operator: str | None
    message: str | None


@dataclass(frozen=True)
class QualificationSummaryView:
    """Display model for the overall requirement-checking summary."""
    overall_status: str
    total_requirements: int
    passed: int
    failed: int
    not_evaluated: int
    checks: tuple[RequirementCheckView, ...]
    messages: tuple[str, ...]


# Complete workflow view model


@dataclass(frozen=True)
class WorkflowSummaryView:
    """
    Stable, UI/report-facing representation of one SensorQA workflow.

    It intentionally excludes the raw measurement DataFrame. The desktop
    application may hold the SensorQADataset separately when it needs plots
    or table views, while result panels and reports can consume this compact
    representation directly.
    """

    success: bool
    dataset: DatasetSummaryView | None
    pipeline_success: bool | None
    analysis_count: int
    successful_analysis_count: int
    skipped_analysis_count: int
    error_analysis_count: int
    execution_order: tuple[str, ...]
    analyses: tuple[AnalysisSummaryView, ...]
    qualification: QualificationSummaryView | None
    pipeline_warnings: tuple[str, ...]
    pipeline_errors: tuple[str, ...]
    workflow_warnings: tuple[str, ...]
    workflow_messages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively JSON-safe dictionary.
        
        Returns:
            Dictionary containing the result values.
        """

        return _plain_value(
            asdict(self)
        )


# Builder


class WorkflowSummaryBuilder:
    """
    Adapt internal SensorQA workflow objects for application presentation.

    The desktop app should not need to understand PipelineResult,
    AnalysisResult, DatasetValidationResult, or QualificationSummary internals.
    This builder provides a stable boundary between backend computation and UI.
    """

    def build(
        self,
        workflow: SensorQAWorkflowResult,
    ) -> WorkflowSummaryView:

        """Build build.
        
        Args:
            workflow: Completed SensorQA workflow.
        
        Returns:
            WorkflowSummaryView returned by this function.
        """
        if not isinstance(
            workflow,
            SensorQAWorkflowResult,
        ):
            raise TypeError(
                "workflow must be a SensorQAWorkflowResult."
            )

        dataset_view = self._dataset_view(
            workflow.dataset
        )

        pipeline = workflow.pipeline_result

        analyses = tuple(
            workflow.analyses
        )

        qualification_status_by_tool = (
            self._qualification_status_by_tool(
                workflow.qualification
            )
        )

        analysis_views = tuple(
            self._analysis_view(
                result,
                qualification_status=(
                    qualification_status_by_tool.get(
                        result.tool_id,
                        QualificationStatus.NOT_EVALUATED,
                    )
                ),
            )
            for result in analyses
        )

        qualification_view = (
            self._qualification_view(
                workflow.qualification
            )
        )

        if pipeline is None:
            pipeline_success = None
            execution_order: tuple[str, ...] = ()
            pipeline_warnings: tuple[str, ...] = ()
            pipeline_errors: tuple[str, ...] = ()
        else:
            pipeline_success = bool(
                pipeline.success
            )
            execution_order = tuple(
                pipeline.execution_order
            )
            pipeline_warnings = tuple(
                pipeline.warnings
            )
            pipeline_errors = tuple(
                pipeline.errors
            )

        successful_count = sum(
            1
            for result in analyses
            if result.status.value == "success"
        )

        skipped_count = sum(
            1
            for result in analyses
            if result.status.value == "skipped"
        )

        error_count = sum(
            1
            for result in analyses
            if result.status.value == "error"
        )

        return WorkflowSummaryView(
            success=bool(
                workflow.success
            ),
            dataset=dataset_view,
            pipeline_success=pipeline_success,
            analysis_count=len(
                analyses
            ),
            successful_analysis_count=successful_count,
            skipped_analysis_count=skipped_count,
            error_analysis_count=error_count,
            execution_order=execution_order,
            analyses=analysis_views,
            qualification=qualification_view,
            pipeline_warnings=pipeline_warnings,
            pipeline_errors=pipeline_errors,
            workflow_warnings=tuple(
                workflow.warnings
            ),
            workflow_messages=tuple(
                workflow.messages
            ),
        )

    def _dataset_view(
        self,
        dataset: SensorQADataset | None,
    ) -> DatasetSummaryView | None:

        """Return dataset view.
        
        Args:
            dataset: Dataset to process.
        
        Returns:
            DatasetSummaryView | None returned by this function.
        """
        if dataset is None:
            return None

        validation = dataset.validation

        if validation is None:
            issues: tuple[DatasetIssueView, ...] = ()
            error_count = 0
            warning_count = 0
            information_count = 0
            sampling_view = None
        else:
            issues = tuple(
                self._dataset_issue_view(
                    issue
                )
                for issue in validation.issues
            )
            error_count = len(
                validation.errors
            )
            warning_count = len(
                validation.warnings
            )
            information_count = len(
                validation.information
            )

            sampling = validation.sampling
            sampling_view = SamplingSummaryView(
                available=bool(
                    sampling.available
                ),
                sample_count=int(
                    sampling.sample_count
                ),
                duration_seconds=sampling.duration_seconds,
                median_interval_seconds=(
                    sampling.median_interval_seconds
                ),
                mean_interval_seconds=(
                    sampling.mean_interval_seconds
                ),
                std_interval_seconds=(
                    sampling.std_interval_seconds
                ),
                estimated_sampling_rate_hz=(
                    sampling.estimated_sampling_rate_hz
                ),
                duplicate_timestamp_count=int(
                    sampling.duplicate_timestamp_count
                ),
                backward_timestamp_count=int(
                    sampling.backward_timestamp_count
                ),
                large_gap_count=int(
                    sampling.large_gap_count
                ),
                largest_gap_seconds=(
                    sampling.largest_gap_seconds
                ),
                coefficient_of_variation=(
                    sampling.coefficient_of_variation
                ),
            )

        sensor_type = dataset.sensor_type

        metadata = (
            dataset.metadata.to_dict()
            if dataset.metadata is not None
            else {}
        )

        return DatasetSummaryView(
            source_file_name=(
                dataset.source_file_name
            ),
            sensor_type=(
                sensor_type.value
                if isinstance(sensor_type, Enum)
                else (
                    str(sensor_type)
                    if sensor_type is not None
                    else None
                )
            ),
            row_count=int(
                dataset.row_count
            ),
            column_count=int(
                dataset.column_count
            ),
            columns=tuple(
                dataset.columns
            ),
            units=dict(
                dataset.units
            ),
            valid=dataset.is_valid,
            error_count=error_count,
            warning_count=warning_count,
            information_count=information_count,
            issues=issues,
            sampling=sampling_view,
            metadata=_plain_value(
                metadata
            ),
        )

    @staticmethod
    def _dataset_issue_view(
        issue: DatasetIssue,
    ) -> DatasetIssueView:

        """Return dataset issue view.
        
        Args:
            issue: Issue used by this function.
        
        Returns:
            DatasetIssueView returned by this function.
        """
        return DatasetIssueView(
            issue_id=issue.issue_id,
            severity=issue.severity.value,
            message=issue.message,
            column=issue.column,
            affected_rows=issue.affected_rows,
            affected_fraction=issue.affected_fraction,
            details=_plain_value(
                issue.details
            ),
        )

    def _analysis_view(
        self,
        result: AnalysisResult,
        *,
        qualification_status: QualificationStatus,
    ) -> AnalysisSummaryView:

        """Return analysis view.
        
        Args:
            result: Result object to process.
            qualification_status: Qualification status used by this function.
        
        Returns:
            AnalysisSummaryView returned by this function.
        """
        return AnalysisSummaryView(
            tool_id=result.tool_id,
            tool_name=result.tool_name,
            tool_version=result.tool_version,
            execution_status=result.status.value,
            qualification_status=(
                qualification_status.value
            ),
            metrics=tuple(
                self._metric_view(
                    metric
                )
                for metric in result.metrics
            ),
            evidence=tuple(
                self._evidence_view(
                    evidence
                )
                for evidence in result.evidence
            ),
            plots=tuple(
                self._plot_view(
                    plot
                )
                for plot in result.plots
            ),
            warnings=tuple(
                result.warnings
            ),
            messages=tuple(
                result.messages
            ),
            metadata=_plain_value(
                result.metadata
            ),
        )

    @staticmethod
    def _metric_view(
        metric: MetricResult,
    ) -> MetricView:

        """Return metric view.
        
        Args:
            metric: Metric used by this function.
        
        Returns:
            MetricView returned by this function.
        """
        value = metric.value

        if isinstance(value, np.generic):
            value = value.item()

        return MetricView(
            name=metric.name,
            value=value,
            unit=metric.unit,
            description=metric.description,
        )

    @staticmethod
    def _evidence_view(
        evidence: Evidence,
    ) -> EvidenceView:

        """Return evidence view.
        
        Args:
            evidence: Evidence used by this function.
        
        Returns:
            EvidenceView returned by this function.
        """
        return EvidenceView(
            statement=evidence.statement,
            strength=evidence.strength.value,
            supporting_metrics=tuple(
                evidence.supporting_metrics
            ),
        )

    @staticmethod
    def _plot_view(
        plot: PlotResult,
    ) -> PlotView:

        """Return plot view.
        
        Args:
            plot: Plot used by this function.
        
        Returns:
            PlotView returned by this function.
        """
        return PlotView(
            plot_id=plot.plot_id,
            title=plot.title,
            plot_type=plot.plot_type,
            description=plot.description,
            file_path=plot.file_path,
        )

    @staticmethod
    def _qualification_status_by_tool(
        qualification: QualificationSummary | None,
    ) -> dict[str, QualificationStatus]:

        """Return qualification status by tool.
        
        Args:
            qualification: Qualification used by this function.
        
        Returns:
            Dictionary containing the result values.
        """
        if qualification is None:
            return {}

        return {
            qualified.analysis.tool_id: (
                qualified.overall_status()
            )
            for qualified in qualification.qualified_results
        }

    def _qualification_view(
        self,
        qualification: QualificationSummary | None,
    ) -> QualificationSummaryView | None:

        """Return qualification view.
        
        Args:
            qualification: Qualification used by this function.
        
        Returns:
            QualificationSummaryView | None returned by this function.
        """
        if qualification is None:
            return None

        checks: list[
            RequirementCheckView
        ] = []

        for qualified in qualification.qualified_results:

            analysis = qualified.analysis

            for check in qualified.requirement_checks:
                checks.append(
                    self._requirement_check_view(
                        check,
                        tool_id=analysis.tool_id,
                        tool_name=analysis.tool_name,
                    )
                )

        return QualificationSummaryView(
            overall_status=(
                qualification.overall_status.value
            ),
            total_requirements=int(
                qualification.total_requirements
            ),
            passed=int(
                qualification.passed
            ),
            failed=int(
                qualification.failed
            ),
            not_evaluated=int(
                qualification.not_evaluated
            ),
            checks=tuple(
                checks
            ),
            messages=tuple(
                qualification.messages
            ),
        )

    @staticmethod
    def _requirement_check_view(
        check: RequirementCheck,
        *,
        tool_id: str,
        tool_name: str,
    ) -> RequirementCheckView:

        """Return requirement check view.
        
        Args:
            check: Check used by this function.
            tool_id: Registered tool identifier.
            tool_name: Tool name used by this function.
        
        Returns:
            RequirementCheckView returned by this function.
        """
        measured_value = check.measured_value
        limit_value = check.limit_value

        if isinstance(measured_value, np.generic):
            measured_value = measured_value.item()

        if isinstance(limit_value, np.generic):
            limit_value = limit_value.item()

        return RequirementCheckView(
            tool_id=tool_id,
            tool_name=tool_name,
            metric_name=check.metric_name,
            measured_value=measured_value,
            unit=check.unit,
            requirement_description=(
                check.requirement_description
            ),
            status=check.status.value,
            limit_value=limit_value,
            operator=check.operator,
            message=check.message,
        )


def build_workflow_summary(
    workflow: SensorQAWorkflowResult,
) -> WorkflowSummaryView:
    """Convenience function for application/report callers.
    
    Args:
        workflow: Completed SensorQA workflow.
    
    Returns:
        WorkflowSummaryView returned by the function.
    """

    return WorkflowSummaryBuilder().build(
        workflow
    )
