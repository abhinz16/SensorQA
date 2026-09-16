#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from sensorqa.core.application import (
    SensorQAApplication,
    SensorQAWorkflowResult,
)
from sensorqa.ingestion.column_mapper import (
    ColumnMapping,
    StandardField,
)
from sensorqa.ingestion.dataset import SensorQADataset
from sensorqa.ingestion.metadata import SensorQAMetadata
from sensorqa.presentation import (
    WorkflowSummaryBuilder,
    WorkflowSummaryView,
)


# =====================================================================
# Request / response models
# =====================================================================


@dataclass(frozen=True)
class CSVAnalysisRequest:
    """
    Application-facing request for analysis of one CSV dataset.

    The future desktop application can construct this object from the
    Upload, Column Mapping, Units, and Test Information screens without
    needing to understand DatasetBuilder's internal call contract.
    """

    source: str | Path
    column_mapping: ColumnMapping
    unit_assignments: Mapping[str | StandardField, str]
    metadata: SensorQAMetadata | None = None
    delimiter: str | None = None
    qualify: bool = True
    raise_on_build_error: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, (str, Path)):
            raise TypeError(
                "source must be a string path or pathlib.Path."
            )

        if not isinstance(self.column_mapping, ColumnMapping):
            raise TypeError(
                "column_mapping must be a ColumnMapping."
            )

        if not isinstance(self.unit_assignments, Mapping):
            raise TypeError(
                "unit_assignments must be a mapping."
            )

        if self.metadata is not None and not isinstance(
            self.metadata,
            SensorQAMetadata,
        ):
            raise TypeError(
                "metadata must be SensorQAMetadata or None."
            )

        if self.delimiter is not None:
            if not isinstance(self.delimiter, str):
                raise TypeError(
                    "delimiter must be a string or None."
                )

            if not self.delimiter:
                raise ValueError(
                    "delimiter must not be an empty string."
                )

        if not isinstance(self.qualify, bool):
            raise TypeError(
                "qualify must be a boolean."
            )

        if not isinstance(self.raise_on_build_error, bool):
            raise TypeError(
                "raise_on_build_error must be a boolean."
            )

    @property
    def source_path(self) -> Path:
        """Resolved local path selected by the application user."""

        return Path(self.source).expanduser().resolve()

    def build_kwargs(self) -> dict[str, object]:
        """
        Convert the application request to DatasetBuilder arguments.

        Keeping this conversion here prevents GUI code from depending on
        DatasetBuilder's keyword names.
        """

        kwargs: dict[str, object] = {
            "column_mapping": self.column_mapping,
            "unit_assignments": dict(self.unit_assignments),
        }

        if self.metadata is not None:
            kwargs["metadata"] = self.metadata

        if self.delimiter is not None:
            kwargs["delimiter"] = self.delimiter

        return kwargs


@dataclass(frozen=True)
class AnalysisServiceResult:
    """
    Complete result returned to an application presentation layer.

    ``workflow`` keeps the full backend result available for advanced views
    and plots, while ``summary`` is the compact presentation-ready model that
    normal UI components and reports should consume.
    """

    success: bool
    workflow: SensorQAWorkflowResult
    summary: WorkflowSummaryView
    source_file: str | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def analysis_count(self) -> int:
        return self.summary.analysis_count

    @property
    def qualification_available(self) -> bool:
        return self.summary.qualification is not None


# =====================================================================
# Application-facing service
# =====================================================================


class AnalysisService:
    """
    High-level analysis façade for the future SensorQA desktop application.

    UI code should not coordinate DatasetBuilder, AnalysisPipeline,
    RequirementsEngine, or WorkflowSummaryBuilder itself.  It supplies a
    request to this service and receives both the complete backend workflow
    and a presentation-ready summary.
    """

    def __init__(
        self,
        application: SensorQAApplication,
        *,
        summary_builder: WorkflowSummaryBuilder | None = None,
    ) -> None:
        if not isinstance(application, SensorQAApplication):
            raise TypeError(
                "application must be a SensorQAApplication."
            )

        if not application.initialized:
            raise RuntimeError(
                "AnalysisService requires an initialized "
                "SensorQAApplication. Use bootstrap_application() first."
            )

        if summary_builder is not None and not isinstance(
            summary_builder,
            WorkflowSummaryBuilder,
        ):
            raise TypeError(
                "summary_builder must be a WorkflowSummaryBuilder or None."
            )

        self.application = application
        self.summary_builder = (
            summary_builder
            if summary_builder is not None
            else WorkflowSummaryBuilder()
        )

    def analyze_csv(
        self,
        request: CSVAnalysisRequest,
    ) -> AnalysisServiceResult:
        """
        Build and analyze one CSV selected by the application user.
        """

        if not isinstance(request, CSVAnalysisRequest):
            raise TypeError(
                "request must be a CSVAnalysisRequest."
            )

        workflow = self.application.analyze_csv(
            source=request.source_path,
            build_kwargs=request.build_kwargs(),
            qualify=request.qualify,
            raise_on_build_error=request.raise_on_build_error,
        )

        return self._build_service_result(
            workflow=workflow,
            source_file=str(request.source_path),
        )

    def analyze_dataset(
        self,
        dataset: SensorQADataset,
        *,
        qualify: bool = True,
    ) -> AnalysisServiceResult:
        """
        Analyze an already-built SensorQADataset.

        This path will be useful when the future application offers an
        interactive ingestion wizard that previews/builds the dataset before
        the user presses the final Analyze button.
        """

        if not isinstance(dataset, SensorQADataset):
            raise TypeError(
                "dataset must be a SensorQADataset."
            )

        if not isinstance(qualify, bool):
            raise TypeError(
                "qualify must be a boolean."
            )

        workflow = self.application.analyze_dataset(
            dataset=dataset,
            qualify=qualify,
        )

        source_file = None

        metadata = getattr(dataset, "metadata", None)
        dataset_source = getattr(metadata, "dataset_source", None)

        if dataset_source is not None:
            source_file = getattr(
                dataset_source,
                "file_name",
                None,
            )

        return self._build_service_result(
            workflow=workflow,
            source_file=source_file,
        )

    def _build_service_result(
        self,
        *,
        workflow: SensorQAWorkflowResult,
        source_file: str | None,
    ) -> AnalysisServiceResult:
        """Create one consistent UI-facing result from a workflow."""

        summary = self.summary_builder.build(
            workflow
        )

        warnings = self._unique_messages(
            *summary.workflow_warnings,
            *summary.pipeline_warnings,
        )

        errors: tuple[str, ...] = ()

        if not workflow.success:
            tool_errors: list[str] = []

            for result in workflow.analyses:
                status = getattr(result.status, "value", str(result.status))

                if status != "error":
                    continue

                details = [
                    str(message).strip()
                    for message in getattr(result, "messages", [])
                    if str(message).strip()
                ]

                if details:
                    tool_errors.append(
                        f"{result.tool_name}: " + " ".join(details)
                    )
                else:
                    tool_errors.append(
                        f"{result.tool_name}: analysis failed."
                    )

            errors = self._unique_messages(
                *summary.pipeline_errors,
                *summary.workflow_messages,
                *tool_errors,
            )

            if not errors:
                errors = (
                    "SensorQA analysis did not complete successfully.",
                )

        return AnalysisServiceResult(
            success=workflow.success,
            workflow=workflow,
            summary=summary,
            source_file=source_file,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def _unique_messages(
        *messages: str,
    ) -> tuple[str, ...]:
        """Deduplicate messages while preserving their original order."""

        seen: set[str] = set()
        result: list[str] = []

        for message in messages:
            text = str(message).strip()

            if not text or text in seen:
                continue

            seen.add(text)
            result.append(text)

        return tuple(result)
