#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from sensorqa.ingestion.column_mapper import (
    ColumnMapper,
    ColumnMapping,
    StandardField,
)
from sensorqa.ingestion.csv_loader import CSVLoader, CSVMetadata
from sensorqa.ingestion.dataset import SensorQADataset
from sensorqa.ingestion.dataset_builder import DatasetBuildResult, DatasetBuilder
from sensorqa.ingestion.metadata import SensorQAMetadata
from sensorqa.ingestion.unit_manager import UnitDimension, UnitManager


@dataclass(frozen=True)
class CSVPreviewRequest:
    """Request used by the application's upload/preview screen."""

    source: str | Path
    delimiter: str | None = None
    preview_rows: int = 20

    def __post_init__(self) -> None:
        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(self.source, (str, Path)):
            raise TypeError("source must be a string path or pathlib.Path.")
        if self.delimiter is not None and (
            not isinstance(self.delimiter, str) or not self.delimiter
        ):
            raise ValueError("delimiter must be a non-empty string or None.")
        if not isinstance(self.preview_rows, int) or self.preview_rows < 1:
            raise ValueError("preview_rows must be a positive integer.")

    @property
    def source_path(self) -> Path:
        """Return the resolved source-file path.
        
        Returns:
            Resolved path.
        """
        return Path(self.source).expanduser().resolve()


@dataclass(frozen=True)
class CSVPreviewResult:
    """UI-ready inspection result for one uploaded CSV."""

    success: bool
    source_file: str
    preview: pd.DataFrame | None = None
    metadata: CSVMetadata | None = None
    suggested_mapping: ColumnMapping | None = None
    required_unit_fields: tuple[str, ...] = ()
    optional_unit_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the columns available in the previewed file.
        
        Returns:
            Tuple containing the calculated values.
        """
        if self.metadata is None:
            return ()
        return tuple(self.metadata.columns)


@dataclass(frozen=True)
class DatasetBuildRequest:
    """Request for creating a SensorQADataset."""

    source: str | Path
    column_mapping: ColumnMapping
    unit_assignments: Mapping[str | StandardField, str]
    metadata: SensorQAMetadata | None = None
    delimiter: str | None = None

    def __post_init__(self) -> None:
        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(self.source, (str, Path)):
            raise TypeError("source must be a string path or pathlib.Path.")
        if not isinstance(self.column_mapping, ColumnMapping):
            raise TypeError("column_mapping must be a ColumnMapping.")
        if not isinstance(self.unit_assignments, Mapping):
            raise TypeError("unit_assignments must be a mapping.")
        if self.metadata is not None and not isinstance(
            self.metadata, SensorQAMetadata
        ):
            raise TypeError("metadata must be SensorQAMetadata or None.")
        if self.delimiter is not None and (
            not isinstance(self.delimiter, str) or not self.delimiter
        ):
            raise ValueError("delimiter must be a non-empty string or None.")

    @property
    def source_path(self) -> Path:
        """Return the resolved source-file path.
        
        Returns:
            Resolved path.
        """
        return Path(self.source).expanduser().resolve()


@dataclass(frozen=True)
class DatasetBuildServiceResult:
    """Structured build result for the application's ingestion wizard."""

    success: bool
    analysis_ready: bool
    dataset: SensorQADataset | None
    failed_stage: str | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


class IngestionService:
    """High-level upload, preview, mapping, and dataset-build service."""

    def __init__(
        self,
        *,
        csv_loader: CSVLoader | None = None,
        column_mapper: ColumnMapper | None = None,
        unit_manager: UnitManager | None = None,
        dataset_builder: DatasetBuilder | None = None,
    ) -> None:
        """Initialize the ingestion service.
        
        Args:
            csv_loader: Csv loader used by this function.
            column_mapper: Column mapper used by this function.
            unit_manager: Unit manager used by this function.
            dataset_builder: Dataset builder used by this function.
        
        Returns:
            None.
        """
        self.csv_loader = csv_loader or CSVLoader()
        self.column_mapper = column_mapper or ColumnMapper()
        self.unit_manager = unit_manager or UnitManager()
        self.dataset_builder = dataset_builder or DatasetBuilder(
            csv_loader=self.csv_loader,
            column_mapper=self.column_mapper,
            unit_manager=self.unit_manager,
        )

    def preview_csv(self, request: CSVPreviewRequest) -> CSVPreviewResult:
        """Load a small CSV preview for the import workflow.
        
        Args:
            request: Request object containing the user inputs.
        
        Returns:
            CSVPreviewResult returned by this function.
        """
        if not isinstance(request, CSVPreviewRequest):
            raise TypeError("request must be a CSVPreviewRequest.")

        source_path = request.source_path
        load_result = self.csv_loader.load(
            source=source_path,
            delimiter=request.delimiter,
        )

        if (
            not load_result.success
            or load_result.data is None
            or load_result.metadata is None
        ):
            return CSVPreviewResult(
                success=False,
                source_file=str(source_path),
                warnings=tuple(load_result.warnings),
                errors=tuple(load_result.errors),
            )

        mapping = self.column_mapper.infer_mapping(
            load_result.data.columns
        )

        required_units, optional_units = self._unit_fields_for_mapping(
            mapping
        )

        warnings = list(load_result.warnings)
        warnings.extend(mapping.warnings)

        return CSVPreviewResult(
            success=True,
            source_file=str(source_path),
            preview=self.csv_loader.preview(
                load_result,
                rows=request.preview_rows,
            ),
            metadata=load_result.metadata,
            suggested_mapping=mapping,
            required_unit_fields=required_units,
            optional_unit_fields=optional_units,
            warnings=tuple(self._unique(warnings)),
            errors=(),
        )

    def infer_mapping(
        self,
        columns,
        *,
        explicit_mapping: ColumnMapping | Mapping | None = None,
    ) -> ColumnMapping:
        """Suggest SensorQA field mappings from source-column names.
        
        Args:
            columns: Columns used by this function.
            explicit_mapping: Explicit mapping used by this function.
        
        Returns:
            ColumnMapping returned by this function.
        """
        return self.column_mapper.infer_mapping(
            columns,
            explicit_mapping=explicit_mapping,
        )

    def build_dataset(
        self,
        request: DatasetBuildRequest,
    ) -> DatasetBuildServiceResult:
        """Build dataset.
        
        Args:
            request: Request object containing the user inputs.
        
        Returns:
            DatasetBuildServiceResult returned by this function.
        """
        if not isinstance(request, DatasetBuildRequest):
            raise TypeError("request must be a DatasetBuildRequest.")

        result: DatasetBuildResult = self.dataset_builder.build(
            source=request.source_path,
            column_mapping=request.column_mapping,
            unit_assignments=dict(request.unit_assignments),
            metadata=request.metadata,
            delimiter=request.delimiter,
        )

        return DatasetBuildServiceResult(
            success=result.success,
            analysis_ready=result.analysis_ready,
            dataset=result.dataset,
            failed_stage=result.failed_stage,
            warnings=tuple(result.warnings),
            errors=tuple(result.errors),
        )

    def unit_fields_for_mapping(
        self,
        mapping: ColumnMapping,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Run unit fields for mapping.
        
        Args:
            mapping: Mapping used by this function.
        
        Returns:
            Tuple containing the calculated values.
        """
        if not isinstance(mapping, ColumnMapping):
            raise TypeError("mapping must be a ColumnMapping.")
        return self._unit_fields_for_mapping(mapping)

    def _unit_fields_for_mapping(
        self,
        mapping: ColumnMapping,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Calculate unit fields for mapping.
        
        Args:
            mapping: Mapping used by this function.
        
        Returns:
            Tuple containing the calculated values.
        """
        required: list[str] = []
        optional: list[str] = []

        for field_ in mapping.mapped_fields:
            field_name = field_.value
            dimension = self.unit_manager.dimension_for_field(field_name)

            if dimension == UnitDimension.DIMENSIONLESS:
                continue

            if dimension == UnitDimension.UNKNOWN:
                optional.append(field_name)
            else:
                required.append(field_name)

        return tuple(sorted(required)), tuple(sorted(optional))

    @staticmethod
    def _unique(messages) -> list[str]:
        """Return unique.
        
        Args:
            messages: Messages used by this function.
        
        Returns:
            List of result values.
        """
        seen: set[str] = set()
        result: list[str] = []
        for message in messages:
            text = str(message).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result
