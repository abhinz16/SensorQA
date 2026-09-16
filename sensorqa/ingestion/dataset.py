#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sensorqa.ingestion.column_mapper import ColumnMapping
from sensorqa.ingestion.dataset_validator import DatasetValidationResult
from sensorqa.ingestion.metadata import SensorQAMetadata
from sensorqa.ingestion.unit_manager import UnitConversionRecord


@dataclass
class DatasetProcessingRecord:
    """
    Records one preprocessing or ingestion operation applied
    while constructing a SensorQADataset.

    Examples:
        "Loaded CSV"
        "Mapped columns"
        "Converted acceleration from g to m/s^2"
        "Validated dataset"
    """

    step: str
    description: str

    details: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class SensorQADataset:
    """
    Canonical dataset object used throughout SensorQA.

    Once ingestion is complete, analysis tools should normally
    receive data originating from this object rather than working
    directly with the raw uploaded CSV.
    """

    # Main normalized dataset

    data: pd.DataFrame

    # Mapping between SensorQA fields and original CSV columns

    column_mapping: ColumnMapping

    # Current units used by normalized data

    units: dict[str, str]

    # Original units before SensorQA normalization

    original_units: dict[str, str] = field(
        default_factory=dict
    )

    # Sensor and experiment metadata

    metadata: SensorQAMetadata | None = None

    # Dataset quality validation

    validation: DatasetValidationResult | None = None

    # Unit conversions performed during ingestion

    unit_conversions: list[UnitConversionRecord] = field(
        default_factory=list
    )

    # Complete ingestion / processing history

    processing_history: list[DatasetProcessingRecord] = field(
        default_factory=list
    )

    # Optional information from the loading stage

    source_file_name: str | None = None
    source_file_path: str | None = None

    # Arbitrary additional information

    extra: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Basic structural validation when a SensorQADataset
        is created.
        
        Returns:
            None.
        """

        if not isinstance(
            self.data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if not isinstance(
            self.column_mapping,
            ColumnMapping,
        ):
            raise TypeError(
                "column_mapping must be a ColumnMapping object."
            )

        if not isinstance(
            self.units,
            dict,
        ):
            raise TypeError(
                "units must be a dictionary."
            )

    # Convenience properties

    @property
    def row_count(self) -> int:
        """Number of measurement rows.
        
        Returns:
            Integer result.
        """

        return len(
            self.data
        )

    @property
    def column_count(self) -> int:
        """Number of columns available to analysis tools.
        
        Returns:
            Integer result.
        """

        return len(
            self.data.columns
        )

    @property
    def columns(self) -> list[str]:
        """List of normalized SensorQA column names.
        
        Returns:
            List of result values.
        """

        return [
            str(column)
            for column in self.data.columns
        ]

    @property
    def is_valid(self) -> bool | None:
        """
        Return dataset validation state.

        Returns:
            True  -> validation passed
            False -> validation failed
            None  -> validation has not yet been performed
        """

        if self.validation is None:
            return None

        return self.validation.valid

    @property
    def sensor_type(self):
        """Return sensor type from metadata when available.
        
        Returns:
            Result returned by the function.
        """

        if self.metadata is None:
            return None

        return self.metadata.sensor.sensor_type

    # Column access

    def has_column(
        self,
        column: str,
    ) -> bool:
        """Return True if the normalized dataset contains a column.
        
        Args:
            column: Column name.
        
        Returns:
            Boolean result.
        """

        return column in self.data.columns

    def get_column(
        self,
        column: str,
    ) -> pd.Series:
        """Return one normalized dataset column.
        
        Raises KeyError if the column does not exist.
        
        Args:
            column: Column name.
        
        Returns:
            pd.Series returned by the function.
        """

        if column not in self.data.columns:
            raise KeyError(
                f"Dataset does not contain column '{column}'."
            )

        return self.data[
            column
        ]

    def get_columns(
        self,
        columns: list[str],
    ) -> pd.DataFrame:
        """Return a subset of normalized columns.
        
        Raises KeyError if any requested column is missing.
        
        Args:
            columns: Value for `columns`.
        
        Returns:
            DataFrame containing the requested data.
        """

        missing = [
            column
            for column in columns
            if column not in self.data.columns
        ]

        if missing:
            raise KeyError(
                "Dataset does not contain required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        return self.data[
            columns
        ].copy()

    # Unit access

    def get_unit(
        self,
        field_name: str,
    ) -> str | None:
        """Return the current normalized unit for a field.
        
        Args:
            field_name: Value for `field_name`.
        
        Returns:
            str | None returned by the function.
        """

        return self.units.get(
            field_name
        )

    def get_original_unit(
        self,
        field_name: str,
    ) -> str | None:
        """Return the unit supplied by the user before normalization.
        
        Args:
            field_name: Value for `field_name`.
        
        Returns:
            str | None returned by the function.
        """

        return self.original_units.get(
            field_name
        )

    # Metadata access

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve custom metadata stored at dataset level.
        
        Args:
            key: Value for `key`.
            default: Value for `default`.
        
        Returns:
            Any returned by the function.
        """

        if key in self.extra:
            return self.extra[
                key
            ]

        if self.metadata is not None:
            return self.metadata.get(
                key,
                default,
            )

        return default

    # Processing history

    def add_processing_record(
        self,
        step: str,
        description: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record an ingestion or processing operation.
        
        Args:
            step: Value for `step`.
            description: Value for `description`.
            details: Value for `details`.
        
        Returns:
            None.
        """

        if not isinstance(
            step,
            str,
        ) or not step.strip():

            raise ValueError(
                "Processing step must be a non-empty string."
            )

        if not isinstance(
            description,
            str,
        ) or not description.strip():

            raise ValueError(
                "Processing description must be a non-empty string."
            )

        self.processing_history.append(
            DatasetProcessingRecord(
                step=step,
                description=description,
                details=details or {},
            )
        )

    # Dataset summary

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary suitable for the API,
        frontend, logs, and reports.
        
        Returns:
            Dictionary containing the result values.
        """

        validation_summary: dict[str, Any] | None = None

        if self.validation is not None:

            validation_summary = {
                "valid": self.validation.valid,
                "error_count": len(
                    self.validation.errors
                ),
                "warning_count": len(
                    self.validation.warnings
                ),
                "info_count": len(
                    self.validation.information
                ),
            }

        sensor_type_value = None

        if self.sensor_type is not None:
            sensor_type_value = (
                self.sensor_type.value
                if hasattr(
                    self.sensor_type,
                    "value",
                )
                else str(
                    self.sensor_type
                )
            )

        return {
            "source_file_name": self.source_file_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "sensor_type": sensor_type_value,
            "units": dict(
                self.units
            ),
            "validation": validation_summary,
            "processing_steps": len(
                self.processing_history
            ),
        }

    # Analysis data copy

    def analysis_view(
        self,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a defensive copy of data for analysis.
        
        Tools should generally avoid modifying the canonical
        SensorQADataset DataFrame directly.
        
        Args:
            columns: Value for `columns`.
        
        Returns:
            DataFrame containing the requested data.
        """

        if columns is None:

            return self.data.copy()

        return self.get_columns(
            columns
        )
