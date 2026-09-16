#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import pandas as pd

from sensorqa.ingestion.column_mapper import (
    ColumnMapper,
    ColumnMapping,
    StandardField,
)
from sensorqa.ingestion.csv_loader import CSVLoader
from sensorqa.ingestion.dataset import SensorQADataset
from sensorqa.ingestion.dataset_validator import DatasetValidator
from sensorqa.ingestion.metadata import SensorQAMetadata
from sensorqa.ingestion.unit_manager import (
    UnitDimension,
    UnitManager,
)


@dataclass
class DatasetBuildResult:
    """
    Result of constructing a SensorQADataset.

    success:
        True means SensorQA successfully created the dataset object.

    analysis_ready:
        True means the dataset also passed data-quality validation.

    A dataset may therefore be built successfully but still not be
    ready for analysis because validation found serious problems.
    """

    success: bool

    dataset: SensorQADataset | None = None

    analysis_ready: bool = False

    failed_stage: str | None = None

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class DatasetBuilder:
    """
    Coordinates the complete SensorQA ingestion workflow.

    Workflow:

        CSV file
            ↓
        CSVLoader
            ↓
        ColumnMapper
            ↓
        UnitManager
            ↓
        DatasetValidator
            ↓
        SensorQADataset
    """

    def __init__(
        self,
        csv_loader: CSVLoader | None = None,
        column_mapper: ColumnMapper | None = None,
        unit_manager: UnitManager | None = None,
        dataset_validator: DatasetValidator | None = None,
    ) -> None:

        """Initialize the dataset builder.
        
        Args:
            csv_loader: Csv loader used by this function.
            column_mapper: Column mapper used by this function.
            unit_manager: Unit manager used by this function.
            dataset_validator: Dataset validator used by this function.
        
        Returns:
            None.
        """
        self.csv_loader = csv_loader or CSVLoader()

        self.column_mapper = (
            column_mapper
            or ColumnMapper()
        )

        self.unit_manager = (
            unit_manager
            or UnitManager()
        )

        self.dataset_validator = (
            dataset_validator
            or DatasetValidator()
        )

    def build(
        self,
        source: str | Path | IO,
        column_mapping: ColumnMapping,
        unit_assignments: dict[str | StandardField, str],
        metadata: SensorQAMetadata | None = None,
        delimiter: str | None = None,
    ) -> DatasetBuildResult:
        """Build a complete SensorQADataset from an uploaded CSV.
        
        Parameters
        ----------
        source:
            CSV path or readable uploaded-file object.
        
        column_mapping:
            Mapping from SensorQA standard fields to the user's
            original CSV column names.
        
        unit_assignments:
            Units associated with mapped fields.
        
            Example:
        
            {
                "timestamp": "ms",
                "ax": "g",
                "ay": "g",
                "az": "g",
                "gx": "deg/s",
                "gy": "deg/s",
                "gz": "deg/s",
                "temperature": "degC"
            }
        
        metadata:
            Optional sensor/test metadata.
        
        delimiter:
            Optional manual CSV delimiter override.
        
        Args:
            source: Input source or source path.
            column_mapping: Mapping from SensorQA fields to source columns.
            unit_assignments: Engineering units assigned to mapped fields.
            metadata: Metadata associated with the dataset or tool.
            delimiter: Value for `delimiter`.
        
        Returns:
            DatasetBuildResult returned by the function.
        """

        warnings: list[str] = []
        errors: list[str] = []

        # STEP 1: Load CSV

        load_result = self.csv_loader.load(
            source=source,
            delimiter=delimiter,
        )

        warnings.extend(
            load_result.warnings
        )

        if (
            not load_result.success
            or load_result.data is None
            or load_result.metadata is None
        ):

            errors.extend(
                load_result.errors
            )

            return DatasetBuildResult(
                success=False,
                analysis_ready=False,
                failed_stage="csv_loading",
                warnings=warnings,
                errors=errors,
            )

        raw_data = load_result.data

        # STEP 2: Validate column mapping

        mapping_validation = self.column_mapper.validate(
            data=raw_data,
            mapping=column_mapping,
        )

        warnings.extend(
            mapping_validation.warnings
        )

        if not mapping_validation.valid:

            errors.extend(
                mapping_validation.errors
            )

            return DatasetBuildResult(
                success=False,
                analysis_ready=False,
                failed_stage="column_mapping",
                warnings=warnings,
                errors=errors,
            )

        # One physical dataset column should not represent two
        # different SensorQA concepts because the mapped DataFrame
        # cannot safely assign two meanings to one column.

        if mapping_validation.duplicated_dataset_columns:

            duplicate_messages = []

            for (
                dataset_column,
                sensorqa_fields,
            ) in mapping_validation.duplicated_dataset_columns.items():

                duplicate_messages.append(
                    f"Dataset column '{dataset_column}' is mapped "
                    f"to multiple SensorQA fields: "
                    + ", ".join(
                        sorted(sensorqa_fields)
                    )
                )

            errors.extend(
                duplicate_messages
            )

            return DatasetBuildResult(
                success=False,
                analysis_ready=False,
                failed_stage="column_mapping",
                warnings=warnings,
                errors=errors,
            )

        # STEP 3: Create SensorQA-standard mapped dataset

        try:

            mapped_data = self.column_mapper.create_mapped_view(
                data=raw_data,
                mapping=column_mapping,
                keep_unmapped_columns=True,
            )

        except Exception as exc:

            return DatasetBuildResult(
                success=False,
                analysis_ready=False,
                failed_stage="column_mapping",
                warnings=warnings,
                errors=[
                    (
                        "Failed to create mapped dataset: "
                        f"{type(exc).__name__}: {exc}"
                    )
                ],
            )

        # STEP 4: Normalize unit-assignment keys

        normalized_unit_assignments = (
            self._normalize_unit_assignment_keys(
                unit_assignments
            )
        )

        # STEP 5: Verify required units are provided

        missing_unit_errors = (
            self._check_required_unit_assignments(
                mapped_data=mapped_data,
                unit_assignments=normalized_unit_assignments,
            )
        )

        if missing_unit_errors:

            errors.extend(
                missing_unit_errors
            )

            return DatasetBuildResult(
                success=False,
                analysis_ready=False,
                failed_stage="unit_assignment",
                warnings=warnings,
                errors=errors,
            )

        # STEP 6: Normalize known physical units

        normalization_result = (
            self.unit_manager.normalize_dataframe(
                data=mapped_data,
                unit_assignments=normalized_unit_assignments,
            )
        )

        warnings.extend(
            normalization_result.warnings
        )

        if normalization_result.errors:

            errors.extend(
                normalization_result.errors
            )

            return DatasetBuildResult(
                success=False,
                analysis_ready=False,
                failed_stage="unit_normalization",
                warnings=warnings,
                errors=errors,
            )

        normalized_data = (
            normalization_result.data
        )

        # STEP 7: Determine expected numeric channels

        numeric_columns = (
            self._determine_numeric_columns(
                normalized_data
            )
        )

        # STEP 8: Run dataset quality validation

        validation_result = (
            self.dataset_validator.validate(
                data=normalized_data,
                numeric_columns=numeric_columns,
                timestamp_column=StandardField.TIMESTAMP.value,
            )
        )

        # STEP 9: Synchronize source metadata

        if metadata is not None:

            if (
                metadata.dataset_source.file_name
                is None
            ):

                metadata.dataset_source.file_name = (
                    load_result.metadata.file_name
                )

        # STEP 10: Construct canonical SensorQADataset

        original_units = {
            field_name: unit
            for field_name, unit
            in normalized_unit_assignments.items()
        }

        dataset = SensorQADataset(
            data=normalized_data,
            column_mapping=column_mapping,
            units=normalization_result.units,
            original_units=original_units,
            metadata=metadata,
            validation=validation_result,
            unit_conversions=normalization_result.conversions,
            source_file_name=load_result.metadata.file_name,
            source_file_path=load_result.metadata.file_path,
        )

        # STEP 11: Record processing history

        dataset.add_processing_record(
            step="csv_loading",
            description="Loaded sensor dataset from CSV.",
            details={
                "delimiter":
                    load_result.metadata.delimiter,
                "rows":
                    load_result.metadata.row_count,
                "columns":
                    load_result.metadata.column_count,
            },
        )

        dataset.add_processing_record(
            step="column_mapping",
            description=(
                "Mapped user dataset columns to "
                "SensorQA standard fields."
            ),
            details={
                "mapping":
                    column_mapping.to_dict(),
            },
        )

        dataset.add_processing_record(
            step="unit_normalization",
            description=(
                "Normalized known physical quantities "
                "to SensorQA canonical units."
            ),
            details={
                "conversions": [
                    {
                        "field":
                            conversion.field,
                        "source_unit":
                            conversion.source_unit,
                        "target_unit":
                            conversion.target_unit,
                        "converted":
                            conversion.converted,
                    }
                    for conversion
                    in normalization_result.conversions
                ]
            },
        )

        dataset.add_processing_record(
            step="dataset_validation",
            description=(
                "Performed structural and numerical "
                "data-quality validation."
            ),
            details={
                "valid":
                    validation_result.valid,
                "error_count":
                    len(validation_result.errors),
                "warning_count":
                    len(validation_result.warnings),
                "info_count":
                    len(validation_result.information),
            },
        )

        # STEP 12: Convert validation findings into build warnings

        for issue in validation_result.warnings:

            warnings.append(
                issue.message
            )

        # Error-level data-quality findings do not prevent SensorQA
        # from constructing the dataset. They prevent it from being
        # considered analysis-ready.

        for issue in validation_result.errors:

            warnings.append(
                f"Data-quality error: {issue.message}"
            )

        return DatasetBuildResult(
            success=True,
            dataset=dataset,
            analysis_ready=validation_result.valid,
            failed_stage=None,
            warnings=warnings,
            errors=[],
        )

    # Unit-assignment helpers

    @staticmethod
    def _normalize_unit_assignment_keys(
        unit_assignments: dict[
            str | StandardField,
            str
        ],
    ) -> dict[str, str]:
        """Convert StandardField keys into their string values.
        
        Example:
        
            StandardField.GX -> "gx"
        
        Args:
            unit_assignments: Engineering units assigned to mapped fields.
        
        Returns:
            Dictionary containing the result values.
        """

        normalized: dict[str, str] = {}

        for field_name, unit in unit_assignments.items():

            if isinstance(
                field_name,
                StandardField,
            ):

                key = field_name.value

            elif isinstance(
                field_name,
                str,
            ):

                key = field_name.strip()

            else:

                raise TypeError(
                    "Unit-assignment keys must be "
                    "StandardField values or strings."
                )

            if not key:

                raise ValueError(
                    "Unit-assignment field name "
                    "cannot be empty."
                )

            normalized[key] = unit

        return normalized

    def _check_required_unit_assignments(
        self,
        mapped_data: pd.DataFrame,
        unit_assignments: dict[str, str],
    ) -> list[str]:
        """Require explicit units for known physical quantities.
        
        SensorQA should not assume, for example, whether gx means
        deg/s or rad/s.
        
        Args:
            mapped_data: Value for `mapped_data`.
            unit_assignments: Engineering units assigned to mapped fields.
        
        Returns:
            List of result values.
        """

        errors: list[str] = []

        for column in mapped_data.columns:

            dimension = (
                self.unit_manager.dimension_for_field(
                    str(column)
                )
            )

            if dimension in {
                UnitDimension.UNKNOWN,
                UnitDimension.DIMENSIONLESS,
            }:
                continue

            if column not in unit_assignments:

                errors.append(
                    f"Unit assignment is required for "
                    f"'{column}' "
                    f"({dimension.value})."
                )

        return errors

    # Numeric-channel helper

    @staticmethod
    def _determine_numeric_columns(
        data: pd.DataFrame,
    ) -> list[str]:
        """Determine fields that SensorQA expects to contain
        numerical values.
        
        This explicitly includes known engineering fields even if
        pandas initially loaded them as object/string columns.
        
        Args:
            data: Input data to process.
        
        Returns:
            List of result values.
        """

        expected_numeric_fields = {
            # Generic sensor quantities
            StandardField.REFERENCE.value,
            StandardField.MEASUREMENT.value,

            # Time / environment
            StandardField.TIMESTAMP.value,
            StandardField.TEMPERATURE.value,
            StandardField.HUMIDITY.value,
            StandardField.SUPPLY_VOLTAGE.value,

            # Accelerometer
            StandardField.AX.value,
            StandardField.AY.value,
            StandardField.AZ.value,

            # Gyroscope
            StandardField.GX.value,
            StandardField.GY.value,
            StandardField.GZ.value,

            # Reference IMU channels
            StandardField.AX_REFERENCE.value,
            StandardField.AY_REFERENCE.value,
            StandardField.AZ_REFERENCE.value,

            StandardField.GX_REFERENCE.value,
            StandardField.GY_REFERENCE.value,
            StandardField.GZ_REFERENCE.value,

            # Position
            StandardField.POSITION_X.value,
            StandardField.POSITION_Y.value,
            StandardField.POSITION_Z.value,

            # Velocity
            StandardField.VELOCITY_X.value,
            StandardField.VELOCITY_Y.value,
            StandardField.VELOCITY_Z.value,

            # Quaternion
            StandardField.QUATERNION_W.value,
            StandardField.QUATERNION_X.value,
            StandardField.QUATERNION_Y.value,
            StandardField.QUATERNION_Z.value,

            # Euler angles
            StandardField.ROLL.value,
            StandardField.PITCH.value,
            StandardField.YAW.value,
        }

        detected_numeric = set(
            str(column)
            for column
            in data.select_dtypes(
                include="number"
            ).columns
        )

        known_numeric_present = {
            field_name
            for field_name
            in expected_numeric_fields
            if field_name in data.columns
        }

        return sorted(
            detected_numeric
            | known_numeric_present
        )
