#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
SensorQA column mapping utilities.

This module defines the canonical SensorQA field vocabulary and the
column-mapping contract used by DatasetBuilder.

DatasetBuilder expects:

    validation = column_mapper.validate(
        data=dataframe,
        mapping=column_mapping,
    )

    validation.valid
    validation.errors
    validation.warnings
    validation.duplicated_dataset_columns

and:

    mapped = column_mapper.create_mapped_view(
        data=dataframe,
        mapping=column_mapping,
        keep_unmapped_columns=True,
    )

The implementation below follows that contract directly.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

import pandas as pd


# Canonical SensorQA fields


class StandardField(str, Enum):
    """
    Canonical SensorQA dataframe column names.
    """

    TIMESTAMP = "timestamp"

    SENSOR_ID = "sensor_id"

    REFERENCE = "reference"

    MEASUREMENT = "measurement"

    TEMPERATURE = "temperature"

    HUMIDITY = "humidity"

    SUPPLY_VOLTAGE = "supply_voltage"

    REVISION = "revision"

    TEST_CYCLE = "test_cycle"

    SWEEP_DIRECTION = "sweep_direction"

    # Accelerometer
    AX = "ax"
    AY = "ay"
    AZ = "az"

    # Gyroscope
    GX = "gx"
    GY = "gy"
    GZ = "gz"

    # Accelerometer reference
    AX_REFERENCE = "ax_reference"
    AY_REFERENCE = "ay_reference"
    AZ_REFERENCE = "az_reference"

    # Gyroscope reference
    GX_REFERENCE = "gx_reference"
    GY_REFERENCE = "gy_reference"
    GZ_REFERENCE = "gz_reference"

    # Position
    POSITION_X = "position_x"
    POSITION_Y = "position_y"
    POSITION_Z = "position_z"

    # Velocity
    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    VELOCITY_Z = "velocity_z"

    # Quaternion
    QUATERNION_W = "quaternion_w"
    QUATERNION_X = "quaternion_x"
    QUATERNION_Y = "quaternion_y"
    QUATERNION_Z = "quaternion_z"

    # Euler orientation
    ROLL = "roll"
    PITCH = "pitch"
    YAW = "yaw"

    # Test context
    ORIENTATION_LABEL = "orientation_label"

    MOTION_STATE = "motion_state"

    EXPERIMENT_LABEL = "experiment_label"


# Helpers


def _coerce_standard_field(
    value: StandardField | str,
) -> StandardField:
    """Convert a string or StandardField to StandardField.
    
    Args:
        value: Value to process.
    
    Returns:
        StandardField returned by the function.
    """

    if isinstance(
        value,
        StandardField,
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "SensorQA field identifiers must be strings "
            "or StandardField values."
        )

    value = value.strip()

    if not value:
        raise ValueError(
            "SensorQA field identifier must not be empty."
        )

    # Canonical value form:
    #
    #     "timestamp"
    try:

        return StandardField(
            value
        )

    except ValueError:
        pass

    # Enum-member form:
    #
    #     "TIMESTAMP"
    try:

        return StandardField[
            value.upper()
        ]

    except KeyError as exc:

        raise ValueError(
            f"Unknown SensorQA standard field: {value!r}"
        ) from exc


def _normalize_name(
    value: Any,
) -> str:
    """Normalize a source-column name for conservative alias matching.
    
    Args:
        value: Value to process.
    
    Returns:
        String representation.
    """

    text = str(
        value
    ).strip().lower()

    text = text.replace(
        "°",
        "deg",
    )

    text = text.replace(
        "%",
        "percent",
    )

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip(
        "_"
    )


# Mapping validation result


@dataclass
class ColumnMappingValidationResult:
    """
    Result returned by ColumnMapper.validate().
    """

    valid: bool

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    missing_required_fields: list[str] = field(
        default_factory=list
    )

    unmapped_columns: list[str] = field(
        default_factory=list
    )

    # Exact property used by DatasetBuilder.
    #
    # Example:
    #
    # {
    #     "sensor_column": [
    #         "ax",
    #         "ay",
    #     ]
    # }
    #
    duplicated_dataset_columns: dict[
        str,
        list[str],
    ] = field(
        default_factory=dict
    )

    resolved_mapping: "ColumnMapping | None" = None

    @property
    def success(
        self,
    ) -> bool:
        """Convenience alias.
        
        Returns:
            Boolean result.
        """

        return self.valid

    @property
    def duplicate_source_columns(
        self,
    ) -> list[str]:
        """Compatibility alias.
        
        Returns:
            List of result values.
        """

        return sorted(
            self.duplicated_dataset_columns
        )

    @property
    def missing_columns(
        self,
    ) -> list[str]:
        """Compatibility alias.
        
        Returns:
            List of result values.
        """

        return list(
            self.missing_required_fields
        )

    def __bool__(
        self,
    ) -> bool:

        """Return the truth value of this object.
        
        Returns:
            True when the condition is met; otherwise False.
        """
        return self.valid


# ColumnMapping


@dataclass
class ColumnMapping:
    """
    Mapping from SensorQA canonical field to original dataset column.

    Example:

        {
            StandardField.TIMESTAMP: "Time_s",
            StandardField.AX: "Accel_X",
        }
    """

    mapping: dict[
        StandardField | str,
        str,
    ] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    unmapped_columns: list[str] = field(
        default_factory=list
    )

    inferred: bool = False

    def __post_init__(
        self,
    ) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        normalized: dict[
            StandardField,
            str,
        ] = {}

        for raw_field, raw_column in (
            self.mapping.items()
        ):

            field_ = _coerce_standard_field(
                raw_field
            )

            if not isinstance(
                raw_column,
                str,
            ):
                raise TypeError(
                    "Mapped dataset column names must be strings."
                )

            source = raw_column.strip()

            if not source:
                raise ValueError(
                    "Mapped dataset column name must not be empty."
                )

            normalized[
                field_
            ] = source

        self.mapping = normalized

        self.warnings = list(
            self.warnings
        )

        self.unmapped_columns = list(
            self.unmapped_columns
        )

    # Basic API

    def __len__(
        self,
    ) -> int:

        """Return the number of mapped items.
        
        Returns:
            Calculated value.
        """
        return len(
            self.mapping
        )

    def __contains__(
        self,
        field_: StandardField | str,
    ) -> bool:

        """Return whether the requested item is present.
        
        Args:
            field_: Field used by this function.
        
        Returns:
            True when the condition is met; otherwise False.
        """
        try:

            field_ = _coerce_standard_field(
                field_
            )

        except (
            TypeError,
            ValueError,
        ):
            return False

        return (
            field_
            in self.mapping
        )

    def get(
        self,
        field_: StandardField | str,
        default=None,
    ):

        """Run get.
        
        Args:
            field_: Field used by this function.
            default: Default used by this function.
        
        Returns:
            Calculated value.
        """
        try:

            field_ = _coerce_standard_field(
                field_
            )

        except (
            TypeError,
            ValueError,
        ):
            return default

        return self.mapping.get(
            field_,
            default,
        )

    def source_for(
        self,
        field_: StandardField | str,
    ) -> str | None:

        """Return the source column mapped to a SensorQA field.
        
        Args:
            field_: Field used by this function.
        
        Returns:
            str | None returned by this function.
        """
        return self.get(
            field_
        )

    source_column = source_for

    def standard_for_source(
        self,
        source_column: str,
    ) -> StandardField | None:

        """Return the SensorQA field mapped from a source column.
        
        Args:
            source_column: Column used for source.
        
        Returns:
            StandardField | None returned by this function.
        """
        for field_, column in (
            self.mapping.items()
        ):

            if column == source_column:
                return field_

        return None

    @property
    def standard_to_source(
        self,
    ) -> dict[
        StandardField,
        str,
    ]:

        """Return the field-to-source mapping.
        
        Returns:
            Dictionary containing the result values.
        """
        return dict(
            self.mapping
        )

    @property
    def source_to_standard(
        self,
    ) -> dict[
        str,
        StandardField,
    ]:
        """Reverse mapping.
        
        This property is only safe when dataset columns are unique.
        Duplicate physical-column assignments are intentionally
        detected separately by validation.
        
        Returns:
            Dictionary containing the result values.
        """

        result = {}

        for field_, source in (
            self.mapping.items()
        ):

            result[
                source
            ] = field_

        return result

    @property
    def mapped_fields(
        self,
    ) -> list[
        StandardField
    ]:

        """Return the mapped SensorQA fields.
        
        Returns:
            List of result values.
        """
        return list(
            self.mapping.keys()
        )

    @property
    def mapped_columns(
        self,
    ) -> list[str]:

        """Return the mapped source columns.
        
        Returns:
            List of result values.
        """
        return list(
            self.mapping.values()
        )

    # Serialization

    def as_dict(
        self,
    ) -> dict[
        str,
        str,
    ]:

        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            field_.value:
                source
            for field_, source
            in self.mapping.items()
        }

    def to_dict(
        self,
    ) -> dict:
        """JSON-friendly mapping description.
        
        Returns:
            Dictionary containing the result values.
        """

        return {
            "mapping":
                self.as_dict(),

            "warnings":
                list(
                    self.warnings
                ),

            "unmapped_columns":
                list(
                    self.unmapped_columns
                ),

            "inferred":
                self.inferred,
        }


# Conservative aliases


DEFAULT_ALIASES: dict[
    StandardField,
    tuple[str, ...],
] = {

    StandardField.TIMESTAMP: (
        "timestamp",
        "time",
        "time_s",
        "time_sec",
        "time_secs",
        "time_seconds",
        "elapsed_time",
        "elapsed_seconds",
    ),

    StandardField.SENSOR_ID: (
        "sensor_id",
        "sensorid",
        "device_id",
        "deviceid",
    ),

    StandardField.REFERENCE: (
        "reference",
        "reference_value",
        "ground_truth",
        "groundtruth",
        "truth",
    ),

    StandardField.MEASUREMENT: (
        "measurement",
        "measured_value",
        "sensor_value",
        "sensor_output",
        "output",
    ),

    StandardField.TEMPERATURE: (
        "temperature",
        "temperature_c",
        "temperature_deg_c",
        "temp",
        "temp_c",
        "temp_deg_c",
    ),

    StandardField.HUMIDITY: (
        "humidity",
        "relative_humidity",
        "rh",
        "rh_percent",
    ),

    StandardField.SUPPLY_VOLTAGE: (
        "supply_voltage",
        "supply_v",
        "vcc",
        "vdd",
    ),

    StandardField.REVISION: (
        "revision",
        "device_revision",
        "hardware_revision",
        "hw_revision",
    ),

    StandardField.TEST_CYCLE: (
        "test_cycle",
        "cycle",
        "cycle_number",
        "test_cycle_number",
    ),

    StandardField.SWEEP_DIRECTION: (
        "sweep_direction",
        "sweep",
        "direction",
    ),

    # Accelerometer

    StandardField.AX: (
        "ax",
        "accel_x",
        "acceleration_x",
        "accelerometer_x",
        "acc_x",
    ),

    StandardField.AY: (
        "ay",
        "accel_y",
        "acceleration_y",
        "accelerometer_y",
        "acc_y",
    ),

    StandardField.AZ: (
        "az",
        "accel_z",
        "acceleration_z",
        "accelerometer_z",
        "acc_z",
    ),

    # Gyroscope

    StandardField.GX: (
        "gx",
        "gyro_x",
        "gyroscope_x",
        "angular_velocity_x",
        "angular_rate_x",
        "omega_x",
    ),

    StandardField.GY: (
        "gy",
        "gyro_y",
        "gyroscope_y",
        "angular_velocity_y",
        "angular_rate_y",
        "omega_y",
    ),

    StandardField.GZ: (
        "gz",
        "gyro_z",
        "gyroscope_z",
        "angular_velocity_z",
        "angular_rate_z",
        "omega_z",
    ),

    # Accelerometer reference

    StandardField.AX_REFERENCE: (
        "ax_reference",
        "ax_ref",
        "accel_x_reference",
        "acceleration_x_reference",
        "ground_truth_ax",
        "truth_ax",
    ),

    StandardField.AY_REFERENCE: (
        "ay_reference",
        "ay_ref",
        "accel_y_reference",
        "acceleration_y_reference",
        "ground_truth_ay",
        "truth_ay",
    ),

    StandardField.AZ_REFERENCE: (
        "az_reference",
        "az_ref",
        "accel_z_reference",
        "acceleration_z_reference",
        "ground_truth_az",
        "truth_az",
    ),

    # Gyroscope reference

    StandardField.GX_REFERENCE: (
        "gx_reference",
        "gx_ref",
        "gyro_x_reference",
        "angular_rate_x_reference",
        "ground_truth_gx",
        "truth_gx",
    ),

    StandardField.GY_REFERENCE: (
        "gy_reference",
        "gy_ref",
        "gyro_y_reference",
        "angular_rate_y_reference",
        "ground_truth_gy",
        "truth_gy",
    ),

    StandardField.GZ_REFERENCE: (
        "gz_reference",
        "gz_ref",
        "gyro_z_reference",
        "angular_rate_z_reference",
        "ground_truth_gz",
        "truth_gz",
    ),

    # Position

    StandardField.POSITION_X: (
        "position_x",
        "pos_x",
        "position_x_m",
    ),

    StandardField.POSITION_Y: (
        "position_y",
        "pos_y",
        "position_y_m",
    ),

    StandardField.POSITION_Z: (
        "position_z",
        "pos_z",
        "position_z_m",
    ),

    # Velocity

    StandardField.VELOCITY_X: (
        "velocity_x",
        "vel_x",
    ),

    StandardField.VELOCITY_Y: (
        "velocity_y",
        "vel_y",
    ),

    StandardField.VELOCITY_Z: (
        "velocity_z",
        "vel_z",
    ),

    # Quaternion

    StandardField.QUATERNION_W: (
        "quaternion_w",
        "quat_w",
        "qw",
    ),

    StandardField.QUATERNION_X: (
        "quaternion_x",
        "quat_x",
        "qx",
    ),

    StandardField.QUATERNION_Y: (
        "quaternion_y",
        "quat_y",
        "qy",
    ),

    StandardField.QUATERNION_Z: (
        "quaternion_z",
        "quat_z",
        "qz",
    ),

    # Euler

    StandardField.ROLL: (
        "roll",
        "roll_angle",
    ),

    StandardField.PITCH: (
        "pitch",
        "pitch_angle",
    ),

    StandardField.YAW: (
        "yaw",
        "yaw_angle",
        "heading",
    ),

    # Test metadata columns

    StandardField.ORIENTATION_LABEL: (
        "orientation_label",
        "orientation",
        "static_orientation",
        "pose_label",
    ),

    StandardField.MOTION_STATE: (
        "motion_state",
        "motion",
        "state",
    ),

    StandardField.EXPERIMENT_LABEL: (
        "experiment_label",
        "experiment",
        "test_label",
        "run_label",
    ),
}


# ColumnMapper


class ColumnMapper:
    """
    SensorQA column mapping service.

    This class implements the exact API used by DatasetBuilder.
    """

    def __init__(
        self,
        aliases: Mapping[
            StandardField | str,
            Sequence[str],
        ] | None = None,
    ) -> None:

        """Initialize the column mapper.
        
        Args:
            aliases: Aliases used by this function.
        
        Returns:
            None.
        """
        self.aliases: dict[
            StandardField,
            tuple[str, ...],
        ] = {
            field_:
                tuple(
                    values
                )
            for field_, values
            in DEFAULT_ALIASES.items()
        }

        if aliases is not None:

            for raw_field, raw_aliases in (
                aliases.items()
            ):

                field_ = (
                    _coerce_standard_field(
                        raw_field
                    )
                )

                if isinstance(
                    raw_aliases,
                    str,
                ):
                    raise TypeError(
                        "Alias collection must be a sequence "
                        "of strings."
                    )

                cleaned = []

                for alias in raw_aliases:

                    if not isinstance(
                        alias,
                        str,
                    ):
                        raise TypeError(
                            "Column aliases must be strings."
                        )

                    alias = alias.strip()

                    if alias:
                        cleaned.append(
                            alias
                        )

                self.aliases[
                    field_
                ] = tuple(
                    cleaned
                )

        self._alias_index = (
            self._build_alias_index()
        )

    # Name normalization

    @staticmethod
    def normalize_column_name(
        value: Any,
    ) -> str:

        """Normalize a column name for alias matching.
        
        Args:
            value: Value to process.
        
        Returns:
            Requested text value.
        """
        return _normalize_name(
            value
        )

    normalize = normalize_column_name

    # Alias index

    def _build_alias_index(
        self,
    ) -> dict[
        str,
        set[StandardField],
    ]:

        """Build the lookup table used for column-name aliases.
        
        Returns:
            Dictionary containing the result values.
        """
        index: dict[
            str,
            set[StandardField],
        ] = {}

        for field_, aliases in (
            self.aliases.items()
        ):

            for alias in (
                field_.value,
                *aliases,
            ):

                normalized = (
                    _normalize_name(
                        alias
                    )
                )

                if not normalized:
                    continue

                index.setdefault(
                    normalized,
                    set(),
                ).add(
                    field_
                )

        return index

    # Mapping conversion

    def _coerce_mapping(
        self,
        mapping,
        *,
        available_columns: Iterable[str] | None = None,
    ) -> ColumnMapping:
        """Normalize ColumnMapping/dictionary input.
        
        Dictionary forms accepted:
        
            canonical -> source
        
        and:
        
            source -> canonical
        
        Args:
            mapping: Value for `mapping`.
            available_columns: Value for `available_columns`.
        
        Returns:
            ColumnMapping returned by the function.
        """

        if isinstance(
            mapping,
            ColumnMapping,
        ):
            return mapping

        if mapping is None:

            if available_columns is None:
                raise ValueError(
                    "Cannot infer mapping without dataset columns."
                )

            return self.infer_mapping(
                available_columns
            )

        if not isinstance(
            mapping,
            Mapping,
        ):
            raise TypeError(
                "mapping must be ColumnMapping or mapping."
            )

        available_list = (
            [
                str(
                    column
                )
                for column
                in available_columns
            ]
            if available_columns is not None
            else None
        )

        available_set = (
            set(
                available_list
            )
            if available_list is not None
            else None
        )

        resolved: dict[
            StandardField,
            str,
        ] = {}

        for raw_key, raw_value in (
            mapping.items()
        ):

            # First attempt canonical -> source.

            canonical_key = None

            try:

                canonical_key = (
                    _coerce_standard_field(
                        raw_key
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

            if canonical_key is not None:

                if not isinstance(
                    raw_value,
                    str,
                ):
                    raise TypeError(
                        "Source column name must be a string."
                    )

                source = raw_value.strip()

                if (
                    available_set is None
                    or source
                    in available_set
                ):

                    resolved[
                        canonical_key
                    ] = source

                    continue

            # Otherwise interpret source -> canonical.

            if not isinstance(
                raw_key,
                str,
            ):
                raise ValueError(
                    f"Could not interpret mapping item "
                    f"{raw_key!r}: {raw_value!r}"
                )

            source = raw_key.strip()

            canonical_value = (
                _coerce_standard_field(
                    raw_value
                )
            )

            if (
                available_set is not None
                and source
                not in available_set
            ):
                raise ValueError(
                    f"Dataset column {source!r} does not exist."
                )

            resolved[
                canonical_value
            ] = source

        mapped_sources = set(
            resolved.values()
        )

        if available_list is None:

            unmapped = []

        else:

            unmapped = [
                column
                for column
                in available_list
                if column
                not in mapped_sources
            ]

        return ColumnMapping(
            mapping=resolved,
            unmapped_columns=unmapped,
            inferred=False,
        )

    # Automatic inference

    def infer_mapping(
        self,
        columns: Iterable[str],
        explicit_mapping=None,
    ) -> ColumnMapping:
        """Conservatively infer canonical fields.
        
        Args:
            columns: Value for `columns`.
            explicit_mapping: Value for `explicit_mapping`.
        
        Returns:
            ColumnMapping returned by the function.
        """

        source_columns = [
            str(
                column
            )
            for column
            in columns
        ]

        if len(
            source_columns
        ) != len(
            set(
                source_columns
            )
        ):
            raise ValueError(
                "Duplicate source dataframe column names "
                "are not supported."
            )

        if explicit_mapping is None:

            resolved: dict[
                StandardField,
                str,
            ] = {}

            warnings: list[str] = []

        else:

            explicit = (
                self._coerce_mapping(
                    explicit_mapping,
                    available_columns=(
                        source_columns
                    ),
                )
            )

            resolved = (
                explicit.standard_to_source
            )

            warnings = list(
                explicit.warnings
            )

        used_sources = set(
            resolved.values()
        )

        normalized_sources = {
            source:
                _normalize_name(
                    source
                )
            for source
            in source_columns
        }

        # Exact canonical names first.

        for field_ in (
            StandardField
        ):

            if field_ in resolved:
                continue

            matches = [
                source
                for source
                in source_columns
                if (
                    source
                    not in used_sources
                    and normalized_sources[
                        source
                    ]
                    == _normalize_name(
                        field_.value
                    )
                )
            ]

            if len(
                matches
            ) == 1:

                source = matches[
                    0
                ]

                resolved[
                    field_
                ] = source

                used_sources.add(
                    source
                )

        # Conservative aliases.

        for source in (
            source_columns
        ):

            if source in used_sources:
                continue

            normalized = (
                normalized_sources[
                    source
                ]
            )

            candidates = set(
                self._alias_index.get(
                    normalized,
                    set(),
                )
            )

            candidates = {
                field_
                for field_
                in candidates
                if field_
                not in resolved
            }

            if len(
                candidates
            ) == 1:

                field_ = next(
                    iter(
                        candidates
                    )
                )

                resolved[
                    field_
                ] = source

                used_sources.add(
                    source
                )

            elif len(
                candidates
            ) > 1:

                warnings.append(
                    f"Column {source!r} matched multiple "
                    "SensorQA fields and was left unmapped."
                )

        unmapped = [
            source
            for source
            in source_columns
            if source
            not in used_sources
        ]

        return ColumnMapping(
            mapping=resolved,
            warnings=warnings,
            unmapped_columns=unmapped,
            inferred=True,
        )

    # Compatibility names.

    map_columns = infer_mapping
    build_mapping = infer_mapping

    # EXACT DatasetBuilder validation contract

    def validate(
        self,
        data: pd.DataFrame,
        mapping,
        required_fields: Iterable[
            StandardField | str
        ] | None = None,
        **_,
    ) -> ColumnMappingValidationResult:
        """Validate a column mapping against a dataframe.
        
        This method intentionally matches DatasetBuilder:
        
            self.column_mapper.validate(
                data=raw_data,
                mapping=column_mapping,
            )
        
        Args:
            data: Input data to process.
            mapping: Value for `mapping`.
            required_fields: Value for `required_fields`.
            **_: Value for `_`.
        
        Returns:
            Validation result.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):

            return ColumnMappingValidationResult(
                valid=False,
                errors=[
                    "data must be a pandas DataFrame."
                ],
            )

        try:

            resolved = (
                self._coerce_mapping(
                    mapping,
                    available_columns=(
                        data.columns
                    ),
                )
            )

        except Exception as exc:

            return ColumnMappingValidationResult(
                valid=False,
                errors=[
                    str(
                        exc
                    )
                ],
            )

        errors: list[str] = []

        warnings = list(
            resolved.warnings
        )

        available_columns = [
            str(
                column
            )
            for column
            in data.columns
        ]

        available_set = set(
            available_columns
        )

        # Missing mapped source columns

        missing_sources = sorted(
            {
                source
                for source
                in resolved.mapping.values()
                if source
                not in available_set
            }
        )

        if missing_sources:

            errors.append(
                "Mapped dataset column(s) do not exist: "
                + ", ".join(
                    missing_sources
                )
            )

        # Duplicate physical-column assignments

        source_to_fields: dict[
            str,
            list[str],
        ] = {}

        for field_, source in (
            resolved.mapping.items()
        ):

            source_to_fields.setdefault(
                source,
                [],
            ).append(
                field_.value
            )

        duplicated_dataset_columns = {
            source:
                sorted(
                    fields
                )
            for source, fields
            in source_to_fields.items()
            if len(
                fields
            ) > 1
        }

        # DatasetBuilder explicitly handles these after checking
        # validation.valid, so they are reported separately rather
        # than being added to validation errors here.

        # Required canonical fields

        missing_required_fields: list[
            str
        ] = []

        if required_fields is not None:

            for raw_field in (
                required_fields
            ):

                field_ = (
                    _coerce_standard_field(
                        raw_field
                    )
                )

                if (
                    field_
                    not in resolved.mapping
                ):

                    missing_required_fields.append(
                        field_.value
                    )

            if missing_required_fields:

                errors.append(
                    "Required SensorQA field(s) are not mapped: "
                    + ", ".join(
                        missing_required_fields
                    )
                )

        mapped_sources = set(
            resolved.mapping.values()
        )

        unmapped_columns = [
            column
            for column
            in available_columns
            if column
            not in mapped_sources
        ]

        return ColumnMappingValidationResult(
            valid=not errors,

            errors=errors,

            warnings=warnings,

            missing_required_fields=(
                missing_required_fields
            ),

            unmapped_columns=(
                unmapped_columns
            ),

            duplicated_dataset_columns=(
                duplicated_dataset_columns
            ),

            resolved_mapping=resolved,
        )

    # EXACT DatasetBuilder mapped-view contract

    def create_mapped_view(
        self,
        data: pd.DataFrame,
        mapping,
        keep_unmapped_columns: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Create SensorQA-standardized dataframe.
        
        Exact DatasetBuilder call supported:
        
            self.column_mapper.create_mapped_view(
                data=raw_data,
                mapping=column_mapping,
                keep_unmapped_columns=True,
            )
        
        include_unmapped is also accepted as an alias for compatibility.
        
        Args:
            data: Input data to process.
            mapping: Value for `mapping`.
            keep_unmapped_columns: Value for `keep_unmapped_columns`.
            **kwargs: Value for `kwargs`.
        
        Returns:
            DataFrame containing the requested data.
        """

        if "include_unmapped" in kwargs:

            keep_unmapped_columns = bool(
                kwargs[
                    "include_unmapped"
                ]
            )

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        resolved = (
            self._coerce_mapping(
                mapping,
                available_columns=(
                    data.columns
                ),
            )
        )

        validation = self.validate(
            data=data,
            mapping=resolved,
        )

        if not validation.valid:

            raise ValueError(
                "; ".join(
                    validation.errors
                )
            )

        if (
            validation
            .duplicated_dataset_columns
        ):

            messages = []

            for source, fields in (
                validation
                .duplicated_dataset_columns
                .items()
            ):

                messages.append(
                    f"{source!r} -> "
                    + ", ".join(
                        fields
                    )
                )

            raise ValueError(
                "Dataset columns cannot be mapped to multiple "
                "SensorQA fields: "
                + "; ".join(
                    messages
                )
            )

        rename_map = {
            source:
                field_.value
            for field_, source
            in resolved.mapping.items()
        }

        mapped_sources = set(
            rename_map
        )

        if keep_unmapped_columns:

            output = data.copy(
                deep=True
            )

        else:

            ordered_sources = [
                source
                for source
                in resolved.mapping.values()
            ]

            output = data.loc[
                :,
                ordered_sources,
            ].copy(
                deep=True
            )

        # Prevent accidental collision.
        #
        # Example:
        #
        # source "Accel X" -> canonical "ax"
        #
        # while an unrelated unmapped dataset column is already "ax".

        if keep_unmapped_columns:

            canonical_targets = set(
                rename_map.values()
            )

            untouched_sources = {
                str(
                    column
                )
                for column
                in data.columns
                if str(
                    column
                )
                not in mapped_sources
            }

            collisions = sorted(
                canonical_targets
                & untouched_sources
            )

            if collisions:

                raise ValueError(
                    "Canonical mapped column name collides with "
                    "unmapped dataset column(s): "
                    + ", ".join(
                        collisions
                    )
                )

        output = output.rename(
            columns=rename_map
        )

        return output

    # Interactive convenience

    def map_dataframe(
        self,
        dataframe: pd.DataFrame,
        explicit_mapping=None,
        *,
        keep_unmapped_columns: bool = True,
    ) -> tuple[
        pd.DataFrame,
        ColumnMapping,
    ]:

        """Rename mapped DataFrame columns to SensorQA field names.
        
        Args:
            dataframe: Dataframe used by this function.
            explicit_mapping: Explicit mapping used by this function.
            keep_unmapped_columns: Keep unmapped columns used by this function.
        
        Returns:
            DataFrame containing the requested data.
        """
        if explicit_mapping is None:

            resolved = (
                self.infer_mapping(
                    dataframe.columns
                )
            )

        else:

            resolved = (
                self._coerce_mapping(
                    explicit_mapping,
                    available_columns=(
                        dataframe.columns
                    ),
                )
            )

        mapped = self.create_mapped_view(
            data=dataframe,
            mapping=resolved,
            keep_unmapped_columns=(
                keep_unmapped_columns
            ),
        )

        return (
            mapped,
            resolved,
        )


# Module convenience function


def create_mapped_view(
    dataframe: pd.DataFrame,
    mapping: ColumnMapping | Mapping,
    *,
    keep_unmapped_columns: bool = True,
) -> pd.DataFrame:

    """Create a DataFrame whose mapped columns use SensorQA field names.
    
    Args:
        dataframe: Dataframe used by this function.
        mapping: Mapping used by this function.
        keep_unmapped_columns: Keep unmapped columns used by this function.
    
    Returns:
        DataFrame containing the requested data.
    """
    mapper = ColumnMapper()

    return mapper.create_mapped_view(
        data=dataframe,
        mapping=mapping,
        keep_unmapped_columns=(
            keep_unmapped_columns
        ),
    )
