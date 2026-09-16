#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from sensorqa.ingestion.column_mapper import StandardField


class UnitDimension(str, Enum):
    """
    Physical quantity associated with a unit.

    SensorQA only standardizes quantities for which it knows
    the physical meaning.
    """

    TIME = "time"
    ACCELERATION = "acceleration"
    ANGULAR_VELOCITY = "angular_velocity"
    TEMPERATURE = "temperature"
    VOLTAGE = "voltage"
    FREQUENCY = "frequency"
    DIMENSIONLESS = "dimensionless"
    UNKNOWN = "unknown"


@dataclass
class UnitAssignment:
    """
    Associates a SensorQA standard field with the unit used
    in the uploaded dataset.

    Example:

        field = "gx"
        source_unit = "deg/s"
    """

    field: str
    source_unit: str


@dataclass
class UnitConversionRecord:
    """
    Records one unit conversion performed by SensorQA.

    This information can later be included in the report
    for reproducibility.
    """

    field: str
    source_unit: str
    target_unit: str
    converted: bool


@dataclass
class UnitNormalizationResult:
    """
    Result of normalizing units in a dataset.
    """

    data: pd.DataFrame

    units: dict[str, str] = field(default_factory=dict)

    conversions: list[UnitConversionRecord] = field(
        default_factory=list
    )

    warnings: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)


class UnitConversionError(ValueError):
    """
    Raised when SensorQA cannot safely perform a requested
    unit conversion.
    """


class UnitManager:
    """
    Handles engineering units used by SensorQA.

    Known sensor quantities are converted into canonical
    internal units.

    Canonical V1 units:

        time             -> s
        acceleration     -> m/s^2
        angular velocity -> rad/s
        temperature      -> degC
        voltage          -> V
        frequency        -> Hz

    Unknown or generic engineering quantities are preserved
    in their original units.
    """

    STANDARD_GRAVITY = 9.80665

    # Public API

    def normalize_dataframe(
        self,
        data: pd.DataFrame,
        unit_assignments: dict[str, str],
    ) -> UnitNormalizationResult:
        """Convert recognized SensorQA fields into canonical units.
        
        The input DataFrame is never modified.
        
        Parameters
        ----------
        data:
            DataFrame whose mapped columns already use SensorQA
            standard field names.
        
        unit_assignments:
            Mapping such as:
        
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
        
        Args:
            data: Input data to process.
            unit_assignments: Engineering units assigned to mapped fields.
        
        Returns:
            UnitNormalizationResult returned by the function.
        """

        normalized_data = data.copy()

        normalized_units: dict[str, str] = {}
        conversions: list[UnitConversionRecord] = []

        warnings: list[str] = []
        errors: list[str] = []

        for field_name, source_unit in unit_assignments.items():

            if field_name not in normalized_data.columns:
                warnings.append(
                    f"Unit was specified for field '{field_name}', "
                    "but that field is not present in the mapped dataset."
                )
                continue

            source_unit = self.normalize_unit_name(
                source_unit
            )

            dimension = self.dimension_for_field(
                field_name
            )

            # Unknown/general engineering quantity

            # Generic sensor fields such as ``measurement`` and
            # ``reference`` intentionally use UNKNOWN because their
            # engineering dimension is supplied by the user rather than
            # hard-coded by SensorQA. Preserve those units exactly instead
            # of asking for a canonical unit that does not exist.

            if dimension == UnitDimension.UNKNOWN:

                normalized_units[field_name] = source_unit

                conversions.append(
                    UnitConversionRecord(
                        field=field_name,
                        source_unit=source_unit,
                        target_unit=source_unit,
                        converted=False,
                    )
                )

                continue

            # Dimensionless quantity

            if dimension == UnitDimension.DIMENSIONLESS:

                normalized_units[field_name] = "1"

                conversions.append(
                    UnitConversionRecord(
                        field=field_name,
                        source_unit=source_unit,
                        target_unit="1",
                        converted=False,
                    )
                )

                continue

            # Known physical quantity

            canonical_unit = self.canonical_unit(
                dimension
            )

            try:

                converted_values = self.convert_series(
                    series=normalized_data[field_name],
                    source_unit=source_unit,
                    target_unit=canonical_unit,
                    dimension=dimension,
                )

                normalized_data[field_name] = converted_values

                normalized_units[field_name] = canonical_unit

                conversions.append(
                    UnitConversionRecord(
                        field=field_name,
                        source_unit=source_unit,
                        target_unit=canonical_unit,
                        converted=(
                            source_unit != canonical_unit
                        ),
                    )
                )

            except UnitConversionError as exc:

                errors.append(
                    f"Could not normalize '{field_name}': {exc}"
                )

        return UnitNormalizationResult(
            data=normalized_data,
            units=normalized_units,
            conversions=conversions,
            warnings=warnings,
            errors=errors,
        )

    def convert_series(
        self,
        series: pd.Series,
        source_unit: str,
        target_unit: str,
        dimension: UnitDimension,
    ) -> pd.Series:
        """Convert a pandas Series from one unit to another.
        
        Only known SensorQA physical dimensions are supported.
        
        Args:
            series: Value for `series`.
            source_unit: Value for `source_unit`.
            target_unit: Value for `target_unit`.
            dimension: Value for `dimension`.
        
        Returns:
            pd.Series returned by the function.
        """

        source_unit = self.normalize_unit_name(
            source_unit
        )

        target_unit = self.normalize_unit_name(
            target_unit
        )

        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        if source_unit == target_unit:
            return numeric_series

        if dimension == UnitDimension.TIME:
            return self._convert_time(
                numeric_series,
                source_unit,
                target_unit,
            )

        if dimension == UnitDimension.ACCELERATION:
            return self._convert_acceleration(
                numeric_series,
                source_unit,
                target_unit,
            )

        if dimension == UnitDimension.ANGULAR_VELOCITY:
            return self._convert_angular_velocity(
                numeric_series,
                source_unit,
                target_unit,
            )

        if dimension == UnitDimension.TEMPERATURE:
            return self._convert_temperature(
                numeric_series,
                source_unit,
                target_unit,
            )

        if dimension == UnitDimension.VOLTAGE:
            return self._convert_voltage(
                numeric_series,
                source_unit,
                target_unit,
            )

        if dimension == UnitDimension.FREQUENCY:
            return self._convert_frequency(
                numeric_series,
                source_unit,
                target_unit,
            )

        raise UnitConversionError(
            f"No conversion is defined for dimension "
            f"'{dimension.value}'."
        )

    # Field interpretation

    @staticmethod
    def dimension_for_field(
        field_name: StandardField | str,
    ) -> UnitDimension:
        """Determine the physical dimension associated with a
        SensorQA standard field.
        
        Args:
            field_name: Value for `field_name`.
        
        Returns:
            UnitDimension returned by the function.
        """

        if isinstance(field_name, StandardField):
            field_name = field_name.value

        time_fields = {
            StandardField.TIMESTAMP.value,
        }

        acceleration_fields = {
            StandardField.AX.value,
            StandardField.AY.value,
            StandardField.AZ.value,
            StandardField.AX_REFERENCE.value,
            StandardField.AY_REFERENCE.value,
            StandardField.AZ_REFERENCE.value,
        }

        angular_velocity_fields = {
            StandardField.GX.value,
            StandardField.GY.value,
            StandardField.GZ.value,
            StandardField.GX_REFERENCE.value,
            StandardField.GY_REFERENCE.value,
            StandardField.GZ_REFERENCE.value,
        }

        temperature_fields = {
            StandardField.TEMPERATURE.value,
        }

        voltage_fields = {
            StandardField.SUPPLY_VOLTAGE.value,
        }

        dimensionless_fields = {
            StandardField.SENSOR_ID.value,
            StandardField.REVISION.value,
            StandardField.TEST_CYCLE.value,
            StandardField.SWEEP_DIRECTION.value,
            StandardField.ORIENTATION_LABEL.value,
            StandardField.MOTION_STATE.value,
            StandardField.EXPERIMENT_LABEL.value,
        }

        if field_name in time_fields:
            return UnitDimension.TIME

        if field_name in acceleration_fields:
            return UnitDimension.ACCELERATION

        if field_name in angular_velocity_fields:
            return UnitDimension.ANGULAR_VELOCITY

        if field_name in temperature_fields:
            return UnitDimension.TEMPERATURE

        if field_name in voltage_fields:
            return UnitDimension.VOLTAGE

        if field_name in dimensionless_fields:
            return UnitDimension.DIMENSIONLESS

        return UnitDimension.UNKNOWN

    @staticmethod
    def canonical_unit(
        dimension: UnitDimension,
    ) -> str:
        """Return SensorQA's canonical internal unit for a
        physical dimension.
        
        Args:
            dimension: Value for `dimension`.
        
        Returns:
            String representation.
        """

        canonical_units = {
            UnitDimension.TIME: "s",
            UnitDimension.ACCELERATION: "m/s^2",
            UnitDimension.ANGULAR_VELOCITY: "rad/s",
            UnitDimension.TEMPERATURE: "degC",
            UnitDimension.VOLTAGE: "V",
            UnitDimension.FREQUENCY: "Hz",
            UnitDimension.DIMENSIONLESS: "1",
        }

        if dimension not in canonical_units:
            raise UnitConversionError(
                f"No canonical unit exists for "
                f"dimension '{dimension.value}'."
            )

        return canonical_units[dimension]

    # Unit-name normalization

    @staticmethod
    def normalize_unit_name(
        unit: str,
    ) -> str:
        """Convert common unit spellings into SensorQA's standard
        unit representation.
        
        This does not perform the numerical conversion.
        
        Args:
            unit: Engineering unit.
        
        Returns:
            String representation.
        """

        if not isinstance(unit, str):
            raise UnitConversionError(
                "Unit must be provided as a string."
            )

        cleaned = unit.strip()

        aliases = {
            # Time
            "sec": "s",
            "second": "s",
            "seconds": "s",

            "millisecond": "ms",
            "milliseconds": "ms",

            "us": "us",
            "µs": "us",
            "μs": "us",
            "microsecond": "us",
            "microseconds": "us",

            "nanosecond": "ns",
            "nanoseconds": "ns",

            "minute": "min",
            "minutes": "min",

            "hour": "h",
            "hours": "h",
            "hr": "h",

            # Acceleration
            "m/s²": "m/s^2",
            "m/s2": "m/s^2",
            "mps2": "m/s^2",
            "meter/second^2": "m/s^2",
            "meters/second^2": "m/s^2",

            "cm/s²": "cm/s^2",
            "cm/s2": "cm/s^2",

            "g0": "g",
            "gravity": "g",

            # Angular velocity
            "deg/sec": "deg/s",
            "degree/s": "deg/s",
            "degrees/s": "deg/s",
            "degree/sec": "deg/s",
            "degrees/sec": "deg/s",
            "°/s": "deg/s",

            "rad/sec": "rad/s",
            "radian/s": "rad/s",
            "radians/s": "rad/s",

            # Temperature
            "c": "degC",
            "°c": "degC",
            "celsius": "degC",
            "degc": "degC",

            "f": "degF",
            "°f": "degF",
            "fahrenheit": "degF",
            "degf": "degF",

            "kelvin": "K",

            # Voltage
            "volt": "V",
            "volts": "V",

            "millivolt": "mV",
            "millivolts": "mV",

            "microvolt": "uV",
            "microvolts": "uV",
            "µv": "uV",
            "μv": "uV",

            # Frequency
            "hz": "Hz",
            "hertz": "Hz",

            "khz": "kHz",
            "kilohertz": "kHz",

            "mhz": "MHz",
            "megahertz": "MHz",

            # Dimensionless
            "": "1",
            "none": "1",
            "dimensionless": "1",
        }

        if cleaned in aliases:
            return aliases[cleaned]

        lowercase = cleaned.lower()

        if lowercase in aliases:
            return aliases[lowercase]

        return cleaned

    # Individual conversion functions

    @staticmethod
    def _convert_time(
        values: pd.Series,
        source_unit: str,
        target_unit: str,
    ) -> pd.Series:
        """Convert time units through seconds.
        
        Args:
            values: Values to process.
            source_unit: Value for `source_unit`.
            target_unit: Value for `target_unit`.
        
        Returns:
            pd.Series returned by the function.
        """

        factors_to_seconds = {
            "ns": 1e-9,
            "us": 1e-6,
            "ms": 1e-3,
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
        }

        return UnitManager._linear_conversion(
            values,
            source_unit,
            target_unit,
            factors_to_seconds,
            dimension_name="time",
        )

    @classmethod
    def _convert_acceleration(
        cls,
        values: pd.Series,
        source_unit: str,
        target_unit: str,
    ) -> pd.Series:
        """Convert acceleration through m/s^2.
        
        Args:
            values: Values to process.
            source_unit: Value for `source_unit`.
            target_unit: Value for `target_unit`.
        
        Returns:
            pd.Series returned by the function.
        """

        factors_to_si = {
            "m/s^2": 1.0,
            "cm/s^2": 0.01,
            "g": cls.STANDARD_GRAVITY,
        }

        return cls._linear_conversion(
            values,
            source_unit,
            target_unit,
            factors_to_si,
            dimension_name="acceleration",
        )

    @staticmethod
    def _convert_angular_velocity(
        values: pd.Series,
        source_unit: str,
        target_unit: str,
    ) -> pd.Series:
        """Convert angular velocity through rad/s.
        
        Args:
            values: Values to process.
            source_unit: Value for `source_unit`.
            target_unit: Value for `target_unit`.
        
        Returns:
            pd.Series returned by the function.
        """

        factors_to_si = {
            "rad/s": 1.0,
            "deg/s": np.pi / 180.0,
        }

        return UnitManager._linear_conversion(
            values,
            source_unit,
            target_unit,
            factors_to_si,
            dimension_name="angular velocity",
        )

    @staticmethod
    def _convert_voltage(
        values: pd.Series,
        source_unit: str,
        target_unit: str,
    ) -> pd.Series:
        """Convert voltage through volts.
        
        Args:
            values: Values to process.
            source_unit: Value for `source_unit`.
            target_unit: Value for `target_unit`.
        
        Returns:
            pd.Series returned by the function.
        """

        factors_to_volts = {
            "uV": 1e-6,
            "mV": 1e-3,
            "V": 1.0,
        }

        return UnitManager._linear_conversion(
            values,
            source_unit,
            target_unit,
            factors_to_volts,
            dimension_name="voltage",
        )

    @staticmethod
    def _convert_frequency(
        values: pd.Series,
        source_unit: str,
        target_unit: str,
    ) -> pd.Series:
        """Convert frequency through hertz.
        
        Args:
            values: Values to process.
            source_unit: Value for `source_unit`.
            target_unit: Value for `target_unit`.
        
        Returns:
            pd.Series returned by the function.
        """

        factors_to_hz = {
            "Hz": 1.0,
            "kHz": 1e3,
            "MHz": 1e6,
        }

        return UnitManager._linear_conversion(
            values,
            source_unit,
            target_unit,
            factors_to_hz,
            dimension_name="frequency",
        )

    @staticmethod
    def _convert_temperature(
        values: pd.Series,
        source_unit: str,
        target_unit: str,
    ) -> pd.Series:
        """Convert temperature.
        
        Temperature requires offset conversions, so it cannot use
        the generic linear scale-factor method.
        
        Args:
            values: Values to process.
            source_unit: Value for `source_unit`.
            target_unit: Value for `target_unit`.
        
        Returns:
            pd.Series returned by the function.
        """

        supported = {
            "degC",
            "degF",
            "K",
        }

        if source_unit not in supported:
            raise UnitConversionError(
                f"Unsupported temperature unit "
                f"'{source_unit}'."
            )

        if target_unit not in supported:
            raise UnitConversionError(
                f"Unsupported temperature unit "
                f"'{target_unit}'."
            )

        # First convert to Celsius.

        if source_unit == "degC":
            celsius = values

        elif source_unit == "K":
            celsius = values - 273.15

        else:
            celsius = (
                values - 32.0
            ) * (5.0 / 9.0)

        # Then convert Celsius to target.

        if target_unit == "degC":
            return celsius

        if target_unit == "K":
            return celsius + 273.15

        return (
            celsius * (9.0 / 5.0)
        ) + 32.0

    @staticmethod
    def _linear_conversion(
        values: pd.Series,
        source_unit: str,
        target_unit: str,
        factors_to_base: dict[str, float],
        dimension_name: str,
    ) -> pd.Series:
        """Convert quantities that differ only by a multiplicative
        scale factor.
        
        All factors describe conversion into a common base unit.
        
        Args:
            values: Values to process.
            source_unit: Value for `source_unit`.
            target_unit: Value for `target_unit`.
            factors_to_base: Value for `factors_to_base`.
            dimension_name: Value for `dimension_name`.
        
        Returns:
            pd.Series returned by the function.
        """

        if source_unit not in factors_to_base:
            raise UnitConversionError(
                f"Unsupported {dimension_name} unit "
                f"'{source_unit}'."
            )

        if target_unit not in factors_to_base:
            raise UnitConversionError(
                f"Unsupported {dimension_name} unit "
                f"'{target_unit}'."
            )

        values_in_base = (
            values
            * factors_to_base[source_unit]
        )

        converted = (
            values_in_base
            / factors_to_base[target_unit]
        )

        return converted
