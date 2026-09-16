#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from sensorqa.core.tool_contract import SensorType


class TestMode(str, Enum):
    """
    Broad experiment type.

    SensorQA uses this to decide which analyses may be appropriate.
    """

    UNKNOWN = "unknown"

    GENERIC_STATIC = "generic_static"
    GENERIC_DYNAMIC = "generic_dynamic"

    IMU_STATIONARY = "imu_stationary"
    IMU_CONTROLLED_ORIENTATION = "imu_controlled_orientation"
    IMU_DYNAMIC = "imu_dynamic"


class ReferenceType(str, Enum):
    """
    Describes the source of ground truth or reference measurements.
    """

    NONE = "none"
    USER_PROVIDED = "user_provided"
    CALIBRATED_INSTRUMENT = "calibrated_instrument"
    MOTION_CAPTURE = "motion_capture"
    GNSS = "gnss"
    ENCODER = "encoder"
    SIMULATION = "simulation"
    OTHER = "other"


@dataclass
class SensorRange:
    """
    Optional sensor full-scale information.

    Examples:

        accelerometer:
            minimum = -16
            maximum = 16
            unit = "g"

        gyroscope:
            minimum = -2000
            maximum = 2000
            unit = "deg/s"
    """

    minimum: float
    maximum: float
    unit: str

    def __post_init__(self) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if self.minimum >= self.maximum:
            raise ValueError(
                "SensorRange minimum must be less than maximum."
            )

        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError(
                "SensorRange unit must be a non-empty string."
            )

    @property
    def span(self) -> float:
        """Total full-scale span.
        
        Returns:
            Numeric result.
        """

        return self.maximum - self.minimum

    @property
    def symmetric_limit(self) -> float | None:
        """Return the positive magnitude if the range is symmetric.
        
        Example:
            -2000 to +2000 -> 2000
        
        Returns None for non-symmetric ranges.
        
        Returns:
            Numeric result.
        """

        if abs(self.minimum) == abs(self.maximum):
            return abs(self.maximum)

        return None


@dataclass
class SensorInformation:
    """
    Describes the physical sensor or sensor system being analyzed.
    """

    sensor_type: SensorType

    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None

    firmware_version: str | None = None
    hardware_revision: str | None = None

    nominal_sampling_rate_hz: float | None = None

    accelerometer_range: SensorRange | None = None
    gyroscope_range: SensorRange | None = None

    notes: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if (
            self.nominal_sampling_rate_hz is not None
            and self.nominal_sampling_rate_hz <= 0
        ):
            raise ValueError(
                "nominal_sampling_rate_hz must be positive."
            )


@dataclass
class TestInformation:
    """
    Describes the experiment that produced the uploaded dataset.
    """

    test_name: str | None = None
    test_mode: TestMode = TestMode.UNKNOWN

    test_date: str | None = None

    operator: str | None = None
    laboratory: str | None = None

    reference_type: ReferenceType = ReferenceType.NONE
    reference_description: str | None = None

    expected_stationary: bool | None = None

    temperature_controlled: bool | None = None

    notes: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSourceInformation:
    """
    Describes where the dataset came from.
    """

    file_name: str | None = None

    source_name: str | None = None

    source_version: str | None = None

    source_url: str | None = None

    license_name: str | None = None

    notes: str | None = None


@dataclass
class SensorQAMetadata:
    """
    Complete metadata associated with one SensorQA analysis session.
    """

    sensor: SensorInformation
    test: TestInformation = field(
        default_factory=TestInformation
    )

    dataset_source: DatasetSourceInformation = field(
        default_factory=DatasetSourceInformation
    )

    user_metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata into a serializable dictionary.
        
        This can later be:
        - saved with analysis results
        - included in reports
        - sent through the API
        
        Returns:
            Dictionary containing the result values.
        """

        return asdict(self)

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve custom user metadata.
        
        Args:
            key: Value for `key`.
            default: Value for `default`.
        
        Returns:
            Any returned by the function.
        """

        return self.user_metadata.get(
            key,
            default,
        )

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Add or update custom metadata.
        
        Args:
            key: Value for `key`.
            value: Value to process.
        
        Returns:
            None.
        """

        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "Metadata key must be a non-empty string."
            )

        self.user_metadata[key] = value
