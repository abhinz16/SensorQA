#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from sensorqa.core.result_schema import (
    AnalysisResult,
    ExecutionStatus,
)


class SensorType(str, Enum):
    """
    Sensor types currently supported by SensorQA.

    A tool can support one or more of these sensor types.
    """

    GENERIC = "generic"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    IMU = "imu"
    ANY = "any"


class ToolCategory(str, Enum):
    """
    High-level category used to organize tools in the UI.
    """

    DATA_QUALITY = "data_quality"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    STABILITY = "stability"
    NOISE = "noise"
    FREQUENCY = "frequency"
    ENVIRONMENTAL = "environmental"
    CALIBRATION = "calibration"
    IMU = "imu"
    DIAGNOSTIC = "diagnostic"
    CUSTOM = "custom"


class ParameterType(str, Enum):
    """
    Parameter types SensorQA understands.

    These will later allow the frontend to automatically
    create appropriate controls for tool parameters.
    """

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    CHOICE = "choice"


@dataclass
class ToolParameter:
    """
    Describes one user-configurable parameter.

    Example:
        name = "max_frequency"
        parameter_type = FLOAT
        default = 100.0
        unit = "Hz"
    """

    name: str
    parameter_type: ParameterType
    default: Any

    description: str | None = None
    unit: str | None = None

    minimum: float | int | None = None
    maximum: float | int | None = None

    choices: list[Any] = field(default_factory=list)

    required: bool = False


@dataclass
class ToolMetadata:
    """
    Metadata describing an analysis tool.

    SensorQA reads this information before executing
    the actual analysis.
    """

    tool_id: str
    name: str
    version: str
    description: str
    category: ToolCategory

    compatible_sensor_types: list[SensorType]

    required_columns: list[str] = field(default_factory=list)
    optional_columns: list[str] = field(default_factory=list)

    parameters: list[ToolParameter] = field(default_factory=list)

    dependencies: list[str] = field(default_factory=list)

    author: str = "SensorQA"

    def supports_sensor_type(self, sensor_type: SensorType) -> bool:
        """Return True if this tool supports the supplied sensor type.
        
        Args:
            sensor_type: Sensor type used to choose compatible tools.
        
        Returns:
            Boolean result.
        """

        if SensorType.ANY in self.compatible_sensor_types:
            return True

        if sensor_type in self.compatible_sensor_types:
            return True

        # An IMU tool may reasonably support both accelerometer
        # and gyroscope datasets.
        if (
            SensorType.IMU in self.compatible_sensor_types
            and sensor_type in {
                SensorType.ACCELEROMETER,
                SensorType.GYROSCOPE,
            }
        ):
            return True

        return False


@dataclass
class ToolContext:
    """
    Additional information supplied to a tool during execution.

    The actual sensor measurements are passed separately
    as a pandas DataFrame.

    Context contains information about how those measurements
    should be interpreted.
    """

    sensor_type: SensorType

    column_mapping: dict[str, str] = field(default_factory=dict)

    units: dict[str, str] = field(default_factory=dict)

    parameters: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolValidationResult:
    """
    Result of checking whether a tool can run on a dataset.
    """

    valid: bool

    missing_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BaseAnalysisTool(ABC):
    """
    Base class that every SensorQA analysis tool must inherit from.

    Built-in tools and user-created custom tools follow exactly
    the same interface.
    """

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Return metadata describing this tool.
        
        Every analysis tool must implement this property.
        
        Returns:
            ToolMetadata returned by the function.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform the analysis.
        
        Every analysis tool must implement this method and
        return an AnalysisResult.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Analysis result containing metrics and messages.
        """
        raise NotImplementedError

    def validate(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> ToolValidationResult:
        """Perform common validation before running a tool.
        
        Individual tools may override this method if they
        require additional checks.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Validation result.
        """

        errors: list[str] = []
        warnings: list[str] = []
        missing_columns: list[str] = []

        metadata = self.metadata

        # Check sensor compatibility

        if not metadata.supports_sensor_type(context.sensor_type):
            errors.append(
                f"Tool '{metadata.name}' is not compatible with "
                f"sensor type '{context.sensor_type.value}'."
            )

        # Check required mapped fields

        for required_field in metadata.required_columns:

            actual_column = context.column_mapping.get(required_field)

            if actual_column is None:
                missing_columns.append(required_field)
                continue

            if actual_column not in data.columns:
                missing_columns.append(required_field)

        if missing_columns:
            errors.append(
                "Missing required input fields: "
                + ", ".join(sorted(set(missing_columns)))
            )

        # Check optional fields

        for optional_field in metadata.optional_columns:

            actual_column = context.column_mapping.get(optional_field)

            if actual_column is None:
                warnings.append(
                    f"Optional field '{optional_field}' is not mapped."
                )

        return ToolValidationResult(
            valid=len(errors) == 0,
            missing_columns=sorted(set(missing_columns)),
            warnings=warnings,
            errors=errors,
        )

    def execute(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Safely validate and execute the tool.
        
        SensorQA should normally call execute(), not run() directly.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        validation = self.validate(data, context)

        # Tool cannot run

        if not validation.valid:

            return AnalysisResult(
                tool_id=self.metadata.tool_id,
                tool_name=self.metadata.name,
                tool_version=self.metadata.version,
                status=ExecutionStatus.SKIPPED,
                warnings=validation.warnings,
                messages=validation.errors,
            )

        # Tool can run

        try:

            result = self.run(data, context)

            # Include validation warnings even when execution succeeds.
            result.warnings.extend(validation.warnings)

            return result

        except Exception as exc:

            return AnalysisResult(
                tool_id=self.metadata.tool_id,
                tool_name=self.metadata.name,
                tool_version=self.metadata.version,
                status=ExecutionStatus.ERROR,
                warnings=validation.warnings,
                messages=[
                    f"Tool execution failed: "
                    f"{type(exc).__name__}: {exc}"
                ],
            )
