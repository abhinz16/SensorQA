#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sensorqa.core.config_loader import SensorQARuntimeConfig
from sensorqa.core.tool_contract import SensorType, ToolParameter
from sensorqa.core.tool_loader import ToolLoader
from sensorqa.core.tool_registry import RegistryBuildReport, ToolRegistry


@dataclass(frozen=True)
class ToolParameterView:
    """Display model for one configurable tool parameter."""
    name: str
    parameter_type: str
    default: Any
    current_value: Any
    description: str | None
    unit: str | None
    minimum: float | int | None
    maximum: float | int | None
    choices: tuple[Any, ...]
    required: bool


@dataclass(frozen=True)
class ToolView:
    """Display model for an analysis tool shown in the Tool Manager."""
    tool_id: str
    name: str
    version: str
    description: str
    category: str
    compatible_sensor_types: tuple[str, ...]
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    dependencies: tuple[str, ...]
    author: str
    source: str
    file_path: str
    custom: bool
    enabled: bool
    parameters: tuple[ToolParameterView, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ToolHealthView:
    """Summary of tool loading and validation health."""
    registered_count: int
    rejected_count: int
    load_error_count: int
    rejected_tools: tuple[dict[str, Any], ...]
    load_errors: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


class ToolService:
    """Application-facing catalog and health view for analysis plug-ins."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        runtime_config: SensorQARuntimeConfig | None = None,
        loader: ToolLoader | None = None,
    ) -> None:
        """Initialize the tool service.
        
        Args:
            registry: Tool registry used for lookups.
            runtime_config: Runtime config used by this function.
            loader: Loader used by this function.
        
        Returns:
            None.
        """
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry.")
        if runtime_config is not None and not isinstance(runtime_config, SensorQARuntimeConfig):
            raise TypeError("runtime_config must be SensorQARuntimeConfig or None.")
        if loader is not None and not isinstance(loader, ToolLoader):
            raise TypeError("loader must be a ToolLoader or None.")

        self.registry = registry
        self.runtime_config = runtime_config
        self.loader = loader

    def list_tools(
        self,
        sensor_type: SensorType | None = None,
    ) -> tuple[ToolView, ...]:
        """Run list tools.
        
        Args:
            sensor_type: Sensor type used to choose compatible tools.
        
        Returns:
            Tuple containing the calculated values.
        """
        if sensor_type is not None and not isinstance(sensor_type, SensorType):
            raise TypeError("sensor_type must be SensorType or None.")

        registered = (
            self.registry.by_sensor_type(sensor_type)
            if sensor_type is not None
            else self.registry.all_tools()
        )

        return tuple(self._tool_view(item) for item in registered)

    def get_tool(self, tool_id: str) -> ToolView | None:
        """Return tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            ToolView | None returned by this function.
        """
        registered = self.registry.get_registered(tool_id)
        if registered is None:
            return None
        return self._tool_view(registered)

    def health(self) -> ToolHealthView:
        """Run health.
        
        Returns:
            ToolHealthView returned by this function.
        """
        report = self.registry.build_report()
        return self._health_view(report)

    def reload_tools(self) -> ToolHealthView:
        """Run reload tools.
        
        Returns:
            ToolHealthView returned by this function.
        """
        if self.loader is None:
            raise RuntimeError(
                "Tool reloading is unavailable because no ToolLoader was supplied."
            )
        load_report = self.loader.reload_tools()
        report = self.registry.rebuild(load_report)
        return self._health_view(report)

    def _tool_view(self, registered) -> ToolView:
        """Return tool view.
        
        Args:
            registered: Registered used by this function.
        
        Returns:
            ToolView returned by this function.
        """
        metadata = registered.tool.metadata
        runtime_parameters = (
            self.runtime_config.parameters_for(metadata.tool_id)
            if self.runtime_config is not None
            else {}
        )
        enabled = (
            self.runtime_config.is_tool_enabled(metadata.tool_id)
            if self.runtime_config is not None
            else True
        )

        parameters = tuple(
            self._parameter_view(parameter, runtime_parameters)
            for parameter in metadata.parameters
        )

        return ToolView(
            tool_id=metadata.tool_id,
            name=metadata.name,
            version=metadata.version,
            description=metadata.description,
            category=metadata.category.value,
            compatible_sensor_types=tuple(
                sensor_type.value
                for sensor_type in metadata.compatible_sensor_types
            ),
            required_columns=tuple(metadata.required_columns),
            optional_columns=tuple(metadata.optional_columns),
            dependencies=tuple(metadata.dependencies),
            author=metadata.author,
            source=registered.source,
            file_path=registered.file_path,
            custom=registered.is_custom,
            enabled=enabled,
            parameters=parameters,
            warnings=tuple(registered.validation_warnings),
        )

    @staticmethod
    def _parameter_view(
        parameter: ToolParameter,
        runtime_parameters: dict[str, Any],
    ) -> ToolParameterView:
        """Return parameter view.
        
        Args:
            parameter: Parameter used by this function.
            runtime_parameters: Runtime parameters used by this function.
        
        Returns:
            ToolParameterView returned by this function.
        """
        return ToolParameterView(
            name=parameter.name,
            parameter_type=parameter.parameter_type.value,
            default=parameter.default,
            current_value=runtime_parameters.get(parameter.name, parameter.default),
            description=parameter.description,
            unit=parameter.unit,
            minimum=parameter.minimum,
            maximum=parameter.maximum,
            choices=tuple(parameter.choices),
            required=parameter.required,
        )

    @staticmethod
    def _health_view(report: RegistryBuildReport) -> ToolHealthView:
        """Return health view.
        
        Args:
            report: Report used by this function.
        
        Returns:
            ToolHealthView returned by this function.
        """
        return ToolHealthView(
            registered_count=report.registered_count,
            rejected_count=report.rejected_count,
            load_error_count=report.load_error_count,
            rejected_tools=tuple(
                {
                    "file_path": item.file_path,
                    "tool_id": item.tool_id,
                    "tool_name": item.tool_name,
                    "errors": list(item.errors),
                    "warnings": list(item.warnings),
                }
                for item in report.rejected_tools
            ),
            load_errors=tuple(
                {
                    "file_path": item.file_path,
                    "error_type": item.error_type,
                    "message": item.message,
                }
                for item in report.load_errors
            ),
            warnings=tuple(report.warnings),
        )
