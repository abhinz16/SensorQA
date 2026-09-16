#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import math

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise ImportError(
            "SensorQA requires Python 3.11+ or the 'tomli' package "
            "to read TOML configuration files."
        ) from exc

from sensorqa.core.tool_contract import (
    ParameterType,
    ToolParameter,
)
from sensorqa.core.tool_registry import (
    RegisteredTool,
    ToolRegistry,
)


# Configuration data structures


@dataclass
class ConfigLoadError:
    """
    One configuration problem discovered while loading
    sensorqa.toml.
    """

    location: str
    message: str


@dataclass
class ToolRuntimeConfig:
    """
    Validated runtime configuration for one analysis tool.
    """

    tool_id: str

    enabled: bool

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    source: str | None = None

    is_custom: bool = False

    explicitly_configured: bool = False


@dataclass
class SensorQARuntimeConfig:
    """
    Fully validated SensorQA runtime configuration.

    This is the object that the application layer can use after
    loading sensorqa.toml.
    """

    application: dict[str, Any]

    pipeline: dict[str, Any]

    tool_discovery: dict[str, Any]

    tools: dict[
        str,
        ToolRuntimeConfig,
    ]

    preferred_tool_order: list[str]

    enabled_tool_ids: list[str]

    tool_parameters: dict[
        str,
        dict[str, Any],
    ]

    custom_tools: dict[str, Any]

    reporting: dict[str, Any]

    output: dict[str, Any]

    logging: dict[str, Any]

    advanced: dict[str, Any]

    file_path: str | None = None

    raw_configuration: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def is_tool_enabled(
        self,
        tool_id: str,
    ) -> bool:
        """Return whether a registered tool is enabled.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Boolean result.
        """

        configuration = self.tools.get(
            tool_id
        )

        if configuration is None:
            return False

        return configuration.enabled

    def parameters_for(
        self,
        tool_id: str,
    ) -> dict[str, Any]:
        """Return a copy of validated runtime parameters for a tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Dictionary containing the result values.
        """

        configuration = self.tools.get(
            tool_id
        )

        if configuration is None:
            return {}

        return dict(
            configuration.parameters
        )


@dataclass
class ConfigLoadReport:
    """
    Result of loading sensorqa.toml.
    """

    success: bool

    config: SensorQARuntimeConfig | None = None

    file_path: str | None = None

    errors: list[
        ConfigLoadError
    ] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def error_count(
        self,
    ) -> int:

        """Return the number of recorded errors.
        
        Returns:
            Calculated value.
        """
        return len(
            self.errors
        )

    @property
    def warning_count(
        self,
    ) -> int:

        """Return the number of recorded warnings.
        
        Returns:
            Calculated value.
        """
        return len(
            self.warnings
        )

    @property
    def enabled_tool_count(
        self,
    ) -> int:

        """Return the number of enabled tools.
        
        Returns:
            Calculated value.
        """
        if self.config is None:
            return 0

        return len(
            self.config.enabled_tool_ids
        )


# Loader


class ConfigLoader:
    """
    Loads and validates SensorQA's main runtime configuration.

    Responsibilities:

        - parse sensorqa.toml
        - validate top-level sections
        - cross-check configured tools with ToolRegistry
        - reject unknown tool IDs
        - reject unknown parameter names
        - validate parameter types
        - validate numeric bounds
        - validate CHOICE values
        - fill unspecified parameters from tool defaults
        - determine enabled tools
        - construct preferred execution order
        - provide pipeline-ready tool_parameters

    This loader does NOT run analysis tools.
    """

    KNOWN_TOP_LEVEL_SECTIONS = {
        "application",
        "pipeline",
        "tool_discovery",
        "tool_order",
        "tools",
        "custom_tools",
        "reporting",
        "output",
        "logging",
        "advanced",
    }

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:

        """Initialize the config loader.
        
        Args:
            registry: Tool registry used for lookups.
        
        Returns:
            None.
        """
        if not isinstance(
            registry,
            ToolRegistry,
        ):

            raise TypeError(
                "registry must be a ToolRegistry."
            )

        self.registry = registry

    # Public API

    def load(
        self,
        source: str | Path,
    ) -> ConfigLoadReport:
        """Load, validate, and normalize sensorqa.toml.
        
        Args:
            source: Input source or source path.
        
        Returns:
            ConfigLoadReport returned by the function.
        """

        path = Path(
            source
        )

        report = ConfigLoadReport(
            success=False,
            file_path=str(
                path
            ),
        )

        # File

        if not path.exists():

            report.errors.append(
                ConfigLoadError(
                    location="file",
                    message=(
                        f"Configuration file does not exist: "
                        f"{path}"
                    ),
                )
            )

            return report

        if not path.is_file():

            report.errors.append(
                ConfigLoadError(
                    location="file",
                    message=(
                        f"Configuration path is not a file: "
                        f"{path}"
                    ),
                )
            )

            return report

        # TOML

        try:

            with path.open(
                "rb"
            ) as file:

                raw = tomllib.load(
                    file
                )

        except (
            OSError,
            tomllib.TOMLDecodeError,
        ) as exc:

            report.errors.append(
                ConfigLoadError(
                    location="file",
                    message=(
                        "Could not parse SensorQA configuration: "
                        f"{exc}"
                    ),
                )
            )

            return report

        if not isinstance(
            raw,
            dict,
        ):

            report.errors.append(
                ConfigLoadError(
                    location="file",
                    message=(
                        "Top-level TOML content must be a table."
                    ),
                )
            )

            return report

        self._warn_unknown_top_level_sections(
            raw=raw,
            report=report,
        )

        # General configuration sections

        application = self._table(
            raw=raw,
            key="application",
            report=report,
        )

        pipeline = self._table(
            raw=raw,
            key="pipeline",
            report=report,
        )

        tool_discovery = self._table(
            raw=raw,
            key="tool_discovery",
            report=report,
        )

        tool_order = self._table(
            raw=raw,
            key="tool_order",
            report=report,
        )

        custom_tools = self._table(
            raw=raw,
            key="custom_tools",
            report=report,
        )

        reporting = self._table(
            raw=raw,
            key="reporting",
            report=report,
        )

        output = self._table(
            raw=raw,
            key="output",
            report=report,
        )

        logging = self._table(
            raw=raw,
            key="logging",
            report=report,
        )

        advanced = self._table(
            raw=raw,
            key="advanced",
            report=report,
        )

        tools_data = self._table(
            raw=raw,
            key="tools",
            report=report,
        )

        # Validate common application sections

        self._validate_application(
            application=application,
            report=report,
        )

        self._validate_pipeline(
            pipeline=pipeline,
            report=report,
        )

        self._validate_tool_discovery(
            section=tool_discovery,
            report=report,
        )

        self._validate_custom_tools(
            section=custom_tools,
            report=report,
        )

        self._validate_reporting(
            section=reporting,
            report=report,
        )

        self._validate_output(
            section=output,
            report=report,
        )

        self._validate_logging(
            section=logging,
            report=report,
        )

        self._validate_advanced(
            section=advanced,
            report=report,
        )

        # Tool configuration

        tool_configurations = self._build_tool_configurations(
            tools_data=tools_data,
            custom_tools_section=custom_tools,
            report=report,
        )

        # Preferred order

        preferred_order = self._parse_preferred_order(
            tool_order=tool_order,
            report=report,
        )

        enabled_tool_ids = self._build_enabled_order(
            tool_configurations=tool_configurations,
            preferred_order=preferred_order,
        )

        # Pipeline-ready parameter dictionary

        tool_parameters = {
            tool_id:
                dict(
                    tool_configurations[
                        tool_id
                    ].parameters
                )
            for tool_id
            in enabled_tool_ids
        }

        if len(
            enabled_tool_ids
        ) == 0:

            report.warnings.append(
                "No analysis tools are enabled."
            )

        # Stop on errors

        if report.errors:

            return report

        runtime_config = SensorQARuntimeConfig(
            application=application,
            pipeline=pipeline,
            tool_discovery=tool_discovery,
            tools=tool_configurations,
            preferred_tool_order=preferred_order,
            enabled_tool_ids=enabled_tool_ids,
            tool_parameters=tool_parameters,
            custom_tools=custom_tools,
            reporting=reporting,
            output=output,
            logging=logging,
            advanced=advanced,
            file_path=str(
                path
            ),
            raw_configuration=raw,
        )

        report.config = runtime_config
        report.success = True

        return report

    # Tool configuration

    def _build_tool_configurations(
        self,
        tools_data: dict[str, Any],
        custom_tools_section: dict[str, Any],
        report: ConfigLoadReport,
    ) -> dict[
        str,
        ToolRuntimeConfig,
    ]:
        """Cross-check [tools.*] sections against the registry.
        
        Args:
            tools_data: Value for `tools_data`.
            custom_tools_section: Value for `custom_tools_section`.
            report: Value for `report`.
        
        Returns:
            Dictionary containing the result values.
        """

        configurations: dict[
            str,
            ToolRuntimeConfig,
        ] = {}

        registry_tool_ids = set(
            self.registry.tool_ids()
        )

        # Configured tools that do not exist

        for configured_tool_id in tools_data:

            if configured_tool_id not in registry_tool_ids:

                report.errors.append(
                    ConfigLoadError(
                        location=(
                            f"tools.{configured_tool_id}"
                        ),
                        message=(
                            f"Configuration references unknown "
                            f"tool '{configured_tool_id}'. "
                            "Check the tool ID or confirm that the "
                            "plug-in was loaded successfully."
                        ),
                    )
                )

        # Build configuration for every registered tool

        for tool_id in self.registry.tool_ids():

            registered = (
                self.registry.get_registered(
                    tool_id
                )
            )

            if registered is None:

                report.errors.append(
                    ConfigLoadError(
                        location=(
                            f"tools.{tool_id}"
                        ),
                        message=(
                            "Tool appears in registry ID list but "
                            "could not be retrieved."
                        ),
                    )
                )

                continue

            configured_section = (
                tools_data.get(
                    tool_id
                )
            )

            explicitly_configured = (
                configured_section is not None
            )

            if explicitly_configured:

                if not isinstance(
                    configured_section,
                    dict,
                ):

                    report.errors.append(
                        ConfigLoadError(
                            location=(
                                f"tools.{tool_id}"
                            ),
                            message=(
                                "Tool configuration must be a "
                                "TOML table."
                            ),
                        )
                    )

                    continue

                configuration = (
                    self._configure_registered_tool(
                        registered=registered,
                        configured_section=(
                            configured_section
                        ),
                        explicitly_configured=True,
                        report=report,
                    )
                )

            else:

                enabled = (
                    self._default_enabled_state(
                        registered=registered,
                        custom_tools_section=(
                            custom_tools_section
                        ),
                    )
                )

                parameters = (
                    self._default_parameters(
                        registered=registered,
                        report=report,
                    )
                )

                configuration = ToolRuntimeConfig(
                    tool_id=tool_id,
                    enabled=enabled,
                    parameters=parameters,
                    source=registered.source,
                    is_custom=registered.is_custom,
                    explicitly_configured=False,
                )

                # Built-in tools are kept disabled when absent from
                # [tools.*]. This avoids silently enabling a newly
                # introduced built-in analysis after an application
                # update.
                if (
                    not registered.is_custom
                    and not explicitly_configured
                ):

                    configuration.enabled = False

            configurations[
                tool_id
            ] = configuration

        return configurations

    def _configure_registered_tool(
        self,
        registered: RegisteredTool,
        configured_section: dict[str, Any],
        explicitly_configured: bool,
        report: ConfigLoadReport,
    ) -> ToolRuntimeConfig:
        """Validate one [tools.<tool_id>] section.
        
        Args:
            registered: Value for `registered`.
            configured_section: Value for `configured_section`.
            explicitly_configured: Value for `explicitly_configured`.
            report: Value for `report`.
        
        Returns:
            ToolRuntimeConfig returned by the function.
        """

        tool = registered.tool
        metadata = tool.metadata
        tool_id = metadata.tool_id

        location = (
            f"tools.{tool_id}"
        )

        enabled = configured_section.get(
            "enabled",
            False,
        )

        if not isinstance(
            enabled,
            bool,
        ):

            report.errors.append(
                ConfigLoadError(
                    location=(
                        f"{location}.enabled"
                    ),
                    message=(
                        "'enabled' must be true or false."
                    ),
                )
            )

            enabled = False

        parameter_definitions = {
            parameter.name:
                parameter
            for parameter
            in metadata.parameters
        }

        # Reject misspelled / unknown parameters

        reserved_keys = {
            "enabled",
        }

        for key in configured_section:

            if key in reserved_keys:
                continue

            if key not in parameter_definitions:

                report.errors.append(
                    ConfigLoadError(
                        location=(
                            f"{location}.{key}"
                        ),
                        message=(
                            f"Unknown parameter '{key}' for tool "
                            f"'{tool_id}'."
                        ),
                    )
                )

        # Build final parameter dictionary

        parameters: dict[
            str,
            Any,
        ] = {}

        for parameter in metadata.parameters:

            if parameter.name in configured_section:

                raw_value = configured_section[
                    parameter.name
                ]

                valid_value = self._validate_parameter_value(
                    parameter=parameter,
                    value=raw_value,
                    location=(
                        f"{location}.{parameter.name}"
                    ),
                    report=report,
                )

                if valid_value is not None:

                    parameters[
                        parameter.name
                    ] = valid_value

            else:

                if parameter.default is not None:

                    parameters[
                        parameter.name
                    ] = parameter.default

                elif parameter.required:

                    report.errors.append(
                        ConfigLoadError(
                            location=(
                                f"{location}.{parameter.name}"
                            ),
                            message=(
                                "Required tool parameter is "
                                "missing and has no default."
                            ),
                        )
                    )

        return ToolRuntimeConfig(
            tool_id=tool_id,
            enabled=enabled,
            parameters=parameters,
            source=registered.source,
            is_custom=registered.is_custom,
            explicitly_configured=(
                explicitly_configured
            ),
        )

    # Parameter validation

    @classmethod
    def _validate_parameter_value(
        cls,
        parameter: ToolParameter,
        value: Any,
        location: str,
        report: ConfigLoadReport,
    ) -> Any | None:
        """Validate a value against ToolParameter.
        
        Args:
            parameter: Value for `parameter`.
            value: Value to process.
            location: Value for `location`.
            report: Value for `report`.
        
        Returns:
            Any | None returned by the function.
        """

        parameter_type = (
            parameter.parameter_type
        )

        # Boolean

        if (
            parameter_type
            == ParameterType.BOOLEAN
        ):

            if not isinstance(
                value,
                bool,
            ):

                report.errors.append(
                    ConfigLoadError(
                        location=location,
                        message=(
                            "Expected a Boolean value "
                            "(true or false)."
                        ),
                    )
                )

                return None

            return value

        # Integer

        if (
            parameter_type
            == ParameterType.INTEGER
        ):

            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    int,
                )
            ):

                report.errors.append(
                    ConfigLoadError(
                        location=location,
                        message=(
                            "Expected an integer value."
                        ),
                    )
                )

                return None

            if not cls._within_bounds(
                parameter=parameter,
                numeric_value=float(
                    value
                ),
                location=location,
                report=report,
            ):

                return None

            return value

        # Float
        #
        # Integer TOML values are accepted for FLOAT parameters
        # because 5 is a valid representation of 5.0.

        if (
            parameter_type
            == ParameterType.FLOAT
        ):

            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
            ):

                report.errors.append(
                    ConfigLoadError(
                        location=location,
                        message=(
                            "Expected a numerical value."
                        ),
                    )
                )

                return None

            numeric_value = float(
                value
            )

            if not math.isfinite(
                numeric_value
            ):

                report.errors.append(
                    ConfigLoadError(
                        location=location,
                        message=(
                            "Numerical parameter must be finite."
                        ),
                    )
                )

                return None

            if not cls._within_bounds(
                parameter=parameter,
                numeric_value=numeric_value,
                location=location,
                report=report,
            ):

                return None

            return numeric_value

        # String

        if (
            parameter_type
            == ParameterType.STRING
        ):

            if not isinstance(
                value,
                str,
            ):

                report.errors.append(
                    ConfigLoadError(
                        location=location,
                        message=(
                            "Expected a string value."
                        ),
                    )
                )

                return None

            if (
                parameter.required
                and not value.strip()
            ):

                report.errors.append(
                    ConfigLoadError(
                        location=location,
                        message=(
                            "Required string parameter cannot "
                            "be empty."
                        ),
                    )
                )

                return None

            return value

        # Choice

        if (
            parameter_type
            == ParameterType.CHOICE
        ):

            choices = (
                parameter.choices
                or []
            )

            if value not in choices:

                report.errors.append(
                    ConfigLoadError(
                        location=location,
                        message=(
                            f"Invalid choice '{value}'. "
                            "Allowed values: "
                            + ", ".join(
                                repr(
                                    choice
                                )
                                for choice in choices
                            )
                        ),
                    )
                )

                return None

            return value

        report.errors.append(
            ConfigLoadError(
                location=location,
                message=(
                    "Unsupported parameter type "
                    f"'{parameter_type}'."
                ),
            )
        )

        return None

    @staticmethod
    def _within_bounds(
        parameter: ToolParameter,
        numeric_value: float,
        location: str,
        report: ConfigLoadReport,
    ) -> bool:
        """Validate optional ToolParameter minimum/maximum bounds.
        
        Args:
            parameter: Value for `parameter`.
            numeric_value: Value for `numeric_value`.
            location: Value for `location`.
            report: Value for `report`.
        
        Returns:
            Boolean result.
        """

        if (
            parameter.minimum is not None
            and numeric_value
            < parameter.minimum
        ):

            report.errors.append(
                ConfigLoadError(
                    location=location,
                    message=(
                        f"Value {numeric_value} is below minimum "
                        f"{parameter.minimum}."
                    ),
                )
            )

            return False

        if (
            parameter.maximum is not None
            and numeric_value
            > parameter.maximum
        ):

            report.errors.append(
                ConfigLoadError(
                    location=location,
                    message=(
                        f"Value {numeric_value} exceeds maximum "
                        f"{parameter.maximum}."
                    ),
                )
            )

            return False

        return True

    # Default parameters

    @staticmethod
    def _default_parameters(
        registered: RegisteredTool,
        report: ConfigLoadReport,
    ) -> dict[str, Any]:
        """Construct default parameters for a registered tool.
        
        Args:
            registered: Value for `registered`.
            report: Value for `report`.
        
        Returns:
            Dictionary containing the result values.
        """

        parameters = {}

        for parameter in (
            registered.tool.metadata.parameters
        ):

            if parameter.default is not None:

                parameters[
                    parameter.name
                ] = parameter.default

            elif parameter.required:

                report.errors.append(
                    ConfigLoadError(
                        location=(
                            f"tools."
                            f"{registered.tool_id}."
                            f"{parameter.name}"
                        ),
                        message=(
                            "Tool declares a required parameter "
                            "without a default, but no explicit "
                            "configuration section is present."
                        ),
                    )
                )

        return parameters

    # Custom-tool defaults

    @staticmethod
    def _default_enabled_state(
        registered: RegisteredTool,
        custom_tools_section: dict[str, Any],
    ) -> bool:
        """Determine state for a registered tool without an explicit
        [tools.<tool_id>] section.
        
        Built-in tools default to disabled when unconfigured.
        
        Unconfigured custom tools require BOTH:
        
            enable_unconfigured_custom_tools = true
        
        and:
        
            default_enabled = true
        
        This makes accidental execution of newly discovered Python
        plug-ins unlikely.
        
        Args:
            registered: Value for `registered`.
            custom_tools_section: Value for `custom_tools_section`.
        
        Returns:
            Boolean result.
        """

        if not registered.is_custom:

            return False

        allow_unconfigured = custom_tools_section.get(
            "enable_unconfigured_custom_tools",
            False,
        )

        default_enabled = custom_tools_section.get(
            "default_enabled",
            False,
        )

        return (
            allow_unconfigured is True
            and default_enabled is True
        )

    # Preferred order

    def _parse_preferred_order(
        self,
        tool_order: dict[str, Any],
        report: ConfigLoadReport,
    ) -> list[str]:
        """Validate [tool_order].preferred.
        
        Args:
            tool_order: Value for `tool_order`.
            report: Value for `report`.
        
        Returns:
            List of result values.
        """

        preferred = tool_order.get(
            "preferred",
            [],
        )

        if not isinstance(
            preferred,
            list,
        ):

            report.errors.append(
                ConfigLoadError(
                    location=(
                        "tool_order.preferred"
                    ),
                    message=(
                        "'preferred' must be an array of tool IDs."
                    ),
                )
            )

            return []

        output = []

        seen = set()

        registered_ids = set(
            self.registry.tool_ids()
        )

        for index, value in enumerate(
            preferred
        ):

            location = (
                f"tool_order.preferred[{index}]"
            )

            if not isinstance(
                value,
                str,
            ) or not value.strip():

                report.errors.append(
                    ConfigLoadError(
                        location=location,
                        message=(
                            "Preferred tool ID must be a "
                            "non-empty string."
                        ),
                    )
                )

                continue

            tool_id = value.strip()

            if tool_id in seen:

                report.warnings.append(
                    f"Tool '{tool_id}' appears more than once "
                    "in preferred tool order."
                )

                continue

            seen.add(
                tool_id
            )

            if tool_id not in registered_ids:

                report.errors.append(
                    ConfigLoadError(
                        location=location,
                        message=(
                            f"Preferred order references unknown "
                            f"tool '{tool_id}'."
                        ),
                    )
                )

                continue

            output.append(
                tool_id
            )

        return output

    @staticmethod
    def _build_enabled_order(
        tool_configurations: dict[
            str,
            ToolRuntimeConfig,
        ],
        preferred_order: list[str],
    ) -> list[str]:
        """Build final enabled execution request order.
        
        Dependency ordering remains the responsibility of
        AnalysisPipeline's topological sort.
        
        Args:
            tool_configurations: Value for `tool_configurations`.
            preferred_order: Value for `preferred_order`.
        
        Returns:
            List of result values.
        """

        enabled = {
            tool_id
            for tool_id, configuration
            in tool_configurations.items()
            if configuration.enabled
        }

        ordered = [
            tool_id
            for tool_id
            in preferred_order
            if tool_id in enabled
        ]

        already_added = set(
            ordered
        )

        # Dict insertion order follows ToolRegistry.tool_ids().
        # This provides a deterministic fallback order.
        for tool_id, configuration in (
            tool_configurations.items()
        ):

            if (
                configuration.enabled
                and tool_id
                not in already_added
            ):

                ordered.append(
                    tool_id
                )

                already_added.add(
                    tool_id
                )

        return ordered

    # Application section validation

    @staticmethod
    def _validate_application(
        application: dict[str, Any],
        report: ConfigLoadReport,
    ) -> None:

        """Validate application.
        
        Args:
            application: Initialized SensorQA application.
            report: Report used by this function.
        
        Returns:
            None.
        """
        ConfigLoader._optional_string(
            section=application,
            key="name",
            location="application",
            report=report,
        )

        ConfigLoader._optional_string(
            section=application,
            key="config_version",
            location="application",
            report=report,
        )

        sensor_type = application.get(
            "default_sensor_type"
        )

        if sensor_type is not None:

            allowed = {
                "generic",
                "accelerometer",
                "gyroscope",
                "imu",
            }

            if (
                not isinstance(
                    sensor_type,
                    str,
                )
                or sensor_type
                not in allowed
            ):

                report.errors.append(
                    ConfigLoadError(
                        location=(
                            "application.default_sensor_type"
                        ),
                        message=(
                            "default_sensor_type must be one of: "
                            + ", ".join(
                                sorted(
                                    allowed
                                )
                            )
                        ),
                    )
                )

    # Pipeline validation

    @staticmethod
    def _validate_pipeline(
        pipeline: dict[str, Any],
        report: ConfigLoadReport,
    ) -> None:

        """Validate pipeline.
        
        Args:
            pipeline: Pipeline used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        for key in (
            "allow_invalid_dataset",
            "stop_on_tool_error",
            "continue_on_skipped_tool",
            "show_disabled_tools",
        ):

            ConfigLoader._optional_boolean(
                section=pipeline,
                key=key,
                location="pipeline",
                report=report,
            )

    # Discovery validation

    @staticmethod
    def _validate_tool_discovery(
        section: dict[str, Any],
        report: ConfigLoadReport,
    ) -> None:

        """Validate tool discovery.
        
        Args:
            section: Section used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        for key in (
            "load_generic_tools",
            "load_imu_tools",
            "load_custom_tools",
            "continue_on_custom_tool_error",
        ):

            ConfigLoader._optional_boolean(
                section=section,
                key=key,
                location="tool_discovery",
                report=report,
            )

    # Custom-tool validation

    @staticmethod
    def _validate_custom_tools(
        section: dict[str, Any],
        report: ConfigLoadReport,
    ) -> None:

        """Validate custom tools.
        
        Args:
            section: Section used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        for key in (
            "default_enabled",
            "enable_unconfigured_custom_tools",
        ):

            ConfigLoader._optional_boolean(
                section=section,
                key=key,
                location="custom_tools",
                report=report,
            )

    # Reporting validation

    @staticmethod
    def _validate_reporting(
        section: dict[str, Any],
        report: ConfigLoadReport,
    ) -> None:

        """Validate reporting.
        
        Args:
            section: Section used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        for key in (
            "include_warnings",
            "include_messages",
            "include_evidence",
            "include_tool_metadata",
            "include_configuration_snapshot",
            "include_disabled_requirements",
            "include_raw_curve_data_in_json",
            "include_raw_curve_data_in_human_report",
        ):

            ConfigLoader._optional_boolean(
                section=section,
                key=key,
                location="reporting",
                report=report,
            )

    # Output validation

    @staticmethod
    def _validate_output(
        section: dict[str, Any],
        report: ConfigLoadReport,
    ) -> None:

        """Validate output.
        
        Args:
            section: Section used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        ConfigLoader._optional_string(
            section=section,
            key="default_directory",
            location="output",
            report=report,
        )

        for key in (
            "save_json_results",
            "save_csv_metrics",
            "save_plots",
            "save_human_report",
            "overwrite_existing",
        ):

            ConfigLoader._optional_boolean(
                section=section,
                key=key,
                location="output",
                report=report,
            )

    # Logging validation

    @staticmethod
    def _validate_logging(
        section: dict[str, Any],
        report: ConfigLoadReport,
    ) -> None:

        """Validate logging.
        
        Args:
            section: Section used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        level = section.get(
            "level"
        )

        if level is not None:

            allowed_levels = {
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            }

            if (
                not isinstance(
                    level,
                    str,
                )
                or level.upper()
                not in allowed_levels
            ):

                report.errors.append(
                    ConfigLoadError(
                        location="logging.level",
                        message=(
                            "Logging level must be one of: "
                            + ", ".join(
                                sorted(
                                    allowed_levels
                                )
                            )
                        ),
                    )
                )

            elif isinstance(
                level,
                str,
            ):

                section[
                    "level"
                ] = level.upper()

        ConfigLoader._optional_boolean(
            section=section,
            key="log_to_console",
            location="logging",
            report=report,
        )

        ConfigLoader._optional_boolean(
            section=section,
            key="log_to_file",
            location="logging",
            report=report,
        )

        ConfigLoader._optional_string(
            section=section,
            key="log_directory",
            location="logging",
            report=report,
        )

    # Advanced validation

    @staticmethod
    def _validate_advanced(
        section: dict[str, Any],
        report: ConfigLoadReport,
    ) -> None:

        """Validate advanced.
        
        Args:
            section: Section used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        for key in (
            "preserve_original_data",
            "record_reproducibility_metadata",
            "allow_implicit_resampling",
            "warn_before_enabling_new_custom_tool",
        ):

            ConfigLoader._optional_boolean(
                section=section,
                key=key,
                location="advanced",
                report=report,
            )

    # General section helpers

    @staticmethod
    def _table(
        raw: dict[str, Any],
        key: str,
        report: ConfigLoadReport,
    ) -> dict[str, Any]:
        """Return a top-level configuration table.
        
        Missing optional tables become empty dictionaries.
        
        Args:
            raw: Value for `raw`.
            key: Value for `key`.
            report: Value for `report`.
        
        Returns:
            Dictionary containing the result values.
        """

        value = raw.get(
            key
        )

        if value is None:

            return {}

        if not isinstance(
            value,
            dict,
        ):

            report.errors.append(
                ConfigLoadError(
                    location=key,
                    message=(
                        f"[{key}] must be a TOML table."
                    ),
                )
            )

            return {}

        return dict(
            value
        )

    @classmethod
    def _warn_unknown_top_level_sections(
        cls,
        raw: dict[str, Any],
        report: ConfigLoadReport,
    ) -> None:

        """Return warn unknown top level sections.
        
        Args:
            raw: Raw used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        for key in raw:

            if (
                key
                not in cls.KNOWN_TOP_LEVEL_SECTIONS
            ):

                report.warnings.append(
                    f"Unknown top-level configuration section "
                    f"'{key}'."
                )

    @staticmethod
    def _optional_boolean(
        section: dict[str, Any],
        key: str,
        location: str,
        report: ConfigLoadReport,
    ) -> None:

        """Read optional boolean.
        
        Args:
            section: Section used by this function.
            key: Key used by this function.
            location: Location used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        if key not in section:
            return

        if not isinstance(
            section[
                key
            ],
            bool,
        ):

            report.errors.append(
                ConfigLoadError(
                    location=(
                        f"{location}.{key}"
                    ),
                    message=(
                        "Expected true or false."
                    ),
                )
            )

    @staticmethod
    def _optional_string(
        section: dict[str, Any],
        key: str,
        location: str,
        report: ConfigLoadReport,
    ) -> None:

        """Read optional string.
        
        Args:
            section: Section used by this function.
            key: Key used by this function.
            location: Location used by this function.
            report: Report used by this function.
        
        Returns:
            None.
        """
        if key not in section:
            return

        value = section[
            key
        ]

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):

            report.errors.append(
                ConfigLoadError(
                    location=(
                        f"{location}.{key}"
                    ),
                    message=(
                        "Expected a non-empty string."
                    ),
                )
            )
