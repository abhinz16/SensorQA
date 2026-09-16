#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field

from sensorqa.core.tool_contract import (
    BaseAnalysisTool,
    SensorType,
    ToolCategory,
)
from sensorqa.core.tool_loader import (
    LoadedTool,
    ToolLoadError,
    ToolLoadReport,
)
from sensorqa.core.tool_validator import (
    ToolDefinitionValidation,
    ToolValidator,
)


@dataclass
class RegisteredTool:
    """
    Represents one validated tool accepted into the SensorQA registry.
    """

    tool: BaseAnalysisTool
    file_path: str
    source: str

    validation_warnings: list[str] = field(default_factory=list)

    @property
    def tool_id(self) -> str:
        """Return the registered tool identifier.
        
        Returns:
            Requested text value.
        """
        return self.tool.metadata.tool_id

    @property
    def name(self) -> str:
        """Return the registered tool name.
        
        Returns:
            Requested text value.
        """
        return self.tool.metadata.name

    @property
    def version(self) -> str:
        """Return the tool version.
        
        Returns:
            Requested text value.
        """
        return self.tool.metadata.version

    @property
    def category(self) -> ToolCategory:
        """Return the tool category.
        
        Returns:
            ToolCategory returned by this function.
        """
        return self.tool.metadata.category

    @property
    def compatible_sensor_types(self) -> list[SensorType]:
        """Return the sensor types supported by the tool.
        
        Returns:
            List of result values.
        """
        return self.tool.metadata.compatible_sensor_types

    @property
    def is_custom(self) -> bool:
        """Return whether the tool came from the custom-tools directory.
        
        Returns:
            True when the condition is met; otherwise False.
        """
        return self.source == "custom"


@dataclass
class RejectedTool:
    """
    Represents a tool that was found but rejected by SensorQA.
    """

    file_path: str
    tool_id: str | None
    tool_name: str | None

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class RegistryBuildReport:
    """
    Summary produced when rebuilding the complete tool registry.
    """

    registered_tools: list[RegisteredTool] = field(default_factory=list)

    rejected_tools: list[RejectedTool] = field(default_factory=list)

    load_errors: list[ToolLoadError] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)

    @property
    def registered_count(self) -> int:
        """Return the number of registered tools.
        
        Returns:
            Calculated value.
        """
        return len(self.registered_tools)

    @property
    def rejected_count(self) -> int:
        """Return the number of rejected tools.
        
        Returns:
            Calculated value.
        """
        return len(self.rejected_tools)

    @property
    def load_error_count(self) -> int:
        """Return the number of tool-loading errors.
        
        Returns:
            Calculated value.
        """
        return len(self.load_errors)


class ToolRegistry:
    """
    Central registry containing every validated SensorQA analysis tool.

    Other parts of SensorQA should access analysis tools through
    this registry rather than scanning tool directories directly.
    """

    def __init__(
        self,
        validator: ToolValidator | None = None,
    ) -> None:

        """Initialize the tool registry.
        
        Args:
            validator: Validator used by this function.
        
        Returns:
            None.
        """
        self.validator = validator or ToolValidator()

        self._tools: dict[str, RegisteredTool] = {}

        self._rejected_tools: list[RejectedTool] = []

        self._load_errors: list[ToolLoadError] = []

        self._warnings: list[str] = []

    def rebuild(
        self,
        load_report: ToolLoadReport,
    ) -> RegistryBuildReport:
        """Rebuild the registry from a ToolLoadReport.
        
        Existing registry contents are cleared first.
        
        Args:
            load_report: Tool loading report.
        
        Returns:
            RegistryBuildReport returned by the function.
        """

        self.clear()

        self._load_errors.extend(load_report.errors)
        self._warnings.extend(load_report.warnings)

        for loaded_tool in load_report.loaded_tools:

            self._process_loaded_tool(loaded_tool)

        self._validate_dependencies()

        return self.build_report()

    def register(
        self,
        loaded_tool: LoadedTool,
    ) -> bool:
        """Validate and register one tool manually.
        
        Returns True if registration succeeds.
        
        Args:
            loaded_tool: Value for `loaded_tool`.
        
        Returns:
            Boolean result.
        """

        return self._process_loaded_tool(loaded_tool)

    def unregister(
        self,
        tool_id: str,
    ) -> bool:
        """Remove a tool from the registry.
        
        Returns True if the tool existed.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Boolean result.
        """

        if tool_id not in self._tools:
            return False

        del self._tools[tool_id]

        return True

    def get(
        self,
        tool_id: str,
    ) -> BaseAnalysisTool | None:
        """Return the analysis tool associated with a tool ID.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            BaseAnalysisTool | None returned by the function.
        """

        registered = self._tools.get(tool_id)

        if registered is None:
            return None

        return registered.tool

    def get_registered(
        self,
        tool_id: str,
    ) -> RegisteredTool | None:
        """Return the complete RegisteredTool record.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            RegisteredTool | None returned by the function.
        """

        return self._tools.get(tool_id)

    def require(
        self,
        tool_id: str,
    ) -> BaseAnalysisTool:
        """Return a tool or raise KeyError if it is not registered.
        
        Useful when the caller expects the tool to exist.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            BaseAnalysisTool returned by the function.
        """

        tool = self.get(tool_id)

        if tool is None:
            raise KeyError(
                f"Tool '{tool_id}' is not registered."
            )

        return tool

    def contains(
        self,
        tool_id: str,
    ) -> bool:
        """Return True if the registry contains the tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Boolean result.
        """

        return tool_id in self._tools

    def all_tools(self) -> list[RegisteredTool]:
        """Return every registered tool sorted by display name.
        
        Returns:
            List of result values.
        """

        return sorted(
            self._tools.values(),
            key=lambda registered: registered.name.lower(),
        )

    def tool_ids(self) -> list[str]:
        """Return all registered tool IDs.
        
        Returns:
            List of result values.
        """

        return sorted(self._tools.keys())

    def custom_tools(self) -> list[RegisteredTool]:
        """Return only user-created custom tools.
        
        Returns:
            List of result values.
        """

        return [
            registered
            for registered in self.all_tools()
            if registered.is_custom
        ]

    def built_in_tools(self) -> list[RegisteredTool]:
        """Return only built-in SensorQA tools.
        
        Returns:
            List of result values.
        """

        return [
            registered
            for registered in self.all_tools()
            if not registered.is_custom
        ]

    def by_sensor_type(
        self,
        sensor_type: SensorType,
    ) -> list[RegisteredTool]:
        """Return tools compatible with a particular sensor type.
        
        Args:
            sensor_type: Sensor type used to choose compatible tools.
        
        Returns:
            List of result values.
        """

        compatible: list[RegisteredTool] = []

        for registered in self.all_tools():

            if registered.tool.metadata.supports_sensor_type(
                sensor_type
            ):
                compatible.append(registered)

        return compatible

    def by_category(
        self,
        category: ToolCategory,
    ) -> list[RegisteredTool]:
        """Return tools belonging to a particular category.
        
        Args:
            category: Value for `category`.
        
        Returns:
            List of result values.
        """

        return [
            registered
            for registered in self.all_tools()
            if registered.category == category
        ]

    def rejected_tools(self) -> list[RejectedTool]:
        """Return tools rejected during validation.
        
        Returns:
            List of result values.
        """

        return list(self._rejected_tools)

    def load_errors(self) -> list[ToolLoadError]:
        """Return file-level loading failures.
        
        Returns:
            List of result values.
        """

        return list(self._load_errors)

    def warnings(self) -> list[str]:
        """Return registry-level warnings.
        
        Returns:
            List of result values.
        """

        return list(self._warnings)

    def clear(self) -> None:
        """Completely clear the registry and previous reports.
        
        Returns:
            None.
        """

        self._tools.clear()
        self._rejected_tools.clear()
        self._load_errors.clear()
        self._warnings.clear()

    def build_report(self) -> RegistryBuildReport:
        """Return the current registry state as a report.
        
        Returns:
            RegistryBuildReport returned by the function.
        """

        return RegistryBuildReport(
            registered_tools=self.all_tools(),
            rejected_tools=list(self._rejected_tools),
            load_errors=list(self._load_errors),
            warnings=list(self._warnings),
        )

    def _process_loaded_tool(
        self,
        loaded_tool: LoadedTool,
    ) -> bool:
        """Validate one loaded tool and add it to the registry
        if it passes validation.
        
        Args:
            loaded_tool: Value for `loaded_tool`.
        
        Returns:
            Boolean result.
        """

        tool = loaded_tool.tool

        validation: ToolDefinitionValidation = (
            self.validator.validate(tool)
        )

        # Tool definition failed validation

        if not validation.valid:

            tool_id: str | None = None
            tool_name: str | None = None

            try:
                tool_id = tool.metadata.tool_id
                tool_name = tool.metadata.name
            except Exception:
                pass

            self._rejected_tools.append(
                RejectedTool(
                    file_path=loaded_tool.file_path,
                    tool_id=tool_id,
                    tool_name=tool_name,
                    errors=validation.errors,
                    warnings=validation.warnings,
                )
            )

            return False

        tool_id = tool.metadata.tool_id

        # Protect against duplicate IDs

        if tool_id in self._tools:

            existing = self._tools[tool_id]

            self._rejected_tools.append(
                RejectedTool(
                    file_path=loaded_tool.file_path,
                    tool_id=tool_id,
                    tool_name=tool.metadata.name,
                    errors=[
                        (
                            f"Tool ID '{tool_id}' is already registered "
                            f"from '{existing.file_path}'."
                        )
                    ],
                    warnings=validation.warnings,
                )
            )

            return False

        # Accept tool into registry

        self._tools[tool_id] = RegisteredTool(
            tool=tool,
            file_path=loaded_tool.file_path,
            source=loaded_tool.source,
            validation_warnings=validation.warnings,
        )

        return True

    def _validate_dependencies(self) -> None:
        """Check whether registered tool dependencies actually exist.
        
        Tools with missing dependencies are removed from the active
        registry and reported as rejected.
        
        Returns:
            None.
        """

        tools_to_reject: list[
            tuple[str, list[str]]
        ] = []

        for tool_id, registered in self._tools.items():

            dependencies = (
                registered.tool.metadata.dependencies
            )

            missing = [
                dependency
                for dependency in dependencies
                if dependency not in self._tools
            ]

            if missing:
                tools_to_reject.append(
                    (
                        tool_id,
                        missing,
                    )
                )

        for tool_id, missing_dependencies in tools_to_reject:

            registered = self._tools.pop(tool_id)

            self._rejected_tools.append(
                RejectedTool(
                    file_path=registered.file_path,
                    tool_id=tool_id,
                    tool_name=registered.name,
                    errors=[
                        (
                            "Missing required tool dependencies: "
                            + ", ".join(
                                sorted(missing_dependencies)
                            )
                        )
                    ],
                    warnings=registered.validation_warnings,
                )
            )
