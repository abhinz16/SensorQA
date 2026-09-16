#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path

from sensorqa.core.tool_contract import BaseAnalysisTool


@dataclass
class ToolLoadError:
    """
    Describes a tool file that SensorQA was unable to load.
    """

    file_path: str
    error_type: str
    message: str


@dataclass
class LoadedTool:
    """
    Represents one successfully loaded analysis tool.
    """

    tool: BaseAnalysisTool
    file_path: str
    source: str

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


@dataclass
class ToolLoadReport:
    """
    Complete result of scanning one or more tool directories.
    """

    loaded_tools: list[LoadedTool] = field(default_factory=list)
    errors: list[ToolLoadError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def loaded_count(self) -> int:
        """Return the number of tools loaded from disk.
        
        Returns:
            Calculated value.
        """
        return len(self.loaded_tools)

    @property
    def error_count(self) -> int:
        """Return the number of recorded errors.
        
        Returns:
            Calculated value.
        """
        return len(self.errors)


class ToolLoader:
    """
    Discovers and loads SensorQA analysis tools from Python files.

    Built-in and custom tools are loaded using the same mechanism.
    """

    def __init__(
        self,
        generic_tools_dir: str | Path,
        imu_tools_dir: str | Path,
        custom_tools_dir: str | Path,
    ) -> None:

        """Initialize the tool loader.
        
        Args:
            generic_tools_dir: Generic tools dir used by this function.
            imu_tools_dir: Imu tools dir used by this function.
            custom_tools_dir: Custom tools dir used by this function.
        
        Returns:
            None.
        """
        self.tool_directories = {
            "built_in_generic": Path(generic_tools_dir),
            "built_in_imu": Path(imu_tools_dir),
            "custom": Path(custom_tools_dir),
        }

    def discover_tools(self) -> ToolLoadReport:
        """Scan all configured tool directories and attempt to load
        every Python analysis tool found.
        
        Invalid tools are reported instead of crashing SensorQA.
        
        Returns:
            ToolLoadReport returned by the function.
        """

        report = ToolLoadReport()

        seen_tool_ids: dict[str, str] = {}

        for source, directory in self.tool_directories.items():

            if not directory.exists():
                report.warnings.append(
                    f"Tool directory does not exist: {directory}"
                )
                continue

            if not directory.is_dir():
                report.warnings.append(
                    f"Tool path is not a directory: {directory}"
                )
                continue

            python_files = self._find_python_files(directory)

            for file_path in python_files:

                try:
                    tools = self._load_tools_from_file(file_path)

                    if not tools:
                        report.errors.append(
                            ToolLoadError(
                                file_path=str(file_path),
                                error_type="NoToolFound",
                                message=(
                                    "No class inheriting from "
                                    "BaseAnalysisTool was found."
                                ),
                            )
                        )
                        continue

                    for tool in tools:

                        tool_id = tool.metadata.tool_id

                        if tool_id in seen_tool_ids:
                            report.errors.append(
                                ToolLoadError(
                                    file_path=str(file_path),
                                    error_type="DuplicateToolID",
                                    message=(
                                        f"Tool ID '{tool_id}' is already used by "
                                        f"{seen_tool_ids[tool_id]}."
                                    ),
                                )
                            )
                            continue

                        seen_tool_ids[tool_id] = str(file_path)

                        report.loaded_tools.append(
                            LoadedTool(
                                tool=tool,
                                file_path=str(file_path),
                                source=source,
                            )
                        )

                except Exception as exc:

                    report.errors.append(
                        ToolLoadError(
                            file_path=str(file_path),
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )

        return report

    def reload_tools(self) -> ToolLoadReport:
        """Re-scan all tool directories.
        
        This is intended for the future 'Reload Tools' button
        in the SensorQA user interface.
        
        Returns:
            ToolLoadReport returned by the function.
        """

        return self.discover_tools()

    @staticmethod
    def _find_python_files(directory: Path) -> list[Path]:
        """Find Python files in a tool directory.
        
        Private files and __init__.py are ignored.
        
        Args:
            directory: Value for `directory`.
        
        Returns:
            Resolved path.
        """

        files: list[Path] = []

        for file_path in sorted(directory.glob("*.py")):

            if file_path.name == "__init__.py":
                continue

            if file_path.name.startswith("_"):
                continue

            files.append(file_path)

        return files

    def _load_tools_from_file(
        self,
        file_path: Path,
    ) -> list[BaseAnalysisTool]:
        """Dynamically import a Python file and instantiate all valid
        BaseAnalysisTool subclasses defined inside it.
        
        Args:
            file_path: Path to the file.
        
        Returns:
            List of result values.
        """

        module = self._import_module(file_path)

        tool_classes = self._find_tool_classes(module)

        tools: list[BaseAnalysisTool] = []

        for tool_class in tool_classes:

            try:
                instance = tool_class()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not instantiate tool class "
                    f"'{tool_class.__name__}': {exc}"
                ) from exc

            tools.append(instance)

        return tools

    @staticmethod
    def _find_tool_classes(module) -> list[type[BaseAnalysisTool]]:
        """Find concrete classes in a module that inherit from
        BaseAnalysisTool.
        
        Imported BaseAnalysisTool subclasses from other modules
        are ignored. Only classes defined in the current file
        are considered.
        
        Args:
            module: Value for `module`.
        
        Returns:
            List of result values.
        """

        discovered_classes: list[type[BaseAnalysisTool]] = []

        for _, obj in inspect.getmembers(module, inspect.isclass):

            if obj is BaseAnalysisTool:
                continue

            if not issubclass(obj, BaseAnalysisTool):
                continue

            # Ignore abstract classes.
            if inspect.isabstract(obj):
                continue

            # Only load classes actually defined in this module.
            if obj.__module__ != module.__name__:
                continue

            discovered_classes.append(obj)

        return discovered_classes

    @staticmethod
    def _import_module(file_path: Path):
        """Import a Python file dynamically.
        
        A unique module name is generated from the full path
        so files with identical names in different directories
        do not conflict.
        
        Args:
            file_path: Path to the file.
        
        Returns:
            Result returned by the function.
        """

        resolved_path = file_path.resolve()

        path_hash = hashlib.sha256(
            str(resolved_path).encode("utf-8")
        ).hexdigest()[:12]

        module_name = (
            f"sensorqa_dynamic_tool_"
            f"{file_path.stem}_{path_hash}"
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            resolved_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to create import specification for "
                f"{resolved_path}."
            )

        module = importlib.util.module_from_spec(spec)

        # Replace an older copy during reload.
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)

        except Exception:
            # Avoid leaving a partially imported module behind.
            sys.modules.pop(module_name, None)
            raise

        return module
