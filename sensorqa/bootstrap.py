#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sensorqa.core.application import (
    ApplicationInitializationResult,
    SensorQAApplication,
)
from sensorqa.core.pipeline import AnalysisPipeline
from sensorqa.core.tool_loader import ToolLoader
from sensorqa.core.tool_registry import (
    RegistryBuildReport,
    ToolRegistry,
)
from sensorqa.core.tool_validator import ToolValidator
from sensorqa.ingestion.column_mapper import ColumnMapper
from sensorqa.ingestion.csv_loader import CSVLoader
from sensorqa.ingestion.dataset_builder import DatasetBuilder
from sensorqa.ingestion.dataset_validator import DatasetValidator
from sensorqa.ingestion.unit_manager import UnitManager


# SensorQA filesystem paths


@dataclass(frozen=True)
class SensorQAPaths:
    """
    Resolved filesystem locations used by one SensorQA instance.

    Keeping path resolution in one place is important because the
    desktop app, CLI, tests, and API should all use the
    same backend resources.
    """

    project_root: Path

    configs_dir: Path
    tools_dir: Path

    generic_tools_dir: Path
    imu_tools_dir: Path
    custom_tools_dir: Path

    runtime_config_path: Path
    requirements_path: Path

    @classmethod
    def from_root(
        cls,
        project_root: str | Path | None = None,
    ) -> "SensorQAPaths":
        """Resolve the SensorQA project/package root.
        
        project_root may point directly to:
        
            SensorQA/sensorqa/
        
        where configs/ and tools/ exist,
        
        or to its parent:
        
            SensorQA/
        
        containing:
        
            SensorQA/sensorqa/configs/
            SensorQA/sensorqa/tools/
        
        If no root is supplied, the location of this bootstrap.py file
        is used.
        
        Args:
            project_root: Value for `project_root`.
        
        Returns:
            Resolved path.
        """

        if project_root is None:

            root = (
                Path(__file__)
                .resolve()
                .parent
            )

        else:

            root = (
                Path(project_root)
                .expanduser()
                .resolve()
            )

        # Direct SensorQA root

        if not _looks_like_sensorqa_root(
            root
        ):

            # Parent directory containing sensorqa/

            nested = (
                root
                / "sensorqa"
            )

            if _looks_like_sensorqa_root(
                nested
            ):

                root = nested

            else:

                raise FileNotFoundError(
                    "Could not locate a SensorQA root containing "
                    "both 'configs' and 'tools'. "
                    f"Checked: {root} and {nested}"
                )

        configs_dir = (
            root
            / "configs"
        )

        tools_dir = (
            root
            / "tools"
        )

        return cls(
            project_root=root,

            configs_dir=configs_dir,

            tools_dir=tools_dir,

            generic_tools_dir=(
                tools_dir
                / "generic"
            ),

            imu_tools_dir=(
                tools_dir
                / "imu"
            ),

            custom_tools_dir=(
                tools_dir
                / "custom"
            ),

            runtime_config_path=(
                configs_dir
                / "sensorqa.toml"
            ),

            requirements_path=(
                configs_dir
                / "requirements.toml"
            ),
        )

    def validate(
        self,
    ) -> list[str]:
        """Check required SensorQA files/directories.
        
        Returns human-readable problems instead of immediately raising.
        This is useful later for displaying startup errors in the GUI.
        
        Returns:
            List of result values.
        """

        problems: list[str] = []

        required_directories = {
            "configs":
                self.configs_dir,

            "tools":
                self.tools_dir,

            "generic tools":
                self.generic_tools_dir,

            "IMU tools":
                self.imu_tools_dir,
        }

        for (
            label,
            path,
        ) in required_directories.items():

            if not path.is_dir():

                problems.append(
                    f"Required {label} directory "
                    f"does not exist: {path}"
                )

        required_files = {
            "runtime configuration":
                self.runtime_config_path,

            "requirements configuration":
                self.requirements_path,
        }

        for (
            label,
            path,
        ) in required_files.items():

            if not path.is_file():

                problems.append(
                    f"Required {label} file "
                    f"does not exist: {path}"
                )

        return problems


# Complete bootstrap result


@dataclass
class SensorQABootstrapResult:
    """
    Complete startup context for a SensorQA process.

    This object keeps the startup details the desktop app needs.

    The UI will be able to inspect:

        - the application service
        - registered tools
        - rejected tools
        - plug-in loading errors
        - configuration initialization
        - startup warnings
        - resolved paths

    without reconstructing the backend itself.
    """

    application: SensorQAApplication

    paths: SensorQAPaths

    tool_loader: ToolLoader

    registry: ToolRegistry

    registry_report: RegistryBuildReport

    initialization: (
        ApplicationInitializationResult
        | None
    ) = None

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def initialized(
        self,
    ) -> bool:
        """True only when both the initialization result and application
        state confirm successful startup.
        
        Returns:
            Boolean result.
        """

        return bool(
            self.initialization
            is not None
            and self.initialization.success
            and self.application.initialized
        )

    @property
    def registered_tool_ids(
        self,
    ) -> list[str]:
        """All successfully registered SensorQA tool IDs.
        
        Returns:
            List of result values.
        """

        return (
            self.registry.tool_ids()
        )

    @property
    def rejected_tool_count(
        self,
    ) -> int:
        """Number of discovered tools rejected during validation.
        
        Returns:
            Integer result.
        """

        return (
            self.registry_report
            .rejected_count
        )

    @property
    def load_error_count(
        self,
    ) -> int:
        """Number of tool files that could not be loaded.
        
        Returns:
            Integer result.
        """

        return (
            self.registry_report
            .load_error_count
        )


# Path helpers


def _looks_like_sensorqa_root(
    path: Path,
) -> bool:
    """Determine whether a directory looks like the SensorQA backend root.
    
    Args:
        path: Path to the file or directory.
    
    Returns:
        Boolean result.
    """

    return (
        path.is_dir()
        and (
            path
            / "configs"
        ).is_dir()
        and (
            path
            / "tools"
        ).is_dir()
    )


# Tool registry construction


def build_tool_registry(
    paths: SensorQAPaths,
) -> tuple[
    ToolLoader,
    ToolRegistry,
    RegistryBuildReport,
]:
    """Discover, validate, and register SensorQA analysis tools.
    
    Built-in generic, built-in IMU, and custom plug-ins all pass through
    the same normal loading and validation process.
    
    Args:
        paths: Resolved SensorQA paths.
    
    Returns:
        Tuple containing the result values.
    """

    loader = ToolLoader(
        generic_tools_dir=(
            paths.generic_tools_dir
        ),

        imu_tools_dir=(
            paths.imu_tools_dir
        ),

        custom_tools_dir=(
            paths.custom_tools_dir
        ),
    )

    load_report = (
        loader.discover_tools()
    )

    registry = ToolRegistry(
        validator=ToolValidator(),
    )

    registry_report = (
        registry.rebuild(
            load_report
        )
    )

    return (
        loader,
        registry,
        registry_report,
    )


# DatasetBuilder construction


def build_dataset_builder(
) -> DatasetBuilder:
    """Construct the ingestion stack.
    
    This gives all entry points exactly the same:
    
        CSV loader
        column mapper
        unit manager
        dataset validator
    
    Returns:
        DatasetBuilder returned by the function.
    """

    return DatasetBuilder(
        csv_loader=CSVLoader(),

        column_mapper=ColumnMapper(),

        unit_manager=UnitManager(),

        dataset_validator=(
            DatasetValidator()
        ),
    )


# Internal composition root


def _assemble_application(
    paths: SensorQAPaths,
) -> tuple[
    SensorQAApplication,
    ToolLoader,
    ToolRegistry,
    RegistryBuildReport,
]:
    """Build the complete SensorQA backend without initializing runtime
    configuration.
    
    Args:
        paths: Resolved SensorQA paths.
    
    Returns:
        Tuple containing the result values.
    """

    path_problems = (
        paths.validate()
    )

    if path_problems:

        raise FileNotFoundError(
            "SensorQA startup paths are invalid:\n- "
            + "\n- ".join(
                path_problems
            )
        )

    (
        loader,
        registry,
        registry_report,
    ) = build_tool_registry(
        paths
    )

    dataset_builder = (
        build_dataset_builder()
    )

    pipeline = AnalysisPipeline(
        registry=registry,
    )

    application = (
        SensorQAApplication(
            registry=registry,
            dataset_builder=dataset_builder,
            pipeline=pipeline,
        )
    )

    return (
        application,
        loader,
        registry,
        registry_report,
    )


# Public application factory


def create_application(
    project_root: str | Path | None = None,
) -> SensorQAApplication:
    """Create an uninitialized SensorQAApplication.
    
    This is the main construction function for:
    
        - Python users
        - unit/integration tests
        - future CLI
        - future API
        - desktop app
    
    Example
    -------
    
    from sensorqa.bootstrap import create_application
    
    app = create_application()
    
    app.initialize(
        runtime_config_path="configs/sensorqa.toml",
        requirements_path="configs/requirements.toml",
    )
    
    Args:
        project_root: Value for `project_root`.
    
    Returns:
        SensorQAApplication returned by the function.
    """

    paths = (
        SensorQAPaths.from_root(
            project_root
        )
    )

    (
        application,
        _,
        _,
        _,
    ) = _assemble_application(
        paths
    )

    return application


# Complete initialized startup


def bootstrap_application(
    project_root: str | Path | None = None,
    *,
    runtime_config_path: str | Path | None = None,
    requirements_path: str | Path | None = None,
    auto_include_dependencies: bool = True,
    raise_on_error: bool = False,
) -> SensorQABootstrapResult:
    """Construct and initialize the complete SensorQA backend.
    
    This will be the preferred startup function for the future SensorQA
    application UI.
    
    Unlike create_application(), this function retains startup
    diagnostics so that the application can eventually show messages
    such as:
    
        "12 tools loaded"
        "1 custom plug-in rejected"
        "Configuration error"
        "Requirements profile loaded"
    
    rather than exposing raw exceptions to normal users.
    
    Args:
        project_root: Value for `project_root`.
        runtime_config_path: Path used for runtime config.
        requirements_path: Path used for requirements.
        auto_include_dependencies: Flag controlling auto include dependencies.
        raise_on_error: Flag controlling raise on error.
    
    Returns:
        SensorQABootstrapResult returned by the function.
    """

    paths = (
        SensorQAPaths.from_root(
            project_root
        )
    )

    (
        application,
        loader,
        registry,
        registry_report,
    ) = _assemble_application(
        paths
    )

    # Runtime configuration

    if runtime_config_path is None:

        resolved_runtime_config = (
            paths.runtime_config_path
        )

    else:

        resolved_runtime_config = (
            Path(
                runtime_config_path
            )
            .expanduser()
            .resolve()
        )

    # Qualification requirements

    if requirements_path is None:

        resolved_requirements = (
            paths.requirements_path
        )

    else:

        resolved_requirements = (
            Path(
                requirements_path
            )
            .expanduser()
            .resolve()
        )

    # Initialize application services

    initialization = (
        application.initialize(
            runtime_config_path=(
                resolved_runtime_config
            ),

            requirements_path=(
                resolved_requirements
            ),

            auto_include_dependencies=(
                auto_include_dependencies
            ),

            raise_on_error=(
                raise_on_error
            ),
        )
    )

    # Aggregate nonfatal startup warnings

    warnings: list[str] = []

    warnings.extend(
        registry_report.warnings
    )

    warnings.extend(
        initialization.warnings
    )

    return SensorQABootstrapResult(
        application=application,

        paths=paths,

        tool_loader=loader,

        registry=registry,

        registry_report=(
            registry_report
        ),

        initialization=(
            initialization
        ),

        warnings=warnings,
    )
