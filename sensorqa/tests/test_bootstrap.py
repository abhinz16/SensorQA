#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.bootstrap import (
    SensorQABootstrapResult,
    SensorQAPaths,
    bootstrap_application,
    build_dataset_builder,
    build_tool_registry,
    create_application,
)
from sensorqa.core.application import SensorQAApplication
from sensorqa.core.pipeline import AnalysisPipeline
from sensorqa.core.tool_registry import ToolRegistry
from sensorqa.ingestion.dataset_builder import DatasetBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent


# Path resolution


def test_paths_resolve_from_sensorqa_root():

    """Check that paths resolve from sensorqa root.
    
    Returns:
        None.
    """
    paths = SensorQAPaths.from_root(
        PROJECT_ROOT
    )

    assert (
        paths.project_root
        == PROJECT_ROOT
    )

    assert (
        paths.configs_dir
        == PROJECT_ROOT / "configs"
    )

    assert (
        paths.tools_dir
        == PROJECT_ROOT / "tools"
    )

    assert (
        paths.generic_tools_dir
        == PROJECT_ROOT
        / "tools"
        / "generic"
    )

    assert (
        paths.imu_tools_dir
        == PROJECT_ROOT
        / "tools"
        / "imu"
    )

    assert (
        paths.custom_tools_dir
        == PROJECT_ROOT
        / "tools"
        / "custom"
    )

    assert (
        paths.runtime_config_path
        == PROJECT_ROOT
        / "configs"
        / "sensorqa.toml"
    )

    assert (
        paths.requirements_path
        == PROJECT_ROOT
        / "configs"
        / "requirements.toml"
    )


def test_paths_resolve_from_parent_directory():

    """Check that paths resolve from parent directory.
    
    Returns:
        None.
    """
    paths = SensorQAPaths.from_root(
        PROJECT_PARENT
    )

    assert (
        paths.project_root
        == PROJECT_ROOT
    )


def test_paths_validate_cleanly():

    """Check that paths validate cleanly.
    
    Returns:
        None.
    """
    paths = SensorQAPaths.from_root(
        PROJECT_ROOT
    )

    assert (
        paths.validate()
        == []
    )


def test_invalid_project_root_is_rejected(
    tmp_path,
):

    """Check that invalid project root is rejected.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
    
    Returns:
        None.
    """
    with pytest.raises(
        FileNotFoundError
    ):

        SensorQAPaths.from_root(
            tmp_path
        )


# Production component construction


def test_build_tool_registry_uses_real_registry():

    """Check that build tool registry uses real registry.
    
    Returns:
        None.
    """
    paths = SensorQAPaths.from_root(
        PROJECT_ROOT
    )

    (
        loader,
        registry,
        report,
    ) = build_tool_registry(
        paths
    )

    assert loader is not None

    assert isinstance(
        registry,
        ToolRegistry,
    )

    assert (
        report.registered_count
        > 0
    )

    assert (
        report.rejected_count
        == 0
    )

    assert (
        report.load_error_count
        == 0
    )


def test_build_tool_registry_contains_core_tools():

    """Check that build tool registry contains core tools.
    
    Returns:
        None.
    """
    paths = SensorQAPaths.from_root(
        PROJECT_ROOT
    )

    (
        _,
        registry,
        _,
    ) = build_tool_registry(
        paths
    )

    expected = {
        "generic_accuracy",
        "imu_sampling_integrity",
        "imu_gyro_bias",
        "imu_accelerometer_bias",
        "imu_stationary_noise",
        "imu_allan_deviation",
        "imu_psd",
        "imu_axis_correlation",
        "imu_gravity_error",
        "imu_saturation",
        "imu_six_position_calibration",
        "imu_temperature_stability",
    }

    assert expected.issubset(
        set(
            registry.tool_ids()
        )
    )


def test_build_dataset_builder_returns_production_builder():

    """Check that build dataset builder returns production builder.
    
    Returns:
        None.
    """
    builder = (
        build_dataset_builder()
    )

    assert isinstance(
        builder,
        DatasetBuilder,
    )

    assert (
        builder.csv_loader
        is not None
    )

    assert (
        builder.column_mapper
        is not None
    )

    assert (
        builder.unit_manager
        is not None
    )

    assert (
        builder.dataset_validator
        is not None
    )


# Application factory


def test_create_application_returns_uninitialized_application():

    """Check that create application returns uninitialized application.
    
    Returns:
        None.
    """
    application = (
        create_application(
            PROJECT_ROOT
        )
    )

    assert isinstance(
        application,
        SensorQAApplication,
    )

    assert (
        application.initialized
        is False
    )

    assert isinstance(
        application.registry,
        ToolRegistry,
    )

    assert isinstance(
        application.pipeline,
        AnalysisPipeline,
    )

    assert isinstance(
        application.dataset_builder,
        DatasetBuilder,
    )


def test_create_application_shares_registry_with_pipeline():

    """Check that create application shares registry with pipeline.
    
    Returns:
        None.
    """
    application = (
        create_application(
            PROJECT_ROOT
        )
    )

    assert (
        application.pipeline.registry
        is application.registry
    )


# Complete bootstrap


def test_bootstrap_application_initializes_successfully():

    """Check that bootstrap application initializes successfully.
    
    Returns:
        None.
    """
    result = (
        bootstrap_application(
            PROJECT_ROOT
        )
    )

    assert isinstance(
        result,
        SensorQABootstrapResult,
    )

    assert (
        result.initialization
        is not None
    )

    assert (
        result.initialization.success
        is True
    )

    assert (
        result.application.initialized
        is True
    )

    assert (
        result.initialized
        is True
    )


def test_bootstrap_result_exposes_ui_startup_diagnostics():

    """Check that bootstrap result exposes ui startup diagnostics.
    
    Returns:
        None.
    """
    result = (
        bootstrap_application(
            PROJECT_ROOT
        )
    )

    assert (
        result.paths.project_root
        == PROJECT_ROOT
    )

    assert isinstance(
        result.registry,
        ToolRegistry,
    )

    assert (
        result.registered_tool_ids
        == result.registry.tool_ids()
    )

    assert (
        result.rejected_tool_count
        == 0
    )

    assert (
        result.load_error_count
        == 0
    )

    assert isinstance(
        result.warnings,
        list,
    )


def test_bootstrap_loads_runtime_configuration_and_requirements():

    """Check that bootstrap loads runtime configuration and requirements.
    
    Returns:
        None.
    """
    result = (
        bootstrap_application(
            PROJECT_ROOT
        )
    )

    assert (
        result.application.runtime_config
        is not None
    )

    assert (
        result.application.requirement_set
        is not None
    )

    assert (
        result.initialization
        is not None
    )

    assert (
        result.initialization.runtime_config
        is result.application.runtime_config
    )

    assert (
        result.initialization.requirements
        is result.application.requirement_set
    )


def test_bootstrap_uses_expected_configuration_files():

    """Check that bootstrap uses expected configuration files.
    
    Returns:
        None.
    """
    result = (
        bootstrap_application(
            PROJECT_ROOT
        )
    )

    assert (
        result.paths.runtime_config_path
        .is_file()
    )

    assert (
        result.paths.requirements_path
        .is_file()
    )

    assert (
        result.paths.runtime_config_path.name
        == "sensorqa.toml"
    )

    assert (
        result.paths.requirements_path.name
        == "requirements.toml"
    )
