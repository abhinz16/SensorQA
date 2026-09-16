#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace

import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.core.config_loader import (
    ConfigLoader,
)
from sensorqa.core.tool_contract import (
    ParameterType,
)
from sensorqa.core.tool_registry import (
    ToolRegistry,
)


# Lightweight test objects


def parameter(
    name: str,
    parameter_type: ParameterType,
    *,
    default=None,
    minimum=None,
    maximum=None,
    choices=None,
    required: bool = False,
):
    """Minimal ToolParameter-compatible object.
    
    These tests intentionally isolate ConfigLoader from the
    ToolParameter constructor so loader behavior can be tested
    independently from tool-contract construction.
    
    Args:
        name: Name of the item.
        parameter_type: Value for `parameter_type`.
        default: Value for `default`.
        minimum: Value for `minimum`.
        maximum: Value for `maximum`.
        choices: Value for `choices`.
        required: Value for `required`.
    
    Returns:
        Result returned by the function.
    """

    return SimpleNamespace(
        name=name,
        parameter_type=parameter_type,
        default=default,
        minimum=minimum,
        maximum=maximum,
        choices=choices,
        required=required,
    )


def registered_tool(
    tool_id: str,
    *,
    parameters=None,
    is_custom: bool = False,
):
    """Build the minimum RegisteredTool-compatible structure required
    by ConfigLoader.
    
    Args:
        tool_id: Registered tool identifier.
        parameters: Tool parameter values.
        is_custom: Value for `is_custom`.
    
    Returns:
        Result returned by the function.
    """

    metadata = SimpleNamespace(
        tool_id=tool_id,
        parameters=list(
            parameters
            or []
        ),
    )

    tool = SimpleNamespace(
        metadata=metadata
    )

    return SimpleNamespace(
        tool_id=tool_id,
        tool=tool,
        source=(
            f"tools/custom/{tool_id}.py"
            if is_custom
            else f"tools/imu/{tool_id}.py"
        ),
        is_custom=is_custom,
    )


class StubRegistry(ToolRegistry):
    """
    Minimal ToolRegistry implementation for ConfigLoader tests.

    We deliberately do not invoke ToolRegistry.__init__ because
    these unit tests only require its public lookup behavior.
    """

    def __init__(
        self,
        tools,
    ):

        """Initialize the stub registry.
        
        Args:
            tools: Tools used by this function.
        
        Returns:
            None.
        """
        self._test_tools = {
            tool.tool_id:
                tool
            for tool
            in tools
        }

    def tool_ids(
        self,
    ) -> list[str]:

        """Run tool ids.
        
        Returns:
            List of result values.
        """
        return list(
            self._test_tools.keys()
        )

    def contains(
        self,
        tool_id: str,
    ) -> bool:

        """Run contains.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Boolean result.
        """
        return (
            tool_id
            in self._test_tools
        )

    def get_registered(
        self,
        tool_id: str,
    ):

        """Return registered.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Calculated value.
        """
        return self._test_tools.get(
            tool_id
        )


# Fixtures


@pytest.fixture
def registry():
    """Registry containing two built-in tools and one custom tool.
    
    Returns:
        Result returned by the function.
    """

    builtin_a = registered_tool(
        "builtin_a",
        parameters=[
            parameter(
                "minimum_samples",
                ParameterType.INTEGER,
                default=100,
                minimum=1,
                maximum=10000,
            ),
            parameter(
                "threshold",
                ParameterType.FLOAT,
                default=0.5,
                minimum=0.0,
                maximum=1.0,
            ),
            parameter(
                "window",
                ParameterType.CHOICE,
                default="hann",
                choices=[
                    "hann",
                    "hamming",
                ],
            ),
            parameter(
                "allow_irregular",
                ParameterType.BOOLEAN,
                default=False,
            ),
            parameter(
                "label",
                ParameterType.STRING,
                default="default",
            ),
        ],
    )

    builtin_b = registered_tool(
        "builtin_b",
        parameters=[
            parameter(
                "gain",
                ParameterType.FLOAT,
                default=1.0,
                minimum=0.0,
                maximum=10.0,
            ),
        ],
    )

    custom_a = registered_tool(
        "custom_a",
        parameters=[
            parameter(
                "custom_threshold",
                ParameterType.FLOAT,
                default=2.0,
                minimum=0.0,
                maximum=100.0,
            ),
        ],
        is_custom=True,
    )

    return StubRegistry(
        [
            builtin_a,
            builtin_b,
            custom_a,
        ]
    )


def write_config(
    tmp_path,
    text: str,
):
    """Write one temporary sensorqa.toml file.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        text: Text to process.
    
    Returns:
        Result returned by the function.
    """

    path = (
        tmp_path
        / "sensorqa.toml"
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    return path


# Basic successful loading


def test_valid_configuration_loads(
    tmp_path,
    registry,
):
    """A valid configured tool should load with explicit and default
    parameters combined.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[application]
name = "SensorQA"
default_sensor_type = "imu"

[tools.builtin_a]
enabled = true
minimum_samples = 500
threshold = 0.75
window = "hann"
allow_irregular = false
label = "test"
""",
    )

    loader = ConfigLoader(
        registry=registry
    )

    report = loader.load(
        path
    )

    assert report.success is True
    assert report.errors == []
    assert report.config is not None

    config = report.config

    assert config.is_tool_enabled(
        "builtin_a"
    )

    assert (
        config.parameters_for(
            "builtin_a"
        )[
            "minimum_samples"
        ]
        == 500
    )

    assert (
        config.parameters_for(
            "builtin_a"
        )[
            "threshold"
        ]
        == pytest.approx(
            0.75
        )
    )

    assert (
        config.parameters_for(
            "builtin_a"
        )[
            "window"
        ]
        == "hann"
    )


def test_unspecified_parameter_uses_tool_default(
    tmp_path,
    registry,
):
    """Missing optional tool parameters should be filled from declared
    tool defaults.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
minimum_samples = 250
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    parameters = (
        report.config.parameters_for(
            "builtin_a"
        )
    )

    assert parameters[
        "minimum_samples"
    ] == 250

    assert parameters[
        "threshold"
    ] == pytest.approx(
        0.5
    )

    assert parameters[
        "window"
    ] == "hann"

    assert (
        parameters[
            "allow_irregular"
        ]
        is False
    )


# Built-in enablement behavior


def test_unconfigured_builtin_tool_is_disabled(
    tmp_path,
    registry,
):
    """Newly discovered built-in tools should not silently become
    enabled merely because they exist in the registry.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert report.config.is_tool_enabled(
        "builtin_a"
    )

    assert not report.config.is_tool_enabled(
        "builtin_b"
    )


def test_explicitly_disabled_tool_is_disabled(
    tmp_path,
    registry,
):
    """Check that explicitly disabled tool is disabled.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = false
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert not report.config.is_tool_enabled(
        "builtin_a"
    )


# Unknown tools / parameters


def test_unknown_tool_is_error(
    tmp_path,
    registry,
):
    """A misspelled or unavailable tool must not be silently ignored.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[tools.does_not_exist]
enabled = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False

    assert any(
        "unknown tool"
        in error.message.lower()
        for error
        in report.errors
    )


def test_unknown_parameter_is_error(
    tmp_path,
    registry,
):
    """Parameter typos must fail loudly.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
minimun_samples = 500
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False

    assert any(
        "unknown parameter"
        in error.message.lower()
        for error
        in report.errors
    )

    assert any(
        "minimun_samples"
        in error.location
        for error
        in report.errors
    )


# Integer validation


def test_integer_parameter_accepts_integer(
    tmp_path,
    registry,
):
    """Check that integer parameter accepts integer.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
minimum_samples = 750
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert (
        report.config.parameters_for(
            "builtin_a"
        )[
            "minimum_samples"
        ]
        == 750
    )


def test_integer_parameter_rejects_float(
    tmp_path,
    registry,
):
    """Check that integer parameter rejects float.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
minimum_samples = 750.5
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False

    assert any(
        "integer"
        in error.message.lower()
        for error
        in report.errors
    )


def test_integer_parameter_rejects_boolean(
    tmp_path,
    registry,
):
    """Python bool is a subclass of int, but it must not be accepted
    as a numerical tool parameter.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
minimum_samples = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


def test_integer_below_minimum_is_error(
    tmp_path,
    registry,
):
    """Check that integer below minimum is error.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
minimum_samples = 0
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False

    assert any(
        "below minimum"
        in error.message.lower()
        for error
        in report.errors
    )


def test_integer_above_maximum_is_error(
    tmp_path,
    registry,
):
    """Check that integer above maximum is error.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
minimum_samples = 10001
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False

    assert any(
        "exceeds maximum"
        in error.message.lower()
        for error
        in report.errors
    )


# Float validation


def test_float_parameter_accepts_integer_toml_value(
    tmp_path,
    registry,
):
    """A FLOAT parameter may legitimately be represented by an integer
    in TOML.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
threshold = 1
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    value = (
        report.config.parameters_for(
            "builtin_a"
        )[
            "threshold"
        ]
    )

    assert isinstance(
        value,
        float,
    )

    assert value == pytest.approx(
        1.0
    )


def test_float_parameter_rejects_string(
    tmp_path,
    registry,
):
    """Check that float parameter rejects string.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
threshold = "0.5"
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


def test_float_parameter_rejects_boolean(
    tmp_path,
    registry,
):
    """Check that float parameter rejects boolean.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
threshold = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


def test_float_below_minimum_is_error(
    tmp_path,
    registry,
):
    """Check that float below minimum is error.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
threshold = -0.01
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


def test_float_above_maximum_is_error(
    tmp_path,
    registry,
):
    """Check that float above maximum is error.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
threshold = 1.01
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


# Boolean validation


def test_boolean_parameter_accepts_boolean(
    tmp_path,
    registry,
):
    """Check that boolean parameter accepts boolean.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
allow_irregular = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert (
        report.config.parameters_for(
            "builtin_a"
        )[
            "allow_irregular"
        ]
        is True
    )


def test_boolean_parameter_rejects_integer(
    tmp_path,
    registry,
):
    """Check that boolean parameter rejects integer.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
allow_irregular = 1
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


# Choice validation


def test_valid_choice_is_accepted(
    tmp_path,
    registry,
):
    """Check that valid choice is accepted.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
window = "hamming"
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert (
        report.config.parameters_for(
            "builtin_a"
        )[
            "window"
        ]
        == "hamming"
    )


def test_invalid_choice_is_error(
    tmp_path,
    registry,
):
    """Check that invalid choice is error.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
window = "banana"
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False

    assert any(
        "invalid choice"
        in error.message.lower()
        for error
        in report.errors
    )


# String validation


def test_string_parameter_accepts_string(
    tmp_path,
    registry,
):
    """Check that string parameter accepts string.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
label = "prototype-A"
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert (
        report.config.parameters_for(
            "builtin_a"
        )[
            "label"
        ]
        == "prototype-A"
    )


def test_string_parameter_rejects_number(
    tmp_path,
    registry,
):
    """Check that string parameter rejects number.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true
label = 123
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


# Custom tool behavior


def test_unconfigured_custom_tool_is_disabled_by_default(
    tmp_path,
    registry,
):
    """Check that unconfigured custom tool is disabled by default.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[custom_tools]
default_enabled = false
enable_unconfigured_custom_tools = false

[tools.builtin_a]
enabled = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert not report.config.is_tool_enabled(
        "custom_a"
    )


def test_custom_tool_requires_both_auto_enable_flags(
    tmp_path,
    registry,
):
    """One opt-in flag alone should not automatically execute an
    unconfigured custom Python plug-in.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[custom_tools]
default_enabled = true
enable_unconfigured_custom_tools = false
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert not report.config.is_tool_enabled(
        "custom_a"
    )


def test_custom_tool_can_be_auto_enabled_with_both_flags(
    tmp_path,
    registry,
):
    """Check that custom tool can be auto enabled with both flags.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[custom_tools]
default_enabled = true
enable_unconfigured_custom_tools = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert report.config.is_tool_enabled(
        "custom_a"
    )

    assert (
        report.config.parameters_for(
            "custom_a"
        )[
            "custom_threshold"
        ]
        == pytest.approx(
            2.0
        )
    )


def test_explicit_custom_tool_configuration_works(
    tmp_path,
    registry,
):
    """Check that explicit custom tool configuration works.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[custom_tools]
default_enabled = false
enable_unconfigured_custom_tools = false

[tools.custom_a]
enabled = true
custom_threshold = 4.5
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert report.config.is_tool_enabled(
        "custom_a"
    )

    assert (
        report.config.parameters_for(
            "custom_a"
        )[
            "custom_threshold"
        ]
        == pytest.approx(
            4.5
        )
    )


# Preferred tool order


def test_preferred_order_controls_independent_enabled_order(
    tmp_path,
    registry,
):
    """Check that preferred order controls independent enabled order.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tool_order]
preferred = [
    "builtin_b",
    "builtin_a",
]

[tools.builtin_a]
enabled = true

[tools.builtin_b]
enabled = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert (
        report.config.enabled_tool_ids
        == [
            "builtin_b",
            "builtin_a",
        ]
    )


def test_disabled_tool_in_preferred_order_is_not_executed(
    tmp_path,
    registry,
):
    """Check that disabled tool in preferred order is not executed.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tool_order]
preferred = [
    "builtin_b",
    "builtin_a",
]

[tools.builtin_a]
enabled = true

[tools.builtin_b]
enabled = false
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert (
        report.config.enabled_tool_ids
        == [
            "builtin_a"
        ]
    )


def test_unknown_tool_in_preferred_order_is_error(
    tmp_path,
    registry,
):
    """Check that unknown tool in preferred order is error.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tool_order]
preferred = [
    "builtin_a",
    "unknown_tool",
]

[tools.builtin_a]
enabled = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False

    assert any(
        "preferred order"
        in error.message.lower()
        and "unknown_tool"
        in error.message
        for error
        in report.errors
    )


def test_duplicate_tool_in_preferred_order_generates_warning(
    tmp_path,
    registry,
):
    """Check that duplicate tool in preferred order generates warning.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tool_order]
preferred = [
    "builtin_a",
    "builtin_a",
]

[tools.builtin_a]
enabled = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert any(
        "more than once"
        in warning.lower()
        for warning
        in report.warnings
    )

    assert (
        report.config.enabled_tool_ids
        == [
            "builtin_a"
        ]
    )


# General runtime sections


def test_invalid_pipeline_boolean_is_error(
    tmp_path,
    registry,
):
    """Check that invalid pipeline boolean is error.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[pipeline]
allow_invalid_dataset = "no"
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


def test_invalid_sensor_type_is_error(
    tmp_path,
    registry,
):
    """Check that invalid sensor type is error.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[application]
default_sensor_type = "thermocouple"
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


def test_logging_level_is_normalized_to_uppercase(
    tmp_path,
    registry,
):
    """Check that logging level is normalized to uppercase.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[logging]
level = "debug"
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert (
        report.config.logging[
            "level"
        ]
        == "DEBUG"
    )


def test_invalid_logging_level_is_error(
    tmp_path,
    registry,
):
    """Check that invalid logging level is error.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[logging]
level = "VERY_LOUD"
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False


# Unknown top-level sections


def test_unknown_top_level_section_generates_warning(
    tmp_path,
    registry,
):
    """Unknown top-level sections are warnings rather than fatal
    errors to preserve reasonable forward compatibility.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[future_feature]
enabled = true

[tools.builtin_a]
enabled = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert any(
        "future_feature"
        in warning
        for warning
        in report.warnings
    )


# File errors


def test_missing_configuration_file(
    tmp_path,
    registry,
):
    """Check that missing configuration file.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = (
        tmp_path
        / "does_not_exist.toml"
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False

    assert any(
        "does not exist"
        in error.message.lower()
        for error
        in report.errors
    )


def test_invalid_toml_is_reported(
    tmp_path,
    registry,
):
    """Check that invalid toml is reported.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a
enabled = true
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success is False

    assert any(
        "parse"
        in error.message.lower()
        for error
        in report.errors
    )


# Enabled tool parameters


def test_tool_parameters_only_include_enabled_tools(
    tmp_path,
    registry,
):
    """Pipeline-ready tool_parameters should contain enabled tools
    only.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """

    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = true

[tools.builtin_b]
enabled = false
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert (
        "builtin_a"
        in report.config.tool_parameters
    )

    assert (
        "builtin_b"
        not in report.config.tool_parameters
    )


def test_no_enabled_tools_generates_warning(
    tmp_path,
    registry,
):
    """Check that no enabled tools generates warning.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
        registry: Tool registry used for lookups.
    
    Returns:
        None.
    """
    path = write_config(
        tmp_path,
        """
[tools.builtin_a]
enabled = false

[tools.builtin_b]
enabled = false
""",
    )

    report = ConfigLoader(
        registry
    ).load(
        path
    )

    assert report.success

    assert (
        report.config.enabled_tool_ids
        == []
    )

    assert any(
        "no analysis tools"
        in warning.lower()
        for warning
        in report.warnings
    )
