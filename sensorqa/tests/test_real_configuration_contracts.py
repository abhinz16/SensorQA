#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from sensorqa.core.config_loader import (
    ConfigLoader,
)
from sensorqa.core.requirements_loader import (
    RequirementsLoader,
)
from sensorqa.core.tool_contract import (
    BaseAnalysisTool,
)
from sensorqa.core.tool_registry import (
    ToolRegistry,
)


# Project discovery


def find_project_root() -> Path:
    """Locate the SensorQA project directory without assuming pytest
    was launched from a particular working directory.
    
    The project root is identified by:
    
        configs/sensorqa.toml
        tools/
    
    Returns:
        Resolved path.
    """

    test_file = Path(
        __file__
    ).resolve()

    candidates = [
        test_file.parent,
        *test_file.parents,
    ]

    for candidate in candidates:

        if (
            (
                candidate
                / "configs"
                / "sensorqa.toml"
            ).is_file()
            and (
                candidate
                / "tools"
            ).is_dir()
        ):

            return candidate

    raise RuntimeError(
        "Could not locate SensorQA project root. "
        "Expected to find configs/sensorqa.toml and tools/."
    )


PROJECT_ROOT = find_project_root()

TOOLS_ROOT = (
    PROJECT_ROOT
    / "tools"
)

SENSORQA_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "sensorqa.toml"
)

REQUIREMENTS_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "requirements.toml"
)


# Real built-in tool discovery


def load_module_from_path(
    path: Path,
):
    """Load one SensorQA tool module directly from its Python file.
    
    This test intentionally inspects the real built-in tool classes.
    ToolLoader itself will be integration-tested separately.
    
    Args:
        path: Path to the file or directory.
    
    Returns:
        Result returned by the function.
    """

    module_name = (
        "_sensorqa_contract_test_"
        + path.parent.name
        + "_"
        + path.stem
    )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            f"Could not create import specification for {path}."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def discover_real_builtin_tools(
) -> dict[str, dict]:
    """
    Discover concrete BaseAnalysisTool implementations from the
    actual tools/generic and tools/imu directories.

    Returns:

        {
            tool_id: {
                "tool": instance,
                "path": Path,
                "is_custom": False,
            }
        }
    """

    discovered = {}

    for category in (
        "generic",
        "imu",
    ):

        directory = (
            TOOLS_ROOT
            / category
        )

        if not directory.is_dir():
            continue

        for path in sorted(
            directory.glob(
                "*.py"
            )
        ):

            if path.name.startswith(
                "_"
            ):

                continue

            module = load_module_from_path(
                path
            )

            for _, candidate in inspect.getmembers(
                module,
                inspect.isclass,
            ):

                if candidate is BaseAnalysisTool:

                    continue

                if not issubclass(
                    candidate,
                    BaseAnalysisTool,
                ):

                    continue

                if inspect.isabstract(
                    candidate
                ):

                    continue

                # Ignore classes imported into a tool module from
                # another module.
                if (
                    candidate.__module__
                    != module.__name__
                ):

                    continue

                instance = candidate()

                tool_id = (
                    instance.metadata.tool_id
                )

                if tool_id in discovered:

                    previous = discovered[
                        tool_id
                    ][
                        "path"
                    ]

                    raise AssertionError(
                        f"Duplicate real built-in tool ID "
                        f"'{tool_id}' found in:\n"
                        f"  {previous}\n"
                        f"  {path}"
                    )

                discovered[
                    tool_id
                ] = {
                    "tool":
                        instance,

                    "path":
                        path,

                    "is_custom":
                        False,
                }

    return discovered


# Registry adapter


class RealToolRegistryAdapter(
    ToolRegistry
):
    """
    ToolRegistry-compatible adapter containing actual instantiated
    SensorQA tool classes.

    This lets the real ConfigLoader validate sensorqa.toml against
    actual ToolMetadata without coupling this test to the internal
    ToolRegistry construction API.
    """

    def __init__(
        self,
        discovered_tools: dict[
            str,
            dict,
        ],
    ) -> None:

        """Initialize the real tool registry adapter.
        
        Args:
            discovered_tools: Discovered tools used by this function.
        
        Returns:
            None.
        """
        self._real_tools = {}

        for tool_id, information in (
            discovered_tools.items()
        ):

            self._real_tools[
                tool_id
            ] = SimpleNamespace(
                tool_id=tool_id,
                tool=information[
                    "tool"
                ],
                source=str(
                    information[
                        "path"
                    ]
                ),
                is_custom=information[
                    "is_custom"
                ],
            )

    def tool_ids(
        self,
    ) -> list[str]:

        """Run tool ids.
        
        Returns:
            List of result values.
        """
        return list(
            self._real_tools.keys()
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
            in self._real_tools
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
        return self._real_tools.get(
            tool_id
        )


# Fixtures


@pytest.fixture(
    scope="module"
)
def real_tools():

    """Run real tools.
    
    Returns:
        Calculated value.
    """
    discovered = (
        discover_real_builtin_tools()
    )

    assert discovered, (
        "No built-in SensorQA analysis tools were discovered."
    )

    return discovered


@pytest.fixture(
    scope="module"
)
def real_registry(
    real_tools,
):

    """Run real registry.
    
    Args:
        real_tools: Fixture containing the real tool definitions.
    
    Returns:
        Calculated value.
    """
    return RealToolRegistryAdapter(
        real_tools
    )


# Real tool discovery sanity checks


def test_expected_core_tools_exist(
    real_tools,
):
    """These are the built-in tools currently expected in SensorQA V1.
    
    If a tool is intentionally renamed, update sensorqa.toml,
    requirements.toml, and this contract test together.
    
    Args:
        real_tools: Fixture containing the real tool definitions.
    
    Returns:
        None.
    """

    expected = {
        "generic_accuracy",
        "imu_sampling_integrity",
        "imu_gyro_bias",
        "imu_accelerometer_bias",
        "imu_stationary_noise",
        "imu_allan_deviation",
        "imu_psd",
        "imu_saturation",
        "imu_temperature_stability",
        "imu_axis_correlation",
        "imu_gravity_error",
        "imu_six_position_calibration",
    }

    discovered = set(
        real_tools
    )

    missing = (
        expected
        - discovered
    )

    assert not missing, (
        "Expected built-in tool(s) were not discovered: "
        + ", ".join(
            sorted(
                missing
            )
        )
    )


def test_real_builtin_tool_ids_are_unique(
    real_tools,
):
    """Discovery itself already detects duplicates, but this makes the
    contract explicit in the test report.
    
    Args:
        real_tools: Fixture containing the real tool definitions.
    
    Returns:
        None.
    """

    ids = list(
        real_tools.keys()
    )

    assert len(
        ids
    ) == len(
        set(
            ids
        )
    )


# Real sensorqa.toml validation


def test_real_sensorqa_configuration_loads_against_real_tools(
    real_registry,
):
    """This is the critical contract test:
    
        actual sensorqa.toml
              +
        actual ToolMetadata
              ↓
        ConfigLoader
    
    Any mismatch in tool IDs, parameter names, types, limits, or
    choices should fail here.
    
    Args:
        real_registry: Fixture containing the real tool registry.
    
    Returns:
        None.
    """

    report = ConfigLoader(
        registry=real_registry
    ).load(
        SENSORQA_CONFIG_PATH
    )

    if not report.success:

        errors = "\n".join(
            (
                f"{error.location}: "
                f"{error.message}"
            )
            for error
            in report.errors
        )

        pytest.fail(
            "Real sensorqa.toml failed validation "
            "against real SensorQA tools:\n"
            + errors
        )

    assert report.config is not None


def test_all_explicitly_configured_tools_exist(
    real_tools,
):
    """Every [tools.<tool_id>] entry in sensorqa.toml must correspond
    to an actual built-in tool.
    
    Args:
        real_tools: Fixture containing the real tool definitions.
    
    Returns:
        None.
    """

    with SENSORQA_CONFIG_PATH.open(
        "rb"
    ) as file:

        raw = tomllib.load(
            file
        )

    configured_tools = set(
        raw.get(
            "tools",
            {}
        ).keys()
    )

    discovered_tools = set(
        real_tools.keys()
    )

    missing = (
        configured_tools
        - discovered_tools
    )

    assert not missing, (
        "sensorqa.toml references tool ID(s) that do not exist: "
        + ", ".join(
            sorted(
                missing
            )
        )
    )


def test_preferred_tool_order_references_real_tools(
    real_tools,
):
    """Every ID in [tool_order].preferred must exist.
    
    Args:
        real_tools: Fixture containing the real tool definitions.
    
    Returns:
        None.
    """

    with SENSORQA_CONFIG_PATH.open(
        "rb"
    ) as file:

        raw = tomllib.load(
            file
        )

    preferred = (
        raw.get(
            "tool_order",
            {}
        ).get(
            "preferred",
            [],
        )
    )

    discovered = set(
        real_tools
    )

    unknown = [
        tool_id
        for tool_id
        in preferred
        if tool_id
        not in discovered
    ]

    assert unknown == [], (
        "Preferred order contains unknown real tool ID(s): "
        + ", ".join(
            unknown
        )
    )


# Real ToolMetadata parameter contract


def test_configured_parameter_names_match_real_tool_metadata(
    real_tools,
):
    """Prevent configuration drift such as:
    
        sensorqa.toml:
            segment_size = 2048
    
        tool:
            segment_length
    
    ConfigLoader also catches this, but this test reports the
    mismatch directly as a tool-contract failure.
    
    Args:
        real_tools: Fixture containing the real tool definitions.
    
    Returns:
        None.
    """

    with SENSORQA_CONFIG_PATH.open(
        "rb"
    ) as file:

        raw = tomllib.load(
            file
        )

    configured_tools = raw.get(
        "tools",
        {}
    )

    problems = []

    for tool_id, configuration in (
        configured_tools.items()
    ):

        if tool_id not in real_tools:
            continue

        tool = real_tools[
            tool_id
        ][
            "tool"
        ]

        valid_parameters = {
            parameter.name
            for parameter
            in tool.metadata.parameters
        }

        configured_parameters = {
            key
            for key
            in configuration.keys()
            if key != "enabled"
        }

        unknown_parameters = (
            configured_parameters
            - valid_parameters
        )

        if unknown_parameters:

            problems.append(
                (
                    tool_id,
                    sorted(
                        unknown_parameters
                    ),
                )
            )

    assert problems == [], (
        "sensorqa.toml contains parameter name(s) that do not "
        f"match real ToolMetadata: {problems}"
    )


def test_real_tool_parameter_names_are_unique_within_each_tool(
    real_tools,
):
    """A tool must not declare the same parameter name more than once.
    
    Args:
        real_tools: Fixture containing the real tool definitions.
    
    Returns:
        None.
    """

    duplicates = []

    for tool_id, information in (
        real_tools.items()
    ):

        parameters = [
            parameter.name
            for parameter
            in information[
                "tool"
            ].metadata.parameters
        ]

        if len(
            parameters
        ) != len(
            set(
                parameters
            )
        ):

            duplicates.append(
                tool_id
            )

    assert duplicates == [], (
        "Tool(s) contain duplicate parameter definitions: "
        + ", ".join(
            duplicates
        )
    )


# Requirements profile contract


def test_real_requirements_toml_loads():
    """Validate the actual qualification profile rather than only
    temporary unit-test profiles.
    
    Returns:
        None.
    """

    report = RequirementsLoader().load(
        REQUIREMENTS_CONFIG_PATH
    )

    if not report.success:

        errors = "\n".join(
            (
                f"{error.location}: "
                f"{error.message}"
            )
            for error
            in report.errors
        )

        pytest.fail(
            "Real requirements.toml failed validation:\n"
            + errors
        )

    assert report.requirement_set is not None


def test_requirement_tool_ids_reference_real_tools(
    real_tools,
):
    """A requirement must not refer to a tool that does not exist.
    
    This test checks both enabled and disabled requirements because
    disabled examples should remain valid configuration examples.
    
    Args:
        real_tools: Fixture containing the real tool definitions.
    
    Returns:
        None.
    """

    report = RequirementsLoader().load(
        REQUIREMENTS_CONFIG_PATH
    )

    assert report.success
    assert report.requirement_set is not None

    available = set(
        real_tools.keys()
    )

    unknown = {
        requirement.tool_id
        for requirement
        in report.requirement_set.requirements
        if requirement.tool_id
        not in available
    }

    assert unknown == set(), (
        "requirements.toml references tool ID(s) that do not "
        "exist: "
        + ", ".join(
            sorted(
                unknown
            )
        )
    )


def test_requirement_ids_are_unique_in_real_profile():
    """Check that requirement ids are unique in real profile.
    
    Returns:
        None.
    """
    report = RequirementsLoader().load(
        REQUIREMENTS_CONFIG_PATH
    )

    assert report.success
    assert report.requirement_set is not None

    ids = [
        requirement.requirement_id
        for requirement
        in report.requirement_set.requirements
    ]

    assert len(
        ids
    ) == len(
        set(
            ids
        )
    )


# Runtime configuration result


def test_real_configuration_produces_enabled_tools(
    real_registry,
):
    """The actual configuration should result in a non-empty analysis
    selection.
    
    Args:
        real_registry: Fixture containing the real tool registry.
    
    Returns:
        None.
    """

    report = ConfigLoader(
        registry=real_registry
    ).load(
        SENSORQA_CONFIG_PATH
    )

    assert report.success
    assert report.config is not None

    assert len(
        report.config.enabled_tool_ids
    ) > 0


def test_real_enabled_tools_have_parameter_dictionaries(
    real_registry,
):
    """Every enabled tool should have a pipeline-ready parameter
    dictionary, even if that dictionary is empty.
    
    Args:
        real_registry: Fixture containing the real tool registry.
    
    Returns:
        None.
    """

    report = ConfigLoader(
        registry=real_registry
    ).load(
        SENSORQA_CONFIG_PATH
    )

    assert report.success
    assert report.config is not None

    for tool_id in (
        report.config.enabled_tool_ids
    ):

        assert (
            tool_id
            in report.config.tool_parameters
        )

        assert isinstance(
            report.config.tool_parameters[
                tool_id
            ],
            dict,
        )


# Configuration/reproducibility expectations


def test_implicit_resampling_is_disabled_in_real_config(
):
    """SensorQA's stated preprocessing principle should be reflected
    in the shipped runtime configuration.
    
    Returns:
        None.
    """

    with SENSORQA_CONFIG_PATH.open(
        "rb"
    ) as file:

        raw = tomllib.load(
            file
        )

    assert (
        raw[
            "advanced"
        ][
            "allow_implicit_resampling"
        ]
        is False
    )


def test_custom_tools_are_not_auto_enabled_by_default(
):
    """Newly discovered Python plug-ins must not silently execute.
    
    Returns:
        None.
    """

    with SENSORQA_CONFIG_PATH.open(
        "rb"
    ) as file:

        raw = tomllib.load(
            file
        )

    custom = raw[
        "custom_tools"
    ]

    assert (
        custom[
            "default_enabled"
        ]
        is False
    )

    assert (
        custom[
            "enable_unconfigured_custom_tools"
        ]
        is False
    )
