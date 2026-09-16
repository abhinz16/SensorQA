#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace

import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.core.dependency_manager import (
    DependencyManager,
)
from sensorqa.core.tool_registry import (
    ToolRegistry,
)


# Lightweight test registry


def registered_tool(
    tool_id: str,
    dependencies: list[str] | None = None,
):
    """Build the minimum registered-tool structure required by
    DependencyManager.
    
    Args:
        tool_id: Registered tool identifier.
        dependencies: Value for `dependencies`.
    
    Returns:
        Result returned by the function.
    """

    metadata = SimpleNamespace(
        tool_id=tool_id,
        dependencies=list(
            dependencies
            or []
        ),
    )

    tool = SimpleNamespace(
        metadata=metadata
    )

    return SimpleNamespace(
        tool_id=tool_id,
        tool=tool,
        source=f"tools/test/{tool_id}.py",
        is_custom=False,
    )


class StubRegistry(ToolRegistry):
    """
    Small ToolRegistry-compatible test double.
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
    """Dependency graph:
    
        A
    
        B -> A
    
        C -> B -> A
    
        D -> A
    
        E
    
    Returns:
        Result returned by the function.
    """

    return StubRegistry(
        [
            registered_tool(
                "A"
            ),
            registered_tool(
                "B",
                [
                    "A"
                ],
            ),
            registered_tool(
                "C",
                [
                    "B"
                ],
            ),
            registered_tool(
                "D",
                [
                    "A"
                ],
            ),
            registered_tool(
                "E"
            ),
        ]
    )


@pytest.fixture
def manager(
    registry,
):

    """Run manager.
    
    Args:
        registry: Tool registry used for lookups.
    
    Returns:
        Calculated value.
    """
    return DependencyManager(
        registry=registry
    )


@pytest.fixture
def cyclic_registry():
    """Dependency graph:
    
        X -> Y -> Z -> X
    
    Returns:
        Result returned by the function.
    """

    return StubRegistry(
        [
            registered_tool(
                "X",
                [
                    "Y"
                ],
            ),
            registered_tool(
                "Y",
                [
                    "Z"
                ],
            ),
            registered_tool(
                "Z",
                [
                    "X"
                ],
            ),
        ]
    )


# Direct dependencies


def test_direct_dependencies(
    manager,
):

    """Check that direct dependencies.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    assert (
        manager.direct_dependencies(
            "B"
        )
        == [
            "A"
        ]
    )


def test_tool_without_dependencies_returns_empty(
    manager,
):

    """Check that tool without dependencies returns empty.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    assert (
        manager.direct_dependencies(
            "A"
        )
        == []
    )


def test_unknown_tool_direct_dependencies_empty(
    manager,
):

    """Check that unknown tool direct dependencies empty.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    assert (
        manager.direct_dependencies(
            "unknown"
        )
        == []
    )


# Recursive dependencies


def test_all_dependencies_recursive(
    manager,
):
    """C depends on B, which depends on A.
    
    Dependency-first order should therefore be:
    
        A
        B
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    assert (
        manager.all_dependencies(
            "C"
        )
        == [
            "A",
            "B",
        ]
    )


def test_all_dependencies_single_level(
    manager,
):

    """Check that all dependencies single level.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    assert (
        manager.all_dependencies(
            "D"
        )
        == [
            "A"
        ]
    )


def test_all_dependencies_unknown_tool_empty(
    manager,
):

    """Check that all dependencies unknown tool empty.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    assert (
        manager.all_dependencies(
            "unknown"
        )
        == []
    )


# Reverse dependencies


def test_direct_dependents(
    manager,
):
    """B and D directly depend on A.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    assert (
        manager.direct_dependents(
            "A"
        )
        == [
            "B",
            "D",
        ]
    )


def test_all_dependents_recursive(
    manager,
):
    """A is required by:
    
        B
        C through B
        D
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    assert (
        manager.all_dependents(
            "A"
        )
        == [
            "B",
            "C",
            "D",
        ]
    )


def test_all_dependents_of_leaf_empty(
    manager,
):

    """Check that all dependents of leaf empty.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    assert (
        manager.all_dependents(
            "C"
        )
        == []
    )


# Strict selection validation


def test_valid_complete_selection(
    manager,
):

    """Check that valid complete selection.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    resolution = (
        manager.validate_selection(
            [
                "A",
                "B",
                "C",
            ]
        )
    )

    assert resolution.valid is True

    assert (
        resolution.missing_dependencies
        == {}
    )

    assert (
        resolution.unavailable_tools
        == []
    )

    assert (
        resolution.execution_order
        == [
            "A",
            "B",
            "C",
        ]
    )


def test_missing_selected_dependency_is_invalid(
    manager,
):
    """Selecting C alone is invalid in strict mode because B is not
    selected.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    resolution = (
        manager.validate_selection(
            [
                "C"
            ]
        )
    )

    assert resolution.valid is False

    assert (
        resolution.missing_dependencies[
            "C"
        ]
        == [
            "B"
        ]
    )

    assert any(
        issue.issue_type
        == "dependency_not_selected"
        for issue
        in resolution.issues
    )


def test_recursive_dependency_must_also_be_selected(
    manager,
):
    """Selecting C and B is still incomplete because B requires A.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    resolution = (
        manager.validate_selection(
            [
                "C",
                "B",
            ]
        )
    )

    assert resolution.valid is False

    assert (
        resolution.missing_dependencies[
            "B"
        ]
        == [
            "A"
        ]
    )


def test_unknown_requested_tool_is_invalid(
    manager,
):

    """Check that unknown requested tool is invalid.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    resolution = (
        manager.validate_selection(
            [
                "unknown"
            ]
        )
    )

    assert resolution.valid is False

    assert (
        resolution.unavailable_tools
        == [
            "unknown"
        ]
    )

    assert any(
        issue.issue_type
        == "unavailable_tool"
        for issue
        in resolution.issues
    )


# Automatic dependency resolution


def test_auto_include_recursive_dependencies(
    manager,
):
    """Requesting C should automatically include B and A.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    resolution = (
        manager.resolve_selection(
            [
                "C"
            ],
            auto_include_dependencies=True,
        )
    )

    assert resolution.valid is True

    assert (
        resolution.automatically_added_tool_ids
        == [
            "B",
            "A",
        ]
    )

    assert (
        resolution.execution_order
        == [
            "A",
            "B",
            "C",
        ]
    )


def test_auto_include_only_missing_dependencies(
    manager,
):
    """If A is already requested, only B needs automatic inclusion.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    resolution = (
        manager.resolve_selection(
            [
                "A",
                "C",
            ],
            auto_include_dependencies=True,
        )
    )

    assert resolution.valid is True

    assert (
        resolution.automatically_added_tool_ids
        == [
            "B"
        ]
    )

    assert (
        resolution.execution_order
        == [
            "A",
            "B",
            "C",
        ]
    )


def test_auto_resolution_warning_mentions_added_dependencies(
    manager,
):

    """Check that auto resolution warning mentions added dependencies.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    resolution = (
        manager.resolve_selection(
            [
                "C"
            ]
        )
    )

    assert resolution.valid

    assert any(
        "automatically added"
        in warning.lower()
        for warning
        in resolution.warnings
    )


def test_auto_include_false_behaves_strictly(
    manager,
):

    """Check that auto include false behaves strictly.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    resolution = (
        manager.resolve_selection(
            [
                "C"
            ],
            auto_include_dependencies=False,
        )
    )

    assert resolution.valid is False

    assert (
        "C"
        in resolution.missing_dependencies
    )


def test_auto_resolution_unknown_tool_invalid(
    manager,
):

    """Check that auto resolution unknown tool invalid.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    resolution = (
        manager.resolve_selection(
            [
                "does_not_exist"
            ]
        )
    )

    assert resolution.valid is False

    assert (
        resolution.unavailable_tools
        == [
            "does_not_exist"
        ]
    )


# Unavailable declared dependency


def test_unavailable_declared_dependency(
):
    """A registered tool may itself contain an invalid dependency
    declaration.
    
    Returns:
        None.
    """

    registry = StubRegistry(
        [
            registered_tool(
                "broken_tool",
                [
                    "missing_tool"
                ],
            )
        ]
    )

    manager = DependencyManager(
        registry
    )

    resolution = (
        manager.resolve_selection(
            [
                "broken_tool"
            ]
        )
    )

    assert resolution.valid is False

    assert (
        resolution.missing_dependencies[
            "broken_tool"
        ]
        == [
            "missing_tool"
        ]
    )

    assert any(
        issue.issue_type
        == "unavailable_dependency"
        for issue
        in resolution.issues
    )


# Execution ordering


def test_execution_order_corrects_reverse_request(
    manager,
):
    """User preference cannot violate dependencies.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    order = (
        manager.execution_order(
            [
                "C",
                "B",
                "A",
            ]
        )
    )

    assert (
        order
        == [
            "A",
            "B",
            "C",
        ]
    )


def test_independent_tool_preference_is_preserved(
    manager,
):
    """E is independent.
    
    Requested order:
        E, D, A
    
    D requires A, so A must precede D, but E may remain first.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    order = (
        manager.execution_order(
            [
                "E",
                "D",
                "A",
            ]
        )
    )

    assert (
        order
        == [
            "E",
            "A",
            "D",
        ]
    )


def test_execution_order_requires_dependency_in_selection(
    manager,
):

    """Check that execution order requires dependency in selection.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        manager.execution_order(
            [
                "B"
            ]
        )


def test_execution_order_rejects_unknown_tool(
    manager,
):

    """Check that execution order rejects unknown tool.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        manager.execution_order(
            [
                "unknown"
            ]
        )


# Duplicate selection cleanup


def test_duplicate_requested_tools_are_removed(
    manager,
):

    """Check that duplicate requested tools are removed.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    resolution = (
        manager.resolve_selection(
            [
                "A",
                "A",
                "B",
                "B",
            ]
        )
    )

    assert resolution.valid

    assert (
        resolution.requested_tool_ids
        == [
            "A",
            "B",
        ]
    )

    assert (
        resolution.execution_order
        == [
            "A",
            "B",
        ]
    )


def test_blank_tool_ids_are_removed(
    manager,
):

    """Check that blank tool ids are removed.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    resolution = (
        manager.resolve_selection(
            [
                "",
                "   ",
                "A",
            ]
        )
    )

    assert resolution.valid

    assert (
        resolution.requested_tool_ids
        == [
            "A"
        ]
    )


# Cycle detection


def test_dependency_cycle_detected(
    cyclic_registry,
):

    """Check that dependency cycle detected.
    
    Args:
        cyclic_registry: Cyclic registry used by this function.
    
    Returns:
        None.
    """
    manager = DependencyManager(
        cyclic_registry
    )

    cycles = manager.find_cycles()

    assert len(
        cycles
    ) >= 1

    cycle = cycles[
        0
    ]

    assert cycle[
        0
    ] == cycle[
        -1
    ]

    assert set(
        cycle[
            :-1
        ]
    ) == {
        "X",
        "Y",
        "Z",
    }


def test_cycle_makes_resolution_invalid(
    cyclic_registry,
):

    """Check that cycle makes resolution invalid.
    
    Args:
        cyclic_registry: Cyclic registry used by this function.
    
    Returns:
        None.
    """
    manager = DependencyManager(
        cyclic_registry
    )

    resolution = (
        manager.resolve_selection(
            [
                "X"
            ]
        )
    )

    assert resolution.valid is False

    assert (
        len(
            resolution.dependency_cycles
        )
        >= 1
    )

    assert any(
        issue.issue_type
        == "dependency_cycle"
        for issue
        in resolution.issues
    )


def test_execution_order_rejects_cycle(
    cyclic_registry,
):

    """Check that execution order rejects cycle.
    
    Args:
        cyclic_registry: Cyclic registry used by this function.
    
    Returns:
        None.
    """
    manager = DependencyManager(
        cyclic_registry
    )

    with pytest.raises(
        ValueError,
        match="cycle",
    ):

        manager.execution_order(
            [
                "X",
                "Y",
                "Z",
            ]
        )


# Disable behavior


def test_cannot_disable_required_tool(
    manager,
):
    """A cannot be disabled while B and C remain enabled.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    can_disable, blockers = (
        manager.can_disable(
            tool_id="A",
            enabled_tool_ids=[
                "A",
                "B",
                "C",
            ],
        )
    )

    assert can_disable is False

    assert set(
        blockers
    ) == {
        "B",
        "C",
    }


def test_can_disable_when_dependents_not_enabled(
    manager,
):
    """B has dependent C, but C is not currently enabled.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """

    can_disable, blockers = (
        manager.can_disable(
            tool_id="B",
            enabled_tool_ids=[
                "A",
                "B",
                "D",
            ],
        )
    )

    assert can_disable is True

    assert blockers == []


# Required enablements


def test_required_enablements_all_missing(
    manager,
):

    """Check that required enablements all missing.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    required = (
        manager.required_enablements(
            tool_id="C",
            currently_enabled=[],
        )
    )

    assert (
        required
        == [
            "A",
            "B",
        ]
    )


def test_required_enablements_excludes_already_enabled(
    manager,
):

    """Check that required enablements excludes already enabled.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    required = (
        manager.required_enablements(
            tool_id="C",
            currently_enabled=[
                "A"
            ],
        )
    )

    assert (
        required
        == [
            "B"
        ]
    )


def test_required_enablements_unknown_tool_empty(
    manager,
):

    """Check that required enablements unknown tool empty.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    assert (
        manager.required_enablements(
            tool_id="unknown",
            currently_enabled=[],
        )
        == []
    )


# Dependency summary


def test_dependency_summary(
    manager,
):

    """Check that dependency summary.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    summary = (
        manager.dependency_summary(
            "B"
        )
    )

    assert summary[
        "available"
    ] is True

    assert summary[
        "direct_dependencies"
    ] == [
        "A"
    ]

    assert summary[
        "all_dependencies"
    ] == [
        "A"
    ]

    assert summary[
        "direct_dependents"
    ] == [
        "C"
    ]

    assert summary[
        "all_dependents"
    ] == [
        "C"
    ]


def test_dependency_summary_unknown_tool(
    manager,
):

    """Check that dependency summary unknown tool.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    summary = (
        manager.dependency_summary(
            "unknown"
        )
    )

    assert summary[
        "available"
    ] is False

    assert summary[
        "direct_dependencies"
    ] == []

    assert summary[
        "all_dependencies"
    ] == []

    assert summary[
        "direct_dependents"
    ] == []

    assert summary[
        "all_dependents"
    ] == []


# Input validation


def test_tool_ids_must_be_list(
    manager,
):

    """Check that tool ids must be list.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    with pytest.raises(
        TypeError
    ):

        manager.resolve_selection(
            "A"
        )


def test_every_tool_id_must_be_string(
    manager,
):

    """Check that every tool id must be string.
    
    Args:
        manager: Dependency manager fixture.
    
    Returns:
        None.
    """
    with pytest.raises(
        TypeError
    ):

        manager.resolve_selection(
            [
                "A",
                123,
            ]
        )
