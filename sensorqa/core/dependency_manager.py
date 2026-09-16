#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field

from sensorqa.core.tool_registry import (
    ToolRegistry,
)


@dataclass
class DependencyIssue:
    """
    Describes one dependency problem.
    """

    tool_id: str

    dependency_id: str | None

    issue_type: str

    message: str


@dataclass
class DependencyResolution:
    """
    Result of validating or resolving a selected tool set.
    """

    valid: bool

    requested_tool_ids: list[str]

    resolved_tool_ids: list[str] = field(
        default_factory=list
    )

    execution_order: list[str] = field(
        default_factory=list
    )

    automatically_added_tool_ids: list[str] = field(
        default_factory=list
    )

    missing_dependencies: dict[
        str,
        list[str],
    ] = field(
        default_factory=dict
    )

    unavailable_tools: list[str] = field(
        default_factory=list
    )

    dependency_cycles: list[
        list[str]
    ] = field(
        default_factory=list
    )

    issues: list[
        DependencyIssue
    ] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )


class DependencyManager:
    """
    Resolves dependencies between registered SensorQA tools.

    Tool dependencies are declared in ToolMetadata:

        dependencies=[
            "some_other_tool"
        ]

    The manager supports:

        - direct dependency lookup
        - recursive dependency lookup
        - reverse dependent lookup
        - selected-tool validation
        - automatic dependency inclusion
        - deterministic topological ordering
        - dependency-cycle detection

    This component does not execute tools.
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:

        """Initialize the dependency manager.
        
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

    # Public dependency queries

    def direct_dependencies(
        self,
        tool_id: str,
    ) -> list[str]:
        """Return dependencies declared directly by a tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            List of result values.
        """

        registered = self.registry.get_registered(
            tool_id
        )

        if registered is None:

            return []

        return list(
            registered.tool.metadata.dependencies
        )

    def all_dependencies(
        self,
        tool_id: str,
    ) -> list[str]:
        """Return all recursive dependencies for one tool.
        
        Dependencies are returned in dependency-first order.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            List of result values.
        """

        if not self.registry.contains(
            tool_id
        ):

            return []

        output: list[str] = []

        visited: set[str] = set()

        active: set[str] = set()

        def visit(
            current_tool_id: str,
        ) -> None:

            """Run visit.
            
            Args:
                current_tool_id: Identifier for current tool.
            
            Returns:
                None.
            """
            if current_tool_id in active:

                # Cycle handling belongs to cycle detection.
                return

            if current_tool_id in visited:
                return

            active.add(
                current_tool_id
            )

            for dependency_id in (
                self.direct_dependencies(
                    current_tool_id
                )
            ):

                if not self.registry.contains(
                    dependency_id
                ):

                    continue

                visit(
                    dependency_id
                )

                if dependency_id not in output:

                    output.append(
                        dependency_id
                    )

            active.remove(
                current_tool_id
            )

            visited.add(
                current_tool_id
            )

        visit(
            tool_id
        )

        return output

    def direct_dependents(
        self,
        tool_id: str,
    ) -> list[str]:
        """Return tools that directly depend on tool_id.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            List of result values.
        """

        dependents = []

        for candidate_id in (
            self.registry.tool_ids()
        ):

            if (
                tool_id
                in self.direct_dependencies(
                    candidate_id
                )
            ):

                dependents.append(
                    candidate_id
                )

        return dependents

    def all_dependents(
        self,
        tool_id: str,
    ) -> list[str]:
        """Return all recursively dependent tools.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            List of result values.
        """

        output: list[str] = []

        visited: set[str] = set()

        def visit(
            current_tool_id: str,
        ) -> None:

            """Run visit.
            
            Args:
                current_tool_id: Identifier for current tool.
            
            Returns:
                None.
            """
            for dependent_id in (
                self.direct_dependents(
                    current_tool_id
                )
            ):

                if dependent_id in visited:
                    continue

                visited.add(
                    dependent_id
                )

                output.append(
                    dependent_id
                )

                visit(
                    dependent_id
                )

        visit(
            tool_id
        )

        return output

    # Selection validation

    def validate_selection(
        self,
        tool_ids: list[str],
    ) -> DependencyResolution:
        """Validate a selected tool set without automatically adding
        dependencies.
        
        A selection is invalid if:
        
            - a selected tool is unavailable
            - one of its dependencies is not selected
            - a dependency cycle exists
        
        Args:
            tool_ids: Tool identifiers to process.
        
        Returns:
            DependencyResolution returned by the function.
        """

        requested = self._clean_tool_ids(
            tool_ids
        )

        resolution = DependencyResolution(
            valid=True,
            requested_tool_ids=requested,
            resolved_tool_ids=list(
                requested
            ),
        )

        # Availability

        for tool_id in requested:

            if not self.registry.contains(
                tool_id
            ):

                resolution.unavailable_tools.append(
                    tool_id
                )

                resolution.issues.append(
                    DependencyIssue(
                        tool_id=tool_id,
                        dependency_id=None,
                        issue_type="unavailable_tool",
                        message=(
                            f"Tool '{tool_id}' is not registered."
                        ),
                    )
                )

        available_requested = [
            tool_id
            for tool_id in requested
            if self.registry.contains(
                tool_id
            )
        ]

        selected_set = set(
            available_requested
        )

        # Missing selected dependencies

        for tool_id in available_requested:

            missing = []

            for dependency_id in (
                self.direct_dependencies(
                    tool_id
                )
            ):

                if not self.registry.contains(
                    dependency_id
                ):

                    missing.append(
                        dependency_id
                    )

                    resolution.issues.append(
                        DependencyIssue(
                            tool_id=tool_id,
                            dependency_id=dependency_id,
                            issue_type=(
                                "unavailable_dependency"
                            ),
                            message=(
                                f"Tool '{tool_id}' depends on "
                                f"'{dependency_id}', but that "
                                "dependency is not registered."
                            ),
                        )
                    )

                    continue

                if dependency_id not in selected_set:

                    missing.append(
                        dependency_id
                    )

                    resolution.issues.append(
                        DependencyIssue(
                            tool_id=tool_id,
                            dependency_id=dependency_id,
                            issue_type=(
                                "dependency_not_selected"
                            ),
                            message=(
                                f"Tool '{tool_id}' requires "
                                f"'{dependency_id}', but the "
                                "dependency is not selected."
                            ),
                        )
                    )

            if missing:

                resolution.missing_dependencies[
                    tool_id
                ] = missing

        # Cycles

        cycles = self.find_cycles(
            available_requested
        )

        resolution.dependency_cycles = cycles

        for cycle in cycles:

            resolution.issues.append(
                DependencyIssue(
                    tool_id=cycle[
                        0
                    ],
                    dependency_id=None,
                    issue_type=(
                        "dependency_cycle"
                    ),
                    message=(
                        "Dependency cycle detected: "
                        + " -> ".join(
                            cycle
                        )
                    ),
                )
            )

        # Execution order

        if (
            not resolution.unavailable_tools
            and not resolution.missing_dependencies
            and not resolution.dependency_cycles
        ):

            resolution.execution_order = (
                self.execution_order(
                    available_requested
                )
            )

        resolution.valid = (
            len(
                resolution.issues
            )
            == 0
        )

        return resolution

    # Automatic dependency resolution

    def resolve_selection(
        self,
        tool_ids: list[str],
        auto_include_dependencies: bool = True,
    ) -> DependencyResolution:
        """Resolve a tool selection.
        
        When auto_include_dependencies is True, recursively required
        dependencies are added automatically.
        
        Example:
        
            requested:
                ["tool_c"]
        
            dependencies:
                tool_c -> tool_b
                tool_b -> tool_a
        
            resolved:
                ["tool_a", "tool_b", "tool_c"]
        
        Args:
            tool_ids: Tool identifiers to process.
            auto_include_dependencies: Flag controlling auto include dependencies.
        
        Returns:
            DependencyResolution returned by the function.
        """

        requested = self._clean_tool_ids(
            tool_ids
        )

        if not auto_include_dependencies:

            return self.validate_selection(
                requested
            )

        resolution = DependencyResolution(
            valid=True,
            requested_tool_ids=requested,
        )

        requested_set = set(
            requested
        )

        # Requested availability

        for tool_id in requested:

            if not self.registry.contains(
                tool_id
            ):

                resolution.unavailable_tools.append(
                    tool_id
                )

                resolution.issues.append(
                    DependencyIssue(
                        tool_id=tool_id,
                        dependency_id=None,
                        issue_type="unavailable_tool",
                        message=(
                            f"Tool '{tool_id}' is not registered."
                        ),
                    )
                )

        if resolution.unavailable_tools:

            resolution.valid = False

            return resolution

        # Recursively expand dependency closure

        resolved_set: set[str] = set(
            requested
        )

        expansion_order: list[str] = list(
            requested
        )

        index = 0

        while index < len(
            expansion_order
        ):

            tool_id = expansion_order[
                index
            ]

            index += 1

            for dependency_id in (
                self.direct_dependencies(
                    tool_id
                )
            ):

                if not self.registry.contains(
                    dependency_id
                ):

                    resolution.missing_dependencies.setdefault(
                        tool_id,
                        [],
                    ).append(
                        dependency_id
                    )

                    resolution.issues.append(
                        DependencyIssue(
                            tool_id=tool_id,
                            dependency_id=dependency_id,
                            issue_type=(
                                "unavailable_dependency"
                            ),
                            message=(
                                f"Tool '{tool_id}' depends on "
                                f"'{dependency_id}', but that "
                                "dependency is not registered."
                            ),
                        )
                    )

                    continue

                if dependency_id not in resolved_set:

                    resolved_set.add(
                        dependency_id
                    )

                    expansion_order.append(
                        dependency_id
                    )

        if resolution.missing_dependencies:

            resolution.valid = False

            return resolution

        # Determine which tools were added automatically

        automatically_added = [
            tool_id
            for tool_id in expansion_order
            if tool_id not in requested_set
        ]

        resolution.automatically_added_tool_ids = (
            automatically_added
        )

        # Preserve user-requested tools as the initial preference,
        # then append added dependencies before topological sorting.

        resolved_preference = list(
            requested
        )

        for tool_id in automatically_added:

            if tool_id not in resolved_preference:

                resolved_preference.append(
                    tool_id
                )

        resolution.resolved_tool_ids = (
            resolved_preference
        )

        # Cycle detection

        cycles = self.find_cycles(
            resolution.resolved_tool_ids
        )

        resolution.dependency_cycles = cycles

        if cycles:

            for cycle in cycles:

                resolution.issues.append(
                    DependencyIssue(
                        tool_id=cycle[
                            0
                        ],
                        dependency_id=None,
                        issue_type=(
                            "dependency_cycle"
                        ),
                        message=(
                            "Dependency cycle detected: "
                            + " -> ".join(
                                cycle
                            )
                        ),
                    )
                )

            resolution.valid = False

            return resolution

        # Dependency-correct execution order

        resolution.execution_order = (
            self.execution_order(
                resolution.resolved_tool_ids
            )
        )

        if automatically_added:

            resolution.warnings.append(
                "SensorQA automatically added required "
                "dependencies: "
                + ", ".join(
                    automatically_added
                )
            )

        resolution.valid = True

        return resolution

    # Topological ordering

    def execution_order(
        self,
        tool_ids: list[str],
    ) -> list[str]:
        """Return deterministic dependency-first execution order.
        
        The input order is treated as a preference among otherwise
        independent tools.
        
        Raises ValueError if the selected graph contains a cycle or
        references unavailable dependencies.
        
        Args:
            tool_ids: Tool identifiers to process.
        
        Returns:
            List of result values.
        """

        requested = self._clean_tool_ids(
            tool_ids
        )

        selected_set = set(
            requested
        )

        for tool_id in requested:

            if not self.registry.contains(
                tool_id
            ):

                raise ValueError(
                    f"Tool '{tool_id}' is not registered."
                )

            for dependency_id in (
                self.direct_dependencies(
                    tool_id
                )
            ):

                if dependency_id not in selected_set:

                    raise ValueError(
                        f"Tool '{tool_id}' depends on "
                        f"'{dependency_id}', but the dependency "
                        "is not included in the selected tool set."
                    )

        cycles = self.find_cycles(
            requested
        )

        if cycles:

            raise ValueError(
                "Dependency cycle detected: "
                + " -> ".join(
                    cycles[
                        0
                    ]
                )
            )

        # Kahn topological sort
        #
        # Use requested ordering as the stable preference.

        indegree = {
            tool_id: 0
            for tool_id in requested
        }

        dependents = {
            tool_id: []
            for tool_id in requested
        }

        for tool_id in requested:

            for dependency_id in (
                self.direct_dependencies(
                    tool_id
                )
            ):

                indegree[
                    tool_id
                ] += 1

                dependents[
                    dependency_id
                ].append(
                    tool_id
                )

        order_index = {
            tool_id: index
            for index, tool_id
            in enumerate(
                requested
            )
        }

        ready = [
            tool_id
            for tool_id
            in requested
            if indegree[
                tool_id
            ] == 0
        ]

        ready.sort(
            key=lambda tool_id:
                order_index[
                    tool_id
                ]
        )

        output = []

        while ready:

            tool_id = ready.pop(
                0
            )

            output.append(
                tool_id
            )

            children = sorted(
                dependents[
                    tool_id
                ],
                key=lambda child:
                    order_index[
                        child
                    ],
            )

            for dependent_id in children:

                indegree[
                    dependent_id
                ] -= 1

                if (
                    indegree[
                        dependent_id
                    ]
                    == 0
                ):

                    ready.append(
                        dependent_id
                    )

                    ready.sort(
                        key=lambda candidate:
                            order_index[
                                candidate
                            ]
                    )

        if len(
            output
        ) != len(
            requested
        ):

            raise ValueError(
                "Selected tools could not be topologically ordered."
            )

        return output

    # Cycle detection

    def find_cycles(
        self,
        tool_ids: list[str] | None = None,
    ) -> list[
        list[str]
    ]:
        """Find dependency cycles.
        
        If tool_ids is None, all registered tools are checked.
        
        Returned cycles include the repeated starting node:
        
            [
                "tool_a",
                "tool_b",
                "tool_c",
                "tool_a",
            ]
        
        Args:
            tool_ids: Tool identifiers to process.
        
        Returns:
            List of result values.
        """

        if tool_ids is None:

            selected = self.registry.tool_ids()

        else:

            selected = [
                tool_id
                for tool_id
                in self._clean_tool_ids(
                    tool_ids
                )
                if self.registry.contains(
                    tool_id
                )
            ]

        selected_set = set(
            selected
        )

        state: dict[
            str,
            int,
        ] = {
            tool_id: 0
            for tool_id in selected
        }

        stack: list[str] = []

        cycles: list[
            list[str]
        ] = []

        cycle_signatures: set[
            tuple[str, ...]
        ] = set()

        def visit(
            tool_id: str,
        ) -> None:

            """Run visit.
            
            Args:
                tool_id: Registered tool identifier.
            
            Returns:
                None.
            """
            state[
                tool_id
            ] = 1

            stack.append(
                tool_id
            )

            for dependency_id in (
                self.direct_dependencies(
                    tool_id
                )
            ):

                if dependency_id not in selected_set:
                    continue

                if state[
                    dependency_id
                ] == 0:

                    visit(
                        dependency_id
                    )

                elif state[
                    dependency_id
                ] == 1:

                    try:

                        start_index = stack.index(
                            dependency_id
                        )

                    except ValueError:

                        continue

                    cycle = (
                        stack[
                            start_index:
                        ]
                        + [
                            dependency_id
                        ]
                    )

                    signature = (
                        self._canonical_cycle_signature(
                            cycle
                        )
                    )

                    if (
                        signature
                        not in cycle_signatures
                    ):

                        cycle_signatures.add(
                            signature
                        )

                        cycles.append(
                            cycle
                        )

            stack.pop()

            state[
                tool_id
            ] = 2

        for tool_id in selected:

            if state[
                tool_id
            ] == 0:

                visit(
                    tool_id
                )

        return cycles

    # Tool Manager helpers

    def can_disable(
        self,
        tool_id: str,
        enabled_tool_ids: list[str],
    ) -> tuple[
        bool,
        list[str],
    ]:
        """Determine whether a tool can be disabled without breaking
        currently enabled dependent tools.
        
        Returns:
        
            (
                can_disable,
                blocking_dependents
            )
        
        Args:
            tool_id: Registered tool identifier.
            enabled_tool_ids: Value for `enabled_tool_ids`.
        """

        enabled_set = set(
            self._clean_tool_ids(
                enabled_tool_ids
            )
        )

        blocking = [
            dependent_id
            for dependent_id
            in self.all_dependents(
                tool_id
            )
            if dependent_id in enabled_set
        ]

        return (
            len(
                blocking
            )
            == 0,
            blocking,
        )

    def required_enablements(
        self,
        tool_id: str,
        currently_enabled: list[str],
    ) -> list[str]:
        """Return additional dependencies that must be enabled before
        tool_id can execute.
        
        Args:
            tool_id: Registered tool identifier.
            currently_enabled: Value for `currently_enabled`.
        
        Returns:
            List of result values.
        """

        if not self.registry.contains(
            tool_id
        ):

            return []

        enabled_set = set(
            self._clean_tool_ids(
                currently_enabled
            )
        )

        return [
            dependency_id
            for dependency_id
            in self.all_dependencies(
                tool_id
            )
            if dependency_id
            not in enabled_set
        ]

    def dependency_summary(
        self,
        tool_id: str,
    ) -> dict:
        """UI-friendly summary for one tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Dictionary containing the result values.
        """

        if not self.registry.contains(
            tool_id
        ):

            return {
                "tool_id":
                    tool_id,

                "available":
                    False,

                "direct_dependencies":
                    [],

                "all_dependencies":
                    [],

                "direct_dependents":
                    [],

                "all_dependents":
                    [],
            }

        return {
            "tool_id":
                tool_id,

            "available":
                True,

            "direct_dependencies":
                self.direct_dependencies(
                    tool_id
                ),

            "all_dependencies":
                self.all_dependencies(
                    tool_id
                ),

            "direct_dependents":
                self.direct_dependents(
                    tool_id
                ),

            "all_dependents":
                self.all_dependents(
                    tool_id
                ),
        }

    # Helpers

    @staticmethod
    def _clean_tool_ids(
        tool_ids: list[str],
    ) -> list[str]:
        """Normalize tool ID list while preserving order.
        
        Args:
            tool_ids: Tool identifiers to process.
        
        Returns:
            List of result values.
        """

        if not isinstance(
            tool_ids,
            list,
        ):

            raise TypeError(
                "tool_ids must be a list of strings."
            )

        output = []

        seen = set()

        for tool_id in tool_ids:

            if not isinstance(
                tool_id,
                str,
            ):

                raise TypeError(
                    "Every tool ID must be a string."
                )

            cleaned = tool_id.strip()

            if not cleaned:

                continue

            if cleaned in seen:
                continue

            output.append(
                cleaned
            )

            seen.add(
                cleaned
            )

        return output

    @staticmethod
    def _canonical_cycle_signature(
        cycle: list[str],
    ) -> tuple[str, ...]:
        """Produce a rotation-independent signature for a directed
        cycle to avoid reporting the same cycle multiple times.
        
        Example:
        
            A -> B -> C -> A
        
        and:
        
            B -> C -> A -> B
        
        represent the same cycle.
        
        Args:
            cycle: Value for `cycle`.
        
        Returns:
            Tuple containing the result values.
        """

        if len(
            cycle
        ) <= 1:

            return tuple(
                cycle
            )

        nodes = cycle[
            :-1
        ]

        if not nodes:

            return tuple(
                cycle
            )

        rotations = []

        for index in range(
            len(nodes)
        ):

            rotation = (
                nodes[
                    index:
                ]
                + nodes[
                    :index
                ]
            )

            rotations.append(
                tuple(
                    rotation
                )
            )

        canonical = min(
            rotations
        )

        return (
            canonical
            + (
                canonical[
                    0
                ],
            )
        )
