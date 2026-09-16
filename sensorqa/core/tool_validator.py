#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sensorqa.core.tool_contract import (
    BaseAnalysisTool,
    ParameterType,
    SensorType,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
)


@dataclass
class ToolDefinitionValidation:
    """
    Result of validating the definition of one SensorQA tool.

    This validation checks the tool itself, not whether a
    particular dataset can support the analysis.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ToolValidator:
    """
    Performs structural validation of SensorQA analysis tools.

    The validator checks metadata, parameters, dependencies,
    IDs, compatible sensor types, and other tool-definition rules.
    """

    TOOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    def validate(
        self,
        tool: BaseAnalysisTool,
    ) -> ToolDefinitionValidation:
        """Validate one SensorQA analysis tool.
        
        Returns a ToolDefinitionValidation object containing
        errors and warnings.
        
        Args:
            tool: Analysis tool instance.
        
        Returns:
            Validation result.
        """

        errors: list[str] = []
        warnings: list[str] = []

        # Metadata must be accessible

        try:
            metadata = tool.metadata

        except Exception as exc:
            return ToolDefinitionValidation(
                valid=False,
                errors=[
                    "Unable to read tool metadata: "
                    f"{type(exc).__name__}: {exc}"
                ],
            )

        # Metadata must be the expected type

        if not isinstance(metadata, ToolMetadata):
            return ToolDefinitionValidation(
                valid=False,
                errors=[
                    "Tool metadata must return a ToolMetadata object."
                ],
            )

        # Validate core metadata fields

        self._validate_tool_id(
            metadata.tool_id,
            errors,
        )

        self._validate_name(
            metadata.name,
            errors,
        )

        self._validate_version(
            metadata.version,
            errors,
            warnings,
        )

        self._validate_description(
            metadata.description,
            warnings,
        )

        self._validate_category(
            metadata.category,
            errors,
        )

        self._validate_sensor_types(
            metadata.compatible_sensor_types,
            errors,
        )

        # Validate columns

        self._validate_columns(
            required_columns=metadata.required_columns,
            optional_columns=metadata.optional_columns,
            errors=errors,
            warnings=warnings,
        )

        # Validate tool parameters

        self._validate_parameters(
            metadata.parameters,
            errors,
            warnings,
        )

        # Validate dependencies

        self._validate_dependencies(
            current_tool_id=metadata.tool_id,
            dependencies=metadata.dependencies,
            errors=errors,
            warnings=warnings,
        )

        return ToolDefinitionValidation(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_tool_id(
        self,
        tool_id: str,
        errors: list[str],
    ) -> None:
        """Validate the unique identifier used for the tool.
        
        Args:
            tool_id: Registered tool identifier.
            errors: Value for `errors`.
        
        Returns:
            None.
        """

        if not isinstance(tool_id, str):
            errors.append(
                "tool_id must be a string."
            )
            return

        if not tool_id.strip():
            errors.append(
                "tool_id cannot be empty."
            )
            return

        if not self.TOOL_ID_PATTERN.fullmatch(tool_id):
            errors.append(
                "tool_id must begin with a lowercase letter and "
                "contain only lowercase letters, numbers, and underscores. "
                f"Received: '{tool_id}'."
            )

    @staticmethod
    def _validate_name(
        name: str,
        errors: list[str],
    ) -> None:
        """Validate the human-readable tool name.
        
        Args:
            name: Name of the item.
            errors: Value for `errors`.
        
        Returns:
            None.
        """

        if not isinstance(name, str):
            errors.append(
                "Tool name must be a string."
            )
            return

        if not name.strip():
            errors.append(
                "Tool name cannot be empty."
            )

    @staticmethod
    def _validate_version(
        version: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate the tool version.
        
        SensorQA does not require strict semantic versioning,
        but semantic-style versions are strongly recommended.
        
        Args:
            version: Value for `version`.
            errors: Value for `errors`.
            warnings: Value for `warnings`.
        
        Returns:
            None.
        """

        if not isinstance(version, str):
            errors.append(
                "Tool version must be a string."
            )
            return

        if not version.strip():
            errors.append(
                "Tool version cannot be empty."
            )
            return

        semantic_version_pattern = re.compile(
            r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?$"
        )

        if not semantic_version_pattern.fullmatch(version):
            warnings.append(
                f"Tool version '{version}' does not follow the "
                "recommended semantic version format such as '1.0.0'."
            )

    @staticmethod
    def _validate_description(
        description: str,
        warnings: list[str],
    ) -> None:
        """Validate the description shown to users.
        
        Args:
            description: Value for `description`.
            warnings: Value for `warnings`.
        
        Returns:
            None.
        """

        if not isinstance(description, str):
            warnings.append(
                "Tool description should be a string."
            )
            return

        if not description.strip():
            warnings.append(
                "Tool description is empty."
            )

    @staticmethod
    def _validate_category(
        category: ToolCategory,
        errors: list[str],
    ) -> None:
        """Verify that the tool uses a supported SensorQA category.
        
        Args:
            category: Value for `category`.
            errors: Value for `errors`.
        
        Returns:
            None.
        """

        if not isinstance(category, ToolCategory):
            errors.append(
                "Tool category must be a ToolCategory value."
            )

    @staticmethod
    def _validate_sensor_types(
        sensor_types: list[SensorType],
        errors: list[str],
    ) -> None:
        """Verify compatible sensor types.
        
        Args:
            sensor_types: Value for `sensor_types`.
            errors: Value for `errors`.
        
        Returns:
            None.
        """

        if not isinstance(sensor_types, list):
            errors.append(
                "compatible_sensor_types must be a list."
            )
            return

        if not sensor_types:
            errors.append(
                "At least one compatible sensor type must be specified."
            )
            return

        for sensor_type in sensor_types:
            if not isinstance(sensor_type, SensorType):
                errors.append(
                    "All compatible sensor types must use "
                    "SensorType enum values."
                )
                break

    @staticmethod
    def _validate_columns(
        required_columns: list[str],
        optional_columns: list[str],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate required and optional input field definitions.
        
        Args:
            required_columns: Value for `required_columns`.
            optional_columns: Value for `optional_columns`.
            errors: Value for `errors`.
            warnings: Value for `warnings`.
        
        Returns:
            None.
        """

        if not isinstance(required_columns, list):
            errors.append(
                "required_columns must be a list."
            )
            return

        if not isinstance(optional_columns, list):
            errors.append(
                "optional_columns must be a list."
            )
            return

        # Check that all entries are valid strings.

        for column in required_columns:
            if not isinstance(column, str) or not column.strip():
                errors.append(
                    "All required column names must be non-empty strings."
                )
                break

        for column in optional_columns:
            if not isinstance(column, str) or not column.strip():
                errors.append(
                    "All optional column names must be non-empty strings."
                )
                break

        # Check duplicates.

        duplicate_required = ToolValidator._find_duplicates(
            required_columns
        )

        if duplicate_required:
            warnings.append(
                "Duplicate required columns found: "
                + ", ".join(sorted(duplicate_required))
            )

        duplicate_optional = ToolValidator._find_duplicates(
            optional_columns
        )

        if duplicate_optional:
            warnings.append(
                "Duplicate optional columns found: "
                + ", ".join(sorted(duplicate_optional))
            )

        # A column should not be both required and optional.

        overlap = set(required_columns) & set(optional_columns)

        if overlap:
            errors.append(
                "The following fields are listed as both required "
                "and optional: "
                + ", ".join(sorted(overlap))
            )

    def _validate_parameters(
        self,
        parameters: list[ToolParameter],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate all configurable parameters declared by a tool.
        
        Args:
            parameters: Tool parameter values.
            errors: Value for `errors`.
            warnings: Value for `warnings`.
        
        Returns:
            None.
        """

        if not isinstance(parameters, list):
            errors.append(
                "Tool parameters must be a list."
            )
            return

        parameter_names: list[str] = []

        for parameter in parameters:

            if not isinstance(parameter, ToolParameter):
                errors.append(
                    "Every parameter must be a ToolParameter object."
                )
                continue

            parameter_names.append(parameter.name)

            self._validate_parameter(
                parameter,
                errors,
                warnings,
            )

        duplicates = self._find_duplicates(parameter_names)

        if duplicates:
            errors.append(
                "Duplicate parameter names found: "
                + ", ".join(sorted(duplicates))
            )

    def _validate_parameter(
        self,
        parameter: ToolParameter,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate one configurable analysis parameter.
        
        Args:
            parameter: Value for `parameter`.
            errors: Value for `errors`.
            warnings: Value for `warnings`.
        
        Returns:
            None.
        """

        if not isinstance(parameter.name, str) or not parameter.name.strip():
            errors.append(
                "Parameter names must be non-empty strings."
            )
            return

        if not self.TOOL_ID_PATTERN.fullmatch(parameter.name):
            errors.append(
                f"Parameter name '{parameter.name}' must use lowercase "
                "letters, numbers, and underscores."
            )

        if not isinstance(
            parameter.parameter_type,
            ParameterType,
        ):
            errors.append(
                f"Parameter '{parameter.name}' has an invalid "
                "parameter type."
            )
            return

        # Validate numeric boundaries

        if (
            parameter.minimum is not None
            and parameter.maximum is not None
            and parameter.minimum > parameter.maximum
        ):
            errors.append(
                f"Parameter '{parameter.name}' has minimum "
                f"{parameter.minimum} greater than maximum "
                f"{parameter.maximum}."
            )

        # Validate default value type

        self._validate_default_type(
            parameter,
            errors,
        )

        # Validate numerical default range

        if parameter.parameter_type in {
            ParameterType.INTEGER,
            ParameterType.FLOAT,
        }:

            if isinstance(parameter.default, (int, float)):

                if (
                    parameter.minimum is not None
                    and parameter.default < parameter.minimum
                ):
                    errors.append(
                        f"Default value for parameter "
                        f"'{parameter.name}' is below its minimum."
                    )

                if (
                    parameter.maximum is not None
                    and parameter.default > parameter.maximum
                ):
                    errors.append(
                        f"Default value for parameter "
                        f"'{parameter.name}' is above its maximum."
                    )

        # Validate CHOICE parameters

        if parameter.parameter_type == ParameterType.CHOICE:

            if not parameter.choices:
                errors.append(
                    f"Choice parameter '{parameter.name}' "
                    "must define at least one choice."
                )

            elif parameter.default not in parameter.choices:
                errors.append(
                    f"Default value '{parameter.default}' for "
                    f"choice parameter '{parameter.name}' "
                    "is not present in its allowed choices."
                )

        elif parameter.choices:
            warnings.append(
                f"Parameter '{parameter.name}' defines choices "
                "but is not a CHOICE parameter."
            )

    @staticmethod
    def _validate_default_type(
        parameter: ToolParameter,
        errors: list[str],
    ) -> None:
        """Check whether a parameter's default value matches
        its declared ParameterType.
        
        Args:
            parameter: Value for `parameter`.
            errors: Value for `errors`.
        
        Returns:
            None.
        """

        default = parameter.default

        if parameter.parameter_type == ParameterType.INTEGER:

            # bool is a subclass of int in Python, so explicitly reject it.
            if isinstance(default, bool) or not isinstance(default, int):
                errors.append(
                    f"Default value for integer parameter "
                    f"'{parameter.name}' must be an integer."
                )

        elif parameter.parameter_type == ParameterType.FLOAT:

            if (
                isinstance(default, bool)
                or not isinstance(default, (int, float))
            ):
                errors.append(
                    f"Default value for float parameter "
                    f"'{parameter.name}' must be numeric."
                )

        elif parameter.parameter_type == ParameterType.STRING:

            if not isinstance(default, str):
                errors.append(
                    f"Default value for string parameter "
                    f"'{parameter.name}' must be a string."
                )

        elif parameter.parameter_type == ParameterType.BOOLEAN:

            if not isinstance(default, bool):
                errors.append(
                    f"Default value for boolean parameter "
                    f"'{parameter.name}' must be True or False."
                )

        # CHOICE can contain arbitrary serializable values,
        # so membership is checked separately.

    @staticmethod
    def _validate_dependencies(
        current_tool_id: str,
        dependencies: list[str],
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate dependency declarations.
        
        Whether those dependencies actually exist is checked later
        once all tools have been loaded into the registry.
        
        Args:
            current_tool_id: Identifier for current tool.
            dependencies: Value for `dependencies`.
            errors: Value for `errors`.
            warnings: Value for `warnings`.
        
        Returns:
            None.
        """

        if not isinstance(dependencies, list):
            errors.append(
                "Tool dependencies must be a list."
            )
            return

        for dependency in dependencies:

            if not isinstance(dependency, str) or not dependency.strip():
                errors.append(
                    "All dependency IDs must be non-empty strings."
                )
                continue

            if dependency == current_tool_id:
                errors.append(
                    f"Tool '{current_tool_id}' cannot depend on itself."
                )

        duplicate_dependencies = ToolValidator._find_duplicates(
            dependencies
        )

        if duplicate_dependencies:
            warnings.append(
                "Duplicate dependencies found: "
                + ", ".join(sorted(duplicate_dependencies))
            )

    @staticmethod
    def _find_duplicates(
        values: list[str],
    ) -> set[str]:
        """Return duplicate values from a list.
        
        Args:
            values: Values to process.
        
        Returns:
            set[str] returned by the function.
        """

        seen: set[str] = set()
        duplicates: set[str] = set()

        for value in values:

            if value in seen:
                duplicates.add(value)

            seen.add(value)

        return duplicates
