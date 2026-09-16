#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

from sensorqa.core.requirements_engine import (
    RequirementDefinition,
    RequirementOperator,
    RequirementSet,
)


@dataclass
class RequirementLoadError:
    """
    One validation/parsing problem found while loading a
    requirements TOML file.
    """

    location: str
    message: str


@dataclass
class RequirementLoadReport:
    """
    Result returned by RequirementsLoader.

    A file can be parsed successfully at the TOML level but still
    fail SensorQA semantic validation.
    """

    success: bool

    requirement_set: RequirementSet | None = None

    file_path: str | None = None

    errors: list[
        RequirementLoadError
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
    def requirement_count(
        self,
    ) -> int:

        """Return the number of loaded requirements.
        
        Returns:
            Calculated value.
        """
        if self.requirement_set is None:

            return 0

        return len(
            self.requirement_set.requirements
        )

    @property
    def enabled_requirement_count(
        self,
    ) -> int:

        """Return the number of enabled requirements.
        
        Returns:
            Calculated value.
        """
        if self.requirement_set is None:

            return 0

        return len(
            self.requirement_set.enabled_requirements()
        )


class RequirementsLoader:
    """
    Loads and validates SensorQA engineering requirements from TOML.

    Responsibilities:

        TOML parsing
        profile validation
        requirement field validation
        duplicate-ID detection
        operator validation
        numerical-limit validation
        RequirementDefinition construction
        RequirementSet construction

    The loader does NOT evaluate sensor measurements.
    """

    PROFILE_KEY = "profile"
    REQUIREMENTS_KEY = "requirements"

    def load(
        self,
        source: str | Path,
    ) -> RequirementLoadReport:
        """Load a requirements TOML file.
        
        Returns a report rather than raising for expected user
        configuration errors.
        
        Args:
            source: Input source or source path.
        
        Returns:
            RequirementLoadReport returned by the function.
        """

        path = Path(
            source
        )

        report = RequirementLoadReport(
            success=False,
            file_path=str(
                path
            ),
        )

        # File checks

        if not path.exists():

            report.errors.append(
                RequirementLoadError(
                    location="file",
                    message=(
                        f"Requirements file does not exist: "
                        f"{path}"
                    ),
                )
            )

            return report

        if not path.is_file():

            report.errors.append(
                RequirementLoadError(
                    location="file",
                    message=(
                        f"Requirements path is not a file: "
                        f"{path}"
                    ),
                )
            )

            return report

        # TOML parsing

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
                RequirementLoadError(
                    location="file",
                    message=(
                        f"Could not parse requirements TOML: "
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
                RequirementLoadError(
                    location="file",
                    message=(
                        "Top-level TOML content must be a table."
                    ),
                )
            )

            return report

        # Profile

        profile_data = raw.get(
            self.PROFILE_KEY
        )

        profile = self._parse_profile(
            profile_data=profile_data,
            report=report,
        )

        # Requirements list

        requirements_data = raw.get(
            self.REQUIREMENTS_KEY
        )

        if requirements_data is None:

            report.errors.append(
                RequirementLoadError(
                    location="requirements",
                    message=(
                        "No [[requirements]] entries were found."
                    ),
                )
            )

            return report

        if not isinstance(
            requirements_data,
            list,
        ):

            report.errors.append(
                RequirementLoadError(
                    location="requirements",
                    message=(
                        "'requirements' must be an array of TOML "
                        "tables using [[requirements]]."
                    ),
                )
            )

            return report

        if len(
            requirements_data
        ) == 0:

            report.warnings.append(
                "Requirements file contains no requirement entries."
            )

        definitions: list[
            RequirementDefinition
        ] = []

        seen_ids: set[str] = set()

        for index, requirement_data in enumerate(
            requirements_data
        ):

            location = (
                f"requirements[{index}]"
            )

            definition = self._parse_requirement(
                requirement_data=requirement_data,
                location=location,
                report=report,
            )

            if definition is None:

                continue

            if (
                definition.requirement_id
                in seen_ids
            ):

                report.errors.append(
                    RequirementLoadError(
                        location=location,
                        message=(
                            "Duplicate requirement ID "
                            f"'{definition.requirement_id}'."
                        ),
                    )
                )

                continue

            seen_ids.add(
                definition.requirement_id
            )

            definitions.append(
                definition
            )

        # Cannot construct a usable set if profile validation or
        # requirement validation failed.

        if (
            profile is None
            or report.errors
        ):

            return report

        requirement_set = RequirementSet(
            name=profile[
                "name"
            ],
            requirements=definitions,
            description=profile.get(
                "description"
            ),
            version=profile.get(
                "version"
            ),
            metadata={
                key: value
                for key, value
                in profile.items()
                if key
                not in {
                    "name",
                    "description",
                    "version",
                }
            },
        )

        # Additional semantic warnings

        enabled_count = len(
            requirement_set.enabled_requirements()
        )

        if enabled_count == 0:

            report.warnings.append(
                "Requirement profile contains no enabled "
                "requirements."
            )

        self._check_metric_duplicates(
            requirement_set=requirement_set,
            report=report,
        )

        report.requirement_set = (
            requirement_set
        )

        report.success = True

        return report

    # Profile parsing

    @staticmethod
    def _parse_profile(
        profile_data: Any,
        report: RequirementLoadReport,
    ) -> dict[str, Any] | None:
        """Validate [profile].
        
        Args:
            profile_data: Value for `profile_data`.
            report: Value for `report`.
        
        Returns:
            Dictionary containing the result values.
        """

        if profile_data is None:

            report.errors.append(
                RequirementLoadError(
                    location="profile",
                    message=(
                        "Missing required [profile] section."
                    ),
                )
            )

            return None

        if not isinstance(
            profile_data,
            dict,
        ):

            report.errors.append(
                RequirementLoadError(
                    location="profile",
                    message=(
                        "[profile] must be a TOML table."
                    ),
                )
            )

            return None

        name = profile_data.get(
            "name"
        )

        if not isinstance(
            name,
            str,
        ) or not name.strip():

            report.errors.append(
                RequirementLoadError(
                    location="profile.name",
                    message=(
                        "Profile name must be a non-empty string."
                    ),
                )
            )

        version = profile_data.get(
            "version"
        )

        if (
            version is not None
            and (
                not isinstance(
                    version,
                    str,
                )
                or not version.strip()
            )
        ):

            report.errors.append(
                RequirementLoadError(
                    location="profile.version",
                    message=(
                        "Profile version must be a non-empty string "
                        "when provided."
                    ),
                )
            )

        description = profile_data.get(
            "description"
        )

        if (
            description is not None
            and not isinstance(
                description,
                str,
            )
        ):

            report.errors.append(
                RequirementLoadError(
                    location="profile.description",
                    message=(
                        "Profile description must be a string."
                    ),
                )
            )

        return dict(
            profile_data
        )

    # Requirement parsing

    @classmethod
    def _parse_requirement(
        cls,
        requirement_data: Any,
        location: str,
        report: RequirementLoadReport,
    ) -> RequirementDefinition | None:
        """Validate and construct one RequirementDefinition.
        
        Args:
            requirement_data: Value for `requirement_data`.
            location: Value for `location`.
            report: Value for `report`.
        
        Returns:
            RequirementDefinition | None returned by the function.
        """

        if not isinstance(
            requirement_data,
            dict,
        ):

            report.errors.append(
                RequirementLoadError(
                    location=location,
                    message=(
                        "Requirement entry must be a TOML table."
                    ),
                )
            )

            return None

        requirement_id = cls._required_string(
            data=requirement_data,
            key="id",
            location=location,
            report=report,
        )

        tool_id = cls._required_string(
            data=requirement_data,
            key="tool_id",
            location=location,
            report=report,
        )

        metric_name = cls._required_string(
            data=requirement_data,
            key="metric",
            location=location,
            report=report,
        )

        description = cls._required_string(
            data=requirement_data,
            key="description",
            location=location,
            report=report,
        )

        operator_text = cls._required_string(
            data=requirement_data,
            key="operator",
            location=location,
            report=report,
        )

        enabled = requirement_data.get(
            "enabled",
            True,
        )

        if not isinstance(
            enabled,
            bool,
        ):

            report.errors.append(
                RequirementLoadError(
                    location=(
                        f"{location}.enabled"
                    ),
                    message=(
                        "'enabled' must be true or false."
                    ),
                )
            )

            enabled = True

        unit = requirement_data.get(
            "unit"
        )

        if (
            unit is not None
            and (
                not isinstance(
                    unit,
                    str,
                )
                or not unit.strip()
            )
        ):

            report.errors.append(
                RequirementLoadError(
                    location=(
                        f"{location}.unit"
                    ),
                    message=(
                        "'unit' must be a non-empty string when "
                        "provided."
                    ),
                )
            )

            unit = None

        operator = None

        if operator_text is not None:

            try:

                operator = (
                    RequirementOperator(
                        operator_text.strip()
                    )
                )

            except ValueError:

                allowed = ", ".join(
                    operator.value
                    for operator
                    in RequirementOperator
                )

                report.errors.append(
                    RequirementLoadError(
                        location=(
                            f"{location}.operator"
                        ),
                        message=(
                            f"Unsupported operator "
                            f"'{operator_text}'. Supported "
                            f"operators: {allowed}."
                        ),
                    )
                )

        # Limits

        limit_value = None
        lower_limit = None
        upper_limit = None

        if (
            operator
            == RequirementOperator.BETWEEN
        ):

            lower_limit = cls._required_number(
                data=requirement_data,
                key="lower_limit",
                location=location,
                report=report,
            )

            upper_limit = cls._required_number(
                data=requirement_data,
                key="upper_limit",
                location=location,
                report=report,
            )

            if (
                lower_limit is not None
                and upper_limit is not None
                and lower_limit
                > upper_limit
            ):

                report.errors.append(
                    RequirementLoadError(
                        location=location,
                        message=(
                            "lower_limit cannot be greater than "
                            "upper_limit."
                        ),
                    )
                )

            if "limit" in requirement_data:

                report.warnings.append(
                    f"{location} uses operator 'between' and also "
                    "defines 'limit'. The 'limit' value will be "
                    "ignored."
                )

        elif operator is not None:

            limit_value = cls._required_number(
                data=requirement_data,
                key="limit",
                location=location,
                report=report,
            )

            if (
                "lower_limit"
                in requirement_data
                or "upper_limit"
                in requirement_data
            ):

                report.warnings.append(
                    f"{location} does not use operator 'between', "
                    "so lower_limit/upper_limit values will be "
                    "ignored."
                )

        # Do not construct if required fields failed.

        if any(
            value is None
            for value in (
                requirement_id,
                tool_id,
                metric_name,
                description,
                operator,
            )
        ):

            return None

        # RequirementDefinition performs another validation layer.
        try:

            definition = RequirementDefinition(
                requirement_id=(
                    requirement_id.strip()
                ),
                tool_id=tool_id.strip(),
                metric_name=(
                    metric_name.strip()
                ),
                operator=operator,
                description=(
                    description.strip()
                ),
                unit=(
                    unit.strip()
                    if isinstance(
                        unit,
                        str,
                    )
                    else None
                ),
                limit_value=limit_value,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                enabled=enabled,
                metadata=cls._extra_requirement_metadata(
                    requirement_data
                ),
            )

        except (
            ValueError,
            TypeError,
        ) as exc:

            report.errors.append(
                RequirementLoadError(
                    location=location,
                    message=str(
                        exc
                    ),
                )
            )

            return None

        return definition

    # Required values

    @staticmethod
    def _required_string(
        data: dict[str, Any],
        key: str,
        location: str,
        report: RequirementLoadReport,
    ) -> str | None:

        """Read required string.
        
        Args:
            data: Input data to process.
            key: Key used by this function.
            location: Location used by this function.
            report: Report used by this function.
        
        Returns:
            str | None returned by this function.
        """
        value = data.get(
            key
        )

        if value is None:

            report.errors.append(
                RequirementLoadError(
                    location=(
                        f"{location}.{key}"
                    ),
                    message=(
                        f"Missing required field '{key}'."
                    ),
                )
            )

            return None

        if not isinstance(
            value,
            str,
        ):

            report.errors.append(
                RequirementLoadError(
                    location=(
                        f"{location}.{key}"
                    ),
                    message=(
                        f"'{key}' must be a string."
                    ),
                )
            )

            return None

        if not value.strip():

            report.errors.append(
                RequirementLoadError(
                    location=(
                        f"{location}.{key}"
                    ),
                    message=(
                        f"'{key}' cannot be empty."
                    ),
                )
            )

            return None

        return value

    @staticmethod
    def _required_number(
        data: dict[str, Any],
        key: str,
        location: str,
        report: RequirementLoadReport,
    ) -> float | None:

        """Read required number.
        
        Args:
            data: Input data to process.
            key: Key used by this function.
            location: Location used by this function.
            report: Report used by this function.
        
        Returns:
            Calculated value.
        """
        value = data.get(
            key
        )

        if value is None:

            report.errors.append(
                RequirementLoadError(
                    location=(
                        f"{location}.{key}"
                    ),
                    message=(
                        f"Missing required numerical field '{key}'."
                    ),
                )
            )

            return None

        # bool is technically an int in Python, so explicitly
        # reject it.
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
                RequirementLoadError(
                    location=(
                        f"{location}.{key}"
                    ),
                    message=(
                        f"'{key}' must be a numerical value."
                    ),
                )
            )

            return None

        numeric_value = float(
            value
        )

        if not cls_is_finite(
            numeric_value
        ):

            report.errors.append(
                RequirementLoadError(
                    location=(
                        f"{location}.{key}"
                    ),
                    message=(
                        f"'{key}' must be finite."
                    ),
                )
            )

            return None

        return numeric_value

    # Extra metadata

    @staticmethod
    def _extra_requirement_metadata(
        requirement_data: dict[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        """Preserve unknown user-defined fields rather than throwing
        them away.
        
        This allows future configuration such as:
        
            severity = "critical"
            owner = "sensor_team"
            requirement_source = "SPEC-001"
        
        without changing the loader.
        
        Args:
            requirement_data: Value for `requirement_data`.
        
        Returns:
            Dictionary containing the result values.
        """

        reserved = {
            "id",
            "enabled",
            "tool_id",
            "metric",
            "operator",
            "limit",
            "lower_limit",
            "upper_limit",
            "unit",
            "description",
        }

        return {
            key: value
            for key, value
            in requirement_data.items()
            if key not in reserved
        }

    # Semantic warnings

    @staticmethod
    def _check_metric_duplicates(
        requirement_set: RequirementSet,
        report: RequirementLoadReport,
    ) -> None:
        """Warn if exactly the same tool/metric pair appears multiple
        times.
        
        Multiple requirements can be intentional, so this is a
        warning rather than an error.
        
        Example:
            temperature >= minimum
            temperature <= maximum
        
        Args:
            requirement_set: Value for `requirement_set`.
            report: Value for `report`.
        
        Returns:
            None.
        """

        occurrences: dict[
            tuple[str, str],
            list[str],
        ] = {}

        for requirement in (
            requirement_set.enabled_requirements()
        ):

            key = (
                requirement.tool_id,
                requirement.metric_name,
            )

            occurrences.setdefault(
                key,
                [],
            ).append(
                requirement.requirement_id
            )

        for (
            tool_id,
            metric_name,
        ), requirement_ids in occurrences.items():

            if len(
                requirement_ids
            ) <= 1:

                continue

            report.warnings.append(
                "Multiple enabled requirements reference metric "
                f"'{metric_name}' from tool '{tool_id}': "
                + ", ".join(
                    requirement_ids
                )
            )


def cls_is_finite(
    value: float,
) -> bool:
    """Small dependency-free finite-value helper.
    
    Args:
        value: Value to process.
    
    Returns:
        Boolean result.
    """

    return (
        value == value
        and value
        not in (
            float("inf"),
            float("-inf"),
        )
    )
