#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class ValidationSeverity(str, Enum):
    """
    Severity assigned to a dataset-quality finding.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class DatasetIssue:
    """
    One data-quality issue detected in a dataset.
    """

    issue_id: str
    severity: ValidationSeverity
    message: str

    column: str | None = None
    affected_rows: int | None = None
    affected_fraction: float | None = None

    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingSummary:
    """
    Summary of timestamp and sampling behavior.
    """

    available: bool = False

    sample_count: int = 0

    duration_seconds: float | None = None

    median_interval_seconds: float | None = None
    mean_interval_seconds: float | None = None
    std_interval_seconds: float | None = None

    estimated_sampling_rate_hz: float | None = None

    duplicate_timestamp_count: int = 0
    backward_timestamp_count: int = 0

    large_gap_count: int = 0
    largest_gap_seconds: float | None = None

    coefficient_of_variation: float | None = None


@dataclass
class DatasetValidationResult:
    """
    Complete result of dataset-quality validation.
    """

    valid: bool

    row_count: int
    column_count: int

    issues: list[DatasetIssue] = field(default_factory=list)

    sampling: SamplingSummary = field(
        default_factory=SamplingSummary
    )

    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[DatasetIssue]:
        """Return validation errors.
        
        Returns:
            List of result values.
        """
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.ERROR
        ]

    @property
    def warnings(self) -> list[DatasetIssue]:
        """Return validation warnings.
        
        Returns:
            List of result values.
        """
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        ]

    @property
    def information(self) -> list[DatasetIssue]:
        """Return informational validation messages.
        
        Returns:
            List of result values.
        """
        return [
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.INFO
        ]


class DatasetValidator:
    """
    Performs structural and numerical quality checks on a SensorQA dataset.

    Expected input:
        - columns have already been mapped to SensorQA standard names
        - known physical units have already been normalized
    """

    def __init__(
        self,
        minimum_rows: int = 10,
        missing_warning_fraction: float = 0.01,
        missing_error_fraction: float = 0.20,
        irregular_sampling_cv_threshold: float = 0.05,
        large_gap_multiplier: float = 5.0,
    ) -> None:

        """Initialize the dataset validator.
        
        Args:
            minimum_rows: Minimum rows accepted.
            missing_warning_fraction: Missing warning fraction used by this function.
            missing_error_fraction: Missing error fraction used by this function.
            irregular_sampling_cv_threshold: Irregular sampling cv threshold used by this function.
            large_gap_multiplier: Large gap multiplier used by this function.
        
        Returns:
            None.
        """
        if minimum_rows < 1:
            raise ValueError(
                "minimum_rows must be at least 1."
            )

        if not 0 <= missing_warning_fraction <= 1:
            raise ValueError(
                "missing_warning_fraction must be between 0 and 1."
            )

        if not 0 <= missing_error_fraction <= 1:
            raise ValueError(
                "missing_error_fraction must be between 0 and 1."
            )

        if missing_error_fraction < missing_warning_fraction:
            raise ValueError(
                "missing_error_fraction must be greater than or "
                "equal to missing_warning_fraction."
            )

        if irregular_sampling_cv_threshold < 0:
            raise ValueError(
                "irregular_sampling_cv_threshold cannot be negative."
            )

        if large_gap_multiplier <= 1:
            raise ValueError(
                "large_gap_multiplier must be greater than 1."
            )

        self.minimum_rows = minimum_rows
        self.missing_warning_fraction = missing_warning_fraction
        self.missing_error_fraction = missing_error_fraction

        self.irregular_sampling_cv_threshold = (
            irregular_sampling_cv_threshold
        )

        self.large_gap_multiplier = large_gap_multiplier

    def validate(
        self,
        data: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        timestamp_column: str = "timestamp",
    ) -> DatasetValidationResult:
        """Validate one mapped SensorQA dataset.
        
        Parameters
        ----------
        data:
            Mapped and unit-normalized SensorQA DataFrame.
        
        numeric_columns:
            Columns expected to contain numerical data.
            If omitted, SensorQA inspects numeric-looking columns
            automatically.
        
        timestamp_column:
            Standard timestamp field name.
        
        Args:
            data: Input data to process.
            numeric_columns: Value for `numeric_columns`.
            timestamp_column: Column used for timestamp.
        
        Returns:
            Validation result.
        """

        issues: list[DatasetIssue] = []

        row_count = len(data)
        column_count = len(data.columns)

        # Empty dataset

        if row_count == 0:

            issues.append(
                DatasetIssue(
                    issue_id="empty_dataset",
                    severity=ValidationSeverity.ERROR,
                    message="Dataset contains no data rows.",
                )
            )

            return DatasetValidationResult(
                valid=False,
                row_count=row_count,
                column_count=column_count,
                issues=issues,
            )

        # Very short dataset

        if row_count < self.minimum_rows:

            issues.append(
                DatasetIssue(
                    issue_id="short_dataset",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Dataset contains only {row_count} rows. "
                        "Some statistical analyses may not be reliable."
                    ),
                    affected_rows=row_count,
                )
            )

        # Duplicate complete rows

        duplicate_row_mask = data.duplicated(
            keep=False
        )

        duplicate_row_count = int(
            duplicate_row_mask.sum()
        )

        if duplicate_row_count > 0:

            issues.append(
                DatasetIssue(
                    issue_id="duplicate_rows",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"{duplicate_row_count} rows belong to "
                        "duplicate row groups."
                    ),
                    affected_rows=duplicate_row_count,
                    affected_fraction=(
                        duplicate_row_count / row_count
                    ),
                )
            )

        # Missing values

        self._check_missing_values(
            data=data,
            issues=issues,
        )

        # Determine numerical columns

        if numeric_columns is None:
            numeric_columns = self._infer_numeric_columns(
                data
            )

        else:
            numeric_columns = [
                column
                for column in numeric_columns
                if column in data.columns
            ]

        # Non-numeric values

        self._check_numeric_columns(
            data=data,
            numeric_columns=numeric_columns,
            issues=issues,
        )

        # Infinite values

        self._check_infinite_values(
            data=data,
            numeric_columns=numeric_columns,
            issues=issues,
        )

        # Constant channels

        self._check_constant_columns(
            data=data,
            numeric_columns=numeric_columns,
            issues=issues,
        )

        # Timestamp / sampling analysis

        sampling_summary = self._analyze_sampling(
            data=data,
            timestamp_column=timestamp_column,
            issues=issues,
        )

        # High-level dataset metrics

        total_cells = (
            row_count * column_count
        )

        total_missing = int(
            data.isna().sum().sum()
        )

        metrics = {
            "total_cells": total_cells,
            "total_missing_values": total_missing,
            "overall_missing_fraction": (
                total_missing / total_cells
                if total_cells > 0
                else 0.0
            ),
            "duplicate_row_count": duplicate_row_count,
            "numeric_column_count": len(numeric_columns),
        }

        # Dataset is invalid only if at least one ERROR exists.

        valid = not any(
            issue.severity == ValidationSeverity.ERROR
            for issue in issues
        )

        return DatasetValidationResult(
            valid=valid,
            row_count=row_count,
            column_count=column_count,
            issues=issues,
            sampling=sampling_summary,
            metrics=metrics,
        )

    # Missing-data checks

    def _check_missing_values(
        self,
        data: pd.DataFrame,
        issues: list[DatasetIssue],
    ) -> None:
        """Inspect missing values column by column.
        
        Args:
            data: Input data to process.
            issues: Value for `issues`.
        
        Returns:
            None.
        """

        row_count = len(data)

        for column in data.columns:

            missing_count = int(
                data[column].isna().sum()
            )

            if missing_count == 0:
                continue

            missing_fraction = (
                missing_count / row_count
            )

            if missing_fraction >= self.missing_error_fraction:

                severity = ValidationSeverity.ERROR

            elif (
                missing_fraction
                >= self.missing_warning_fraction
            ):

                severity = ValidationSeverity.WARNING

            else:

                severity = ValidationSeverity.INFO

            issues.append(
                DatasetIssue(
                    issue_id="missing_values",
                    severity=severity,
                    message=(
                        f"Column '{column}' contains "
                        f"{missing_count} missing values "
                        f"({missing_fraction:.2%})."
                    ),
                    column=str(column),
                    affected_rows=missing_count,
                    affected_fraction=missing_fraction,
                )
            )

    # Numerical-data checks

    @staticmethod
    def _infer_numeric_columns(
        data: pd.DataFrame,
    ) -> list[str]:
        """Identify columns already represented as numeric types.
        
        Args:
            data: Input data to process.
        
        Returns:
            List of result values.
        """

        return [
            str(column)
            for column in data.select_dtypes(
                include=[np.number]
            ).columns
        ]

    def _check_numeric_columns(
        self,
        data: pd.DataFrame,
        numeric_columns: list[str],
        issues: list[DatasetIssue],
    ) -> None:
        """Find values that cannot be converted to numbers in channels
        expected to be numerical.
        
        Args:
            data: Input data to process.
            numeric_columns: Value for `numeric_columns`.
            issues: Value for `issues`.
        
        Returns:
            None.
        """

        row_count = len(data)

        for column in numeric_columns:

            original = data[column]

            converted = pd.to_numeric(
                original,
                errors="coerce",
            )

            invalid_mask = (
                original.notna()
                & converted.isna()
            )

            invalid_count = int(
                invalid_mask.sum()
            )

            if invalid_count == 0:
                continue

            invalid_fraction = (
                invalid_count / row_count
            )

            severity = (
                ValidationSeverity.ERROR
                if invalid_fraction >= 0.05
                else ValidationSeverity.WARNING
            )

            issues.append(
                DatasetIssue(
                    issue_id="non_numeric_values",
                    severity=severity,
                    message=(
                        f"Column '{column}' contains "
                        f"{invalid_count} non-numeric values."
                    ),
                    column=column,
                    affected_rows=invalid_count,
                    affected_fraction=invalid_fraction,
                )
            )

    @staticmethod
    def _check_infinite_values(
        data: pd.DataFrame,
        numeric_columns: list[str],
        issues: list[DatasetIssue],
    ) -> None:
        """Detect +inf and -inf values.
        
        Args:
            data: Input data to process.
            numeric_columns: Value for `numeric_columns`.
            issues: Value for `issues`.
        
        Returns:
            None.
        """

        row_count = len(data)

        for column in numeric_columns:

            converted = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            infinite_mask = np.isinf(
                converted.to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            )

            infinite_count = int(
                infinite_mask.sum()
            )

            if infinite_count == 0:
                continue

            issues.append(
                DatasetIssue(
                    issue_id="infinite_values",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Column '{column}' contains "
                        f"{infinite_count} infinite values."
                    ),
                    column=column,
                    affected_rows=infinite_count,
                    affected_fraction=(
                        infinite_count / row_count
                    ),
                )
            )

    @staticmethod
    def _check_constant_columns(
        data: pd.DataFrame,
        numeric_columns: list[str],
        issues: list[DatasetIssue],
    ) -> None:
        """Detect numeric channels that never change.
        
        Args:
            data: Input data to process.
            numeric_columns: Value for `numeric_columns`.
            issues: Value for `issues`.
        
        Returns:
            None.
        """

        for column in numeric_columns:

            converted = pd.to_numeric(
                data[column],
                errors="coerce",
            ).dropna()

            if converted.empty:
                continue

            unique_count = int(
                converted.nunique()
            )

            if unique_count == 1:

                issues.append(
                    DatasetIssue(
                        issue_id="constant_channel",
                        severity=ValidationSeverity.INFO,
                        message=(
                            f"Numeric column '{column}' is constant "
                            "throughout the dataset."
                        ),
                        column=column,
                        affected_rows=len(converted),
                        details={
                            "constant_value": float(
                                converted.iloc[0]
                            )
                        },
                    )
                )

    # Timestamp / sampling checks

    def _analyze_sampling(
        self,
        data: pd.DataFrame,
        timestamp_column: str,
        issues: list[DatasetIssue],
    ) -> SamplingSummary:
        """Analyze timestamp integrity and basic sampling behavior.
        
        Args:
            data: Input data to process.
            timestamp_column: Column used for timestamp.
            issues: Value for `issues`.
        
        Returns:
            SamplingSummary returned by the function.
        """

        if timestamp_column not in data.columns:

            issues.append(
                DatasetIssue(
                    issue_id="timestamp_unavailable",
                    severity=ValidationSeverity.INFO,
                    message=(
                        "No timestamp column is available. "
                        "Time-dependent analyses will be limited."
                    ),
                )
            )

            return SamplingSummary(
                available=False
            )

        timestamps = pd.to_numeric(
            data[timestamp_column],
            errors="coerce",
        )

        valid_timestamps = timestamps.dropna()

        if len(valid_timestamps) < 2:

            issues.append(
                DatasetIssue(
                    issue_id="insufficient_timestamps",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        "At least two valid timestamps are required "
                        "for sampling analysis."
                    ),
                    column=timestamp_column,
                )
            )

            return SamplingSummary(
                available=True,
                sample_count=len(valid_timestamps),
            )

        timestamp_array = valid_timestamps.to_numpy(
            dtype=float
        )

        intervals = np.diff(
            timestamp_array
        )

        duplicate_count = int(
            np.sum(intervals == 0)
        )

        backward_count = int(
            np.sum(intervals < 0)
        )

        positive_intervals = intervals[
            intervals > 0
        ]

        if duplicate_count > 0:

            issues.append(
                DatasetIssue(
                    issue_id="duplicate_timestamps",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"{duplicate_count} duplicate timestamp "
                        "intervals were detected."
                    ),
                    column=timestamp_column,
                    affected_rows=duplicate_count,
                )
            )

        if backward_count > 0:

            issues.append(
                DatasetIssue(
                    issue_id="backward_timestamps",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"{backward_count} timestamp intervals move "
                        "backward in time."
                    ),
                    column=timestamp_column,
                    affected_rows=backward_count,
                )
            )

        if len(positive_intervals) == 0:

            issues.append(
                DatasetIssue(
                    issue_id="no_positive_time_intervals",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        "No positive timestamp intervals were found."
                    ),
                    column=timestamp_column,
                )
            )

            return SamplingSummary(
                available=True,
                sample_count=len(valid_timestamps),
                duplicate_timestamp_count=duplicate_count,
                backward_timestamp_count=backward_count,
            )

        median_interval = float(
            np.median(
                positive_intervals
            )
        )

        mean_interval = float(
            np.mean(
                positive_intervals
            )
        )

        std_interval = float(
            np.std(
                positive_intervals,
                ddof=0,
            )
        )

        sampling_rate = (
            1.0 / median_interval
            if median_interval > 0
            else None
        )

        duration = float(
            timestamp_array[-1]
            - timestamp_array[0]
        )

        coefficient_of_variation = (
            std_interval / mean_interval
            if mean_interval > 0
            else None
        )

        # Irregular sampling

        if (
            coefficient_of_variation is not None
            and coefficient_of_variation
            > self.irregular_sampling_cv_threshold
        ):

            issues.append(
                DatasetIssue(
                    issue_id="irregular_sampling",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        "Sampling intervals are irregular. "
                        f"Interval coefficient of variation = "
                        f"{coefficient_of_variation:.4f}."
                    ),
                    column=timestamp_column,
                    details={
                        "coefficient_of_variation":
                            coefficient_of_variation,
                    },
                )
            )

        # Large sampling gaps

        large_gap_threshold = (
            median_interval
            * self.large_gap_multiplier
        )

        large_gaps = positive_intervals[
            positive_intervals
            > large_gap_threshold
        ]

        large_gap_count = len(
            large_gaps
        )

        largest_gap = (
            float(np.max(large_gaps))
            if large_gap_count > 0
            else None
        )

        if large_gap_count > 0:

            issues.append(
                DatasetIssue(
                    issue_id="large_time_gaps",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"{large_gap_count} large timestamp gaps "
                        f"were detected. A gap is classified as "
                        f"greater than "
                        f"{self.large_gap_multiplier:.1f} times "
                        "the median sample interval."
                    ),
                    column=timestamp_column,
                    affected_rows=large_gap_count,
                    details={
                        "threshold_seconds":
                            large_gap_threshold,
                        "largest_gap_seconds":
                            largest_gap,
                    },
                )
            )

        return SamplingSummary(
            available=True,
            sample_count=len(valid_timestamps),
            duration_seconds=duration,
            median_interval_seconds=median_interval,
            mean_interval_seconds=mean_interval,
            std_interval_seconds=std_interval,
            estimated_sampling_rate_hz=sampling_rate,
            duplicate_timestamp_count=duplicate_count,
            backward_timestamp_count=backward_count,
            large_gap_count=large_gap_count,
            largest_gap_seconds=largest_gap,
            coefficient_of_variation=coefficient_of_variation,
        )
