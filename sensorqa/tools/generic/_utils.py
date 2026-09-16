#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def numeric_column(data: pd.DataFrame, column: str) -> np.ndarray:
    """Convert a DataFrame column to a floating-point NumPy array.
    
    Args:
        data: Input data to process.
        column: Column name.
    
    Returns:
        np.ndarray returned by this function.
    """
    return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float, copy=True)


def finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only positions where both input arrays are finite.
    
    Args:
        a: First input array.
        b: Second input array.
    
    Returns:
        Tuple containing the calculated values.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask], mask


def finite_many(*arrays: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Keep positions that are finite across every input array.
    
    Args:
        *arrays: Input arrays.
    
    Returns:
        List of result values.
    """
    if not arrays:
        raise ValueError("At least one array is required.")
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        if len(array) != len(arrays[0]):
            raise ValueError("All arrays must have the same length.")
        mask &= np.isfinite(array)
    return [array[mask] for array in arrays], mask


def safe_r2(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Calculate R-squared, returning None when the target has no variance.
    
    Args:
        actual: Reference or observed values.
        predicted: Predicted values.
    
    Returns:
        Calculated value.
    """
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    if np.isclose(total, 0.0, rtol=0.0, atol=1e-15):
        return None
    residual = float(np.sum((actual - predicted) ** 2))
    return float(1.0 - residual / total)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float | None]:
    """Fit a straight line and return its slope, intercept, and R-squared.
    
    Args:
        x: Independent-variable values.
        y: Dependent-variable values.
    
    Returns:
        Tuple containing the calculated values.
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("At least two paired observations are required.")
    if np.isclose(float(np.var(x)), 0.0, rtol=0.0, atol=1e-15):
        raise ValueError("Independent variable has effectively zero variance.")
    design = np.column_stack((x, np.ones_like(x)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        raise ValueError("Unable to determine independent slope and intercept.")
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    predicted = slope * x + intercept
    return slope, intercept, safe_r2(y, predicted)


def error_metrics(error: np.ndarray) -> dict[str, float]:
    """Calculate common error statistics for an error vector.
    
    Args:
        error: Error values.
    
    Returns:
        Dictionary containing the result values.
    """
    if error.size == 0:
        raise ValueError("At least one error value is required.")
    absolute = np.abs(error)
    return {
        "bias": float(np.mean(error)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "std": float(np.std(error, ddof=1)) if len(error) > 1 else 0.0,
        "max_abs": float(np.max(absolute)),
        "median_abs": float(np.median(absolute)),
    }


def sampling_information(timestamps: np.ndarray) -> dict[str, float] | None:
    """Calculate basic sampling statistics from timestamps.
    
    Args:
        timestamps: Timestamps used by this function.
    
    Returns:
        Dictionary containing the result values.
    """
    timestamps = timestamps[np.isfinite(timestamps)]
    if len(timestamps) < 2:
        return None
    intervals = np.diff(timestamps)
    positive = intervals[intervals > 0.0]
    if len(positive) == 0:
        return None
    median_dt = float(np.median(positive))
    mean_dt = float(np.mean(positive))
    if median_dt <= 0.0:
        return None
    std_dt = float(np.std(positive, ddof=1)) if len(positive) > 1 else 0.0
    cv = float(std_dt / mean_dt) if mean_dt > 0.0 else float("inf")
    return {
        "median_interval": median_dt,
        "mean_interval": mean_dt,
        "sampling_frequency_hz": float(1.0 / median_dt),
        "coefficient_of_variation": cv,
        "positive_interval_count": float(len(positive)),
    }


def group_reference_indices(reference: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Group row positions by repeated reference condition.
    
    Args:
        reference: Reference used by this function.
        tolerance: Tolerance used by this function.
    
    Returns:
        List of result values.
    """
    if reference.size == 0:
        return []
    if tolerance == 0.0:
        return [np.flatnonzero(reference == value) for value in np.unique(reference)]
    order = np.argsort(reference, kind="stable")
    groups: list[list[int]] = []
    current: list[int] = []
    for raw_position in order:
        position = int(raw_position)
        if not current:
            current = [position]
            continue
        center = float(np.mean(reference[np.asarray(current, dtype=int)]))
        if abs(float(reference[position]) - center) <= tolerance:
            current.append(position)
        else:
            groups.append(current)
            current = [position]
    if current:
        groups.append(current)
    return [np.asarray(group, dtype=int) for group in groups]


def unit_ratio(numerator: str | None, denominator: str | None) -> str | None:
    """Build a readable unit for a ratio.
    
    Args:
        numerator: Numerator used by this function.
        denominator: Denominator used by this function.
    
    Returns:
        str | None returned by this function.
    """
    if numerator is None or denominator is None:
        return None
    if numerator == denominator:
        return None
    return f"{numerator}/{denominator}"


def add_exclusion_warning(result, excluded: int, total: int, label: str = "rows") -> None:
    """Add a warning when rows were excluded from an analysis.
    
    Args:
        result: Result object to process.
        excluded: Excluded used by this function.
        total: Total used by this function.
        label: Label shown to the user.
    
    Returns:
        None.
    """
    if excluded > 0:
        result.add_warning(
            f"{excluded} of {total} {label} were excluded because required values "
            "were non-numeric or non-finite."
        )
