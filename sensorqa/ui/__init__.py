"""SensorQA desktop user interface package.

This module also contains small display-formatting helpers shared by the UI.
They convert internal names and engineering-unit strings into readable labels
without changing the values passed to the analysis backend.
"""

from __future__ import annotations

import re


_WORDS = {
    "imu": "IMU",
    "psd": "PSD",
    "asd": "ASD",
    "rmse": "RMSE",
    "mae": "MAE",
    "snr": "SNR",
    "r2": "R²",
    "r_squared": "R²",
    "cv": "CV",
    "id": "ID",
    "csv": "CSV",
    "json": "JSON",
    "html": "HTML",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "ax": "Accel X",
    "ay": "Accel Y",
    "az": "Accel Z",
    "gx": "Gyro X",
    "gy": "Gyro Y",
    "gz": "Gyro Z",
}


def humanize_identifier(value: object) -> str:
    """Convert an internal identifier into a readable UI label.

    Args:
        value: Identifier or other value to format for display.

    Returns:
        A user-facing label with underscores removed and common engineering
        acronyms preserved.
    """

    text = str(value or "").strip()
    if not text:
        return ""

    if text.lower() in _WORDS:
        return _WORDS[text.lower()]

    parts = re.split(r"[_\s]+", text)
    formatted: list[str] = []
    for part in parts:
        key = part.lower()
        if key in _WORDS:
            formatted.append(_WORDS[key])
        elif part.isupper() and len(part) <= 5:
            formatted.append(part)
        else:
            formatted.append(part[:1].upper() + part[1:])
    return " ".join(formatted)


def humanize_unit(unit: object | None) -> str:
    """Convert an internal engineering-unit string to readable notation.

    Args:
        unit: Unit string used by the backend, such as ``m/s^2`` or
            ``(rad/s)^2/Hz``. ``None`` is allowed.

    Returns:
        A display-only unit string using symbols such as ``²``, ``°C`` and
        ``√Hz``. The backend unit is not modified.
    """

    if unit is None:
        return ""

    text = str(unit).strip()
    if not text:
        return ""

    replacements = {
        "degC": "°C",
        "degF": "°F",
        "deg/s": "°/s",
        "sqrt(Hz)": "√Hz",
        "sqrt(s)": "√s",
        "^0.5": "½",
        "^2": "²",
        "^3": "³",
        "*": "·",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    # A few internal spellings occasionally appear in metadata or plug-ins.
    text = text.replace("degrees C", "°C")
    text = text.replace("degrees F", "°F")
    return text


def machine_unit(unit: object | None) -> str:
    """Convert common display units back to SensorQA's internal notation.

    Args:
        unit: Unit entered or displayed in the desktop interface.

    Returns:
        Unit string suitable for the existing ingestion and unit-management
        code. Unknown units are returned unchanged.
    """

    text = str(unit or "").strip()
    if not text:
        return ""

    replacements = {
        "°C": "degC",
        "°F": "degF",
        "°/s": "deg/s",
        "√Hz": "sqrt(Hz)",
        "√s": "sqrt(s)",
        "²": "^2",
        "³": "^3",
        "·": "*",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


__all__ = [
    "humanize_identifier",
    "humanize_unit",
    "machine_unit",
]
