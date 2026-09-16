#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SensorQA desktop application visual design system.

The module is intentionally safe to import without PySide6 installed.  Most of
SensorQA's backend and test suite should not require a GUI dependency.  Qt is
imported only inside :func:`apply_theme` when an actual desktop application is
running.

The design aims for a technical, futuristic workstation feel without relying
on decorative effects that reduce readability.  The palette, spacing, radii,
typography, chart colors, and Qt Style Sheet live in one place so every page of
the future application presents a coherent interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ============================================================================
# Design tokens
# ============================================================================


@dataclass(frozen=True)
class ColorTokens:
    """Semantic color palette for the SensorQA desktop application."""

    # Application surfaces
    background: str = "#070B12"
    background_elevated: str = "#0A101A"
    sidebar: str = "#090E17"
    surface: str = "#0E1623"
    surface_alt: str = "#121C2B"
    surface_hover: str = "#172336"
    surface_pressed: str = "#1B2A40"
    overlay: str = "#0B1320"

    # Lines / separators
    border: str = "#243249"
    border_soft: str = "#1A273A"
    border_strong: str = "#344764"

    # Text hierarchy
    text_primary: str = "#F5F8FC"
    text_secondary: str = "#A8B4C7"
    text_muted: str = "#8795AA"
    text_disabled: str = "#4E5A6C"

    # Brand / action accents
    accent: str = "#22D3EE"
    accent_hover: str = "#67E8F9"
    accent_pressed: str = "#0891B2"
    accent_soft: str = "#102F3D"
    accent_border: str = "#155E75"

    secondary: str = "#818CF8"
    secondary_hover: str = "#A5B4FC"
    secondary_soft: str = "#1E224A"

    # Semantic states
    success: str = "#34D399"
    success_soft: str = "#102F28"
    success_border: str = "#1D6A55"

    warning: str = "#FBBF24"
    warning_soft: str = "#33270D"
    warning_border: str = "#6E5314"

    danger: str = "#FB7185"
    danger_soft: str = "#381722"
    danger_border: str = "#7A2C3C"

    info: str = "#60A5FA"
    info_soft: str = "#142844"
    info_border: str = "#285A91"

    neutral: str = "#94A3B8"

    # Plot palette
    plot_1: str = "#22D3EE"
    plot_2: str = "#818CF8"
    plot_3: str = "#34D399"
    plot_4: str = "#FBBF24"
    plot_5: str = "#FB7185"
    plot_6: str = "#C084FC"
    plot_grid: str = "#243249"
    plot_axis: str = "#8291A6"


@dataclass(frozen=True)
class SpacingTokens:
    """Spacing scale in device-independent pixels."""

    xxs: int = 4
    xs: int = 8
    sm: int = 12
    md: int = 16
    lg: int = 24
    xl: int = 32
    xxl: int = 48
    xxxl: int = 64


@dataclass(frozen=True)
class RadiusTokens:
    """Corner-radius scale."""

    small: int = 6
    medium: int = 10
    large: int = 14
    xlarge: int = 18
    pill: int = 999


@dataclass(frozen=True)
class TypographyTokens:
    """Typography defaults used by the Qt UI and chart layer."""

    # Cross-platform UI font fallback list. Qt accepts comma-separated
    # alternatives in a style sheet.
    family: str = '"Inter", "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial'
    mono_family: str = '"JetBrains Mono", "SFMono-Regular", Consolas, monospace'

    size_caption: int = 11
    size_body: int = 13
    size_body_large: int = 14
    size_subtitle: int = 16
    size_title: int = 20
    size_heading: int = 26
    size_display: int = 34
    size_metric: int = 38

    weight_regular: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700


@dataclass(frozen=True)
class ThemeTokens:
    """Complete immutable SensorQA theme token collection."""

    colors: ColorTokens = ColorTokens()
    spacing: SpacingTokens = SpacingTokens()
    radii: RadiusTokens = RadiusTokens()
    typography: TypographyTokens = TypographyTokens()


THEME = ThemeTokens()


# ============================================================================
# Semantic helpers
# ============================================================================


STATUS_COLORS: dict[str, str] = {
    "pass": THEME.colors.success,
    "passed": THEME.colors.success,
    "success": THEME.colors.success,
    "fail": THEME.colors.danger,
    "failed": THEME.colors.danger,
    "error": THEME.colors.danger,
    "warning": THEME.colors.warning,
    "not_evaluated": THEME.colors.neutral,
    "not evaluated": THEME.colors.neutral,
    "skipped": THEME.colors.neutral,
    "running": THEME.colors.accent,
    "info": THEME.colors.info,
}

STATUS_BACKGROUND_COLORS: dict[str, str] = {
    "pass": THEME.colors.success_soft,
    "passed": THEME.colors.success_soft,
    "success": THEME.colors.success_soft,
    "fail": THEME.colors.danger_soft,
    "failed": THEME.colors.danger_soft,
    "error": THEME.colors.danger_soft,
    "warning": THEME.colors.warning_soft,
    "not_evaluated": THEME.colors.surface_alt,
    "not evaluated": THEME.colors.surface_alt,
    "skipped": THEME.colors.surface_alt,
    "running": THEME.colors.accent_soft,
    "info": THEME.colors.info_soft,
}


def status_color(status: Any) -> str:
    """Return the semantic foreground color for a status-like value."""

    value = getattr(status, "value", status)
    key = str(value).strip().lower()

    return STATUS_COLORS.get(
        key,
        THEME.colors.neutral,
    )


def status_background(status: Any) -> str:
    """Return the semantic background color for a status-like value."""

    value = getattr(status, "value", status)
    key = str(value).strip().lower()

    return STATUS_BACKGROUND_COLORS.get(
        key,
        THEME.colors.surface_alt,
    )


def chart_palette() -> tuple[str, ...]:
    """Return the standard ordered SensorQA chart palette."""

    c = THEME.colors

    return (
        c.plot_1,
        c.plot_2,
        c.plot_3,
        c.plot_4,
        c.plot_5,
        c.plot_6,
    )


def matplotlib_rc() -> dict[str, Any]:
    """Return matplotlib rcParams compatible with the SensorQA UI theme.

    This function does not import matplotlib.  The plotting layer can pass the
    returned dictionary to ``matplotlib.rc_context`` or ``rcParams.update``.
    """

    c = THEME.colors

    return {
        "figure.facecolor": c.surface,
        "axes.facecolor": c.surface,
        "axes.edgecolor": c.border,
        "axes.labelcolor": c.text_secondary,
        "axes.titlecolor": c.text_primary,
        "xtick.color": c.text_muted,
        "ytick.color": c.text_muted,
        "grid.color": c.plot_grid,
        "grid.alpha": 0.65,
        "text.color": c.text_primary,
        "legend.facecolor": c.surface_alt,
        "legend.edgecolor": c.border,
        "legend.labelcolor": c.text_secondary,
        "savefig.facecolor": c.surface,
        "savefig.edgecolor": c.surface,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "lines.linewidth": 1.8,
    }


def apply_matplotlib_theme() -> None:
    """Apply the SensorQA chart theme to matplotlib globally.

    Matplotlib is imported lazily so the rest of SensorQA can be used without
    the plotting dependency. The desktop plot widgets call this helper before
    creating figures so embedded charts match the Qt application theme.
    """

    try:
        import matplotlib as mpl
    except ImportError as exc:  # pragma: no cover - depends on optional GUI install
        raise RuntimeError(
            "Matplotlib is required for SensorQA desktop plots. "
            "Install matplotlib before using plot widgets."
        ) from exc

    mpl.rcParams.update(
        matplotlib_rc()
    )


# ============================================================================
# Qt style sheet
# ============================================================================


def build_stylesheet() -> str:
    """Build the complete SensorQA Qt Style Sheet.

    Widgets can opt into specialized styles by assigning either an object name
    or a dynamic ``role`` property.  This makes the style reusable across all
    application pages without requiring bespoke CSS per widget.

    Common roles
    ------------
    ``role="card"``
        Elevated dashboard/result cards.

    ``role="mutedCard"``
        Lower-emphasis information containers.

    ``role="pageTitle"`` / ``role="pageSubtitle"``
        Page heading hierarchy.

    ``role="metricValue"`` / ``role="metricLabel"``
        KPI cards.

    ``role="primaryButton"`` / ``role="secondaryButton"`` /
    ``role="ghostButton"`` / ``role="dangerButton"``
        Action hierarchy.

    ``role="navButton"``
        Sidebar navigation item; set ``active=true`` for the current page.

    ``status="pass|fail|warning|not_evaluated|running"``
        Semantic status badge.
    """

    c = THEME.colors
    r = THEME.radii
    t = THEME.typography

    return f"""
/* ========================================================================
   SENSORQA - GLOBAL
   ======================================================================== */

QWidget {{
    background-color: {c.background};
    color: {c.text_primary};
    font-family: {t.family};
    font-size: {t.size_body}px;
    selection-background-color: {c.accent_pressed};
    selection-color: {c.text_primary};
}}

QMainWindow,
QDialog {{
    background-color: {c.background};
}}

QWidget:disabled {{
    color: {c.text_disabled};
}}

/* ========================================================================
   TEXT
   ======================================================================== */

QLabel {{
    background: transparent;
    color: {c.text_secondary};
}}

QLabel[role="pageTitle"] {{
    color: {c.text_primary};
    font-size: {t.size_heading}px;
    font-weight: {t.weight_bold};
}}

QLabel[role="pageSubtitle"] {{
    color: {c.text_muted};
    font-size: {t.size_body_large}px;
}}

QLabel[role="sectionTitle"] {{
    color: {c.text_primary};
    font-size: {t.size_subtitle}px;
    font-weight: {t.weight_semibold};
}}

QLabel[role="eyebrow"] {{
    color: {c.accent};
    font-size: {t.size_caption}px;
    font-weight: {t.weight_semibold};
}}

QLabel[role="metricValue"] {{
    color: {c.text_primary};
    font-size: {t.size_metric}px;
    font-weight: {t.weight_bold};
}}

QLabel[role="metricLabel"] {{
    color: {c.text_muted};
    font-size: {t.size_caption}px;
    font-weight: {t.weight_semibold};
}}

QLabel[role="muted"] {{
    color: {c.text_muted};
}}

QLabel[role="topbarContext"] {{
    color: {c.text_secondary};
    font-size: 13px;
    font-weight: {t.weight_medium};
}}

QLabel[role="accent"] {{
    color: {c.accent};
}}

QLabel[role="mono"] {{
    font-family: {t.mono_family};
    color: {c.text_secondary};
}}

/* ========================================================================
   SURFACES / CARDS
   ======================================================================== */

QFrame[role="card"],
QWidget[role="card"] {{
    background-color: {c.surface};
    border: 1px solid {c.border_soft};
    border-radius: {r.large}px;
}}

QFrame[role="card"][interactive="true"]:hover,
QWidget[role="card"][interactive="true"]:hover {{
    background-color: {c.surface_hover};
    border-color: {c.border_strong};
}}

QFrame[role="accentCard"],
QWidget[role="accentCard"] {{
    background-color: {c.accent_soft};
    border: 1px solid {c.accent_border};
    border-radius: {r.large}px;
}}

QFrame[role="mutedCard"],
QWidget[role="mutedCard"] {{
    background-color: {c.background_elevated};
    border: 1px solid {c.border_soft};
    border-radius: {r.medium}px;
}}

QFrame[role="dropZone"],
QWidget[role="dropZone"] {{
    background-color: {c.background_elevated};
    border: 2px dashed {c.border_strong};
    border-radius: {r.xlarge}px;
}}

QFrame[role="dropZone"][dragActive="true"],
QWidget[role="dropZone"][dragActive="true"] {{
    background-color: {c.accent_soft};
    border-color: {c.accent};
}}

/* ========================================================================
   BUTTONS
   ======================================================================== */

QPushButton {{
    background-color: {c.surface_alt};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {r.medium}px;
    padding: 9px 16px;
    font-weight: {t.weight_semibold};
}}

QPushButton:hover {{
    background-color: {c.surface_hover};
    border-color: {c.border_strong};
}}

QPushButton:pressed {{
    background-color: {c.surface_pressed};
}}

QPushButton:disabled {{
    background-color: {c.background_elevated};
    border-color: {c.border_soft};
    color: {c.text_disabled};
}}

QPushButton[role="primaryButton"] {{
    background-color: {c.accent};
    color: {c.background};
    border: 1px solid {c.accent};
}}

QPushButton[role="primaryButton"]:hover {{
    background-color: {c.accent_hover};
    border-color: {c.accent_hover};
}}

QPushButton[role="primaryButton"]:pressed {{
    background-color: {c.accent_pressed};
    border-color: {c.accent_pressed};
    color: {c.text_primary};
}}

QPushButton[role="secondaryButton"] {{
    background-color: {c.secondary_soft};
    color: {c.secondary_hover};
    border: 1px solid {c.secondary};
}}

QPushButton[role="secondaryButton"]:hover {{
    background-color: {c.secondary};
    color: {c.background};
}}

QPushButton[role="ghostButton"] {{
    background: transparent;
    color: {c.text_secondary};
    border: 1px solid transparent;
}}

QPushButton[role="ghostButton"]:hover {{
    background-color: {c.surface_hover};
    color: {c.text_primary};
    border-color: {c.border_soft};
}}

QPushButton[role="dangerButton"] {{
    background-color: {c.danger_soft};
    color: {c.danger};
    border: 1px solid {c.danger_border};
}}

QPushButton[role="dangerButton"]:hover {{
    background-color: {c.danger};
    color: {c.background};
}}


/* ========================================================================
   INLINE MESSAGES
   ======================================================================== */

QFrame[role="messageBanner"] {{
    background-color: {c.info_soft};
    border: 1px solid {c.info_border};
    border-radius: {r.medium}px;
}}

QFrame[role="messageBanner"][state="error"] {{
    background-color: {c.danger_soft};
    border-color: {c.danger_border};
}}

QFrame[role="messageBanner"][state="warning"] {{
    background-color: {c.warning_soft};
    border-color: {c.warning_border};
}}

QFrame[role="messageBanner"][state="success"] {{
    background-color: {c.success_soft};
    border-color: {c.success_border};
}}

QLabel[role="messageTitle"] {{
    color: {c.text_primary};
    font-size: 14px;
    font-weight: {t.weight_bold};
}}

QLabel[role="messageBody"] {{
    color: {c.text_secondary};
    font-size: 13px;
}}

QFrame[role="messageBanner"][state="error"] QLabel[role="messageTitle"] {{
    color: {c.danger};
}}

QFrame[role="messageBanner"][state="warning"] QLabel[role="messageTitle"] {{
    color: {c.warning};
}}

QFrame[role="messageBanner"][state="success"] QLabel[role="messageTitle"] {{
    color: {c.success};
}}

/* ========================================================================
   SIDEBAR NAVIGATION
   ======================================================================== */

QWidget#Sidebar {{
    background-color: {c.sidebar};
    border-right: 1px solid {c.border_soft};
}}

QPushButton[role="navButton"] {{
    background: transparent;
    color: {c.text_secondary};
    border: 1px solid transparent;
    border-radius: {r.medium}px;
    padding: 11px 13px;
    text-align: left;
    font-size: 14px;
    font-weight: {t.weight_semibold};
}}

QPushButton[role="navButton"]:hover {{
    background-color: {c.surface_alt};
    color: {c.text_primary};
}}

QPushButton[role="navButton"][active="true"] {{
    background-color: {c.accent_soft};
    color: {c.accent_hover};
    border-color: {c.accent_border};
    font-weight: {t.weight_bold};
}}

/* ========================================================================
   INPUTS
   ======================================================================== */

QLineEdit,
QTextEdit,
QPlainTextEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox,
QDateEdit,
QDateTimeEdit {{
    background-color: {c.background_elevated};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {r.medium}px;
    padding: 8px 10px;
}}

QLineEdit:hover,
QTextEdit:hover,
QPlainTextEdit:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QComboBox:hover {{
    border-color: {c.border_strong};
}}

QLineEdit:focus,
QTextEdit:focus,
QPlainTextEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {{
    border: 1px solid {c.accent};
    background-color: {c.surface};
}}

QLineEdit[error="true"],
QComboBox[error="true"],
QSpinBox[error="true"],
QDoubleSpinBox[error="true"] {{
    border-color: {c.danger};
}}

QComboBox::drop-down {{
    border: none;
    width: 28px;
}}

QComboBox QAbstractItemView {{
    background-color: {c.surface_alt};
    color: {c.text_primary};
    border: 1px solid {c.border};
    selection-background-color: {c.accent_soft};
    selection-color: {c.accent_hover};
    outline: 0;
    padding: 4px;
}}

/* ========================================================================
   CHECKABLE CONTROLS
   ======================================================================== */

QCheckBox,
QRadioButton {{
    color: {c.text_secondary};
    spacing: 8px;
    background: transparent;
}}

QCheckBox:hover,
QRadioButton:hover {{
    color: {c.text_primary};
}}

QCheckBox::indicator,
QRadioButton::indicator {{
    width: 17px;
    height: 17px;
}}

QCheckBox::indicator:unchecked {{
    background-color: {c.background_elevated};
    border: 1px solid {c.border_strong};
    border-radius: 4px;
}}

QCheckBox::indicator:checked {{
    background-color: {c.accent};
    border: 1px solid {c.accent};
    border-radius: 4px;
}}

QRadioButton::indicator:unchecked {{
    background-color: {c.background_elevated};
    border: 1px solid {c.border_strong};
    border-radius: 9px;
}}

QRadioButton::indicator:checked {{
    background-color: {c.accent};
    border: 4px solid {c.accent_soft};
    border-radius: 9px;
}}

/* ========================================================================
   STATUS BADGES
   ======================================================================== */

QLabel[role="statusBadge"] {{
    border-radius: {r.small}px;
    padding: 4px 9px;
    font-size: {t.size_caption}px;
    font-weight: {t.weight_bold};
}}

QLabel[role="statusBadge"][status="pass"],
QLabel[role="statusBadge"][status="success"] {{
    background-color: {c.success_soft};
    color: {c.success};
    border: 1px solid {c.success_border};
}}

QLabel[role="statusBadge"][status="fail"],
QLabel[role="statusBadge"][status="error"] {{
    background-color: {c.danger_soft};
    color: {c.danger};
    border: 1px solid {c.danger_border};
}}

QLabel[role="statusBadge"][status="warning"] {{
    background-color: {c.warning_soft};
    color: {c.warning};
    border: 1px solid {c.warning_border};
}}

QLabel[role="statusBadge"][status="not_evaluated"],
QLabel[role="statusBadge"][status="skipped"] {{
    background-color: {c.surface_alt};
    color: {c.neutral};
    border: 1px solid {c.border};
}}

QLabel[role="statusBadge"][status="running"] {{
    background-color: {c.accent_soft};
    color: {c.accent};
    border: 1px solid {c.accent_border};
}}

/* ========================================================================
   TABLES
   ======================================================================== */

QTableView,
QTableWidget,
QTreeView,
QListView {{
    background-color: {c.surface};
    alternate-background-color: {c.background_elevated};
    color: {c.text_secondary};
    border: 1px solid {c.border_soft};
    border-radius: {r.medium}px;
    gridline-color: {c.border_soft};
    outline: 0;
    selection-background-color: {c.accent_soft};
    selection-color: {c.text_primary};
}}

QTableView::item,
QTableWidget::item,
QTreeView::item,
QListView::item {{
    padding: 7px;
    border: none;
}}

QTableView::item:hover,
QTableWidget::item:hover,
QTreeView::item:hover,
QListView::item:hover {{
    background-color: {c.surface_hover};
}}

QHeaderView::section {{
    background-color: {c.background_elevated};
    color: {c.text_muted};
    border: none;
    border-bottom: 1px solid {c.border};
    padding: 9px;
    font-weight: {t.weight_semibold};
}}

QTableCornerButton::section {{
    background-color: {c.background_elevated};
    border: none;
    border-bottom: 1px solid {c.border};
}}

/* ========================================================================
   TABS
   ======================================================================== */

QTabWidget::pane {{
    background-color: {c.surface};
    border: 1px solid {c.border_soft};
    border-radius: {r.medium}px;
}}

QTabBar::tab {{
    background: transparent;
    color: {c.text_muted};
    border: none;
    padding: 9px 14px;
    margin-right: 4px;
}}

QTabBar::tab:hover {{
    color: {c.text_primary};
}}

QTabBar::tab:selected {{
    color: {c.accent};
    border-bottom: 2px solid {c.accent};
}}

/* ========================================================================
   PROGRESS
   ======================================================================== */

QProgressBar {{
    background-color: {c.background_elevated};
    border: 1px solid {c.border_soft};
    border-radius: {r.small}px;
    color: {c.text_secondary};
    text-align: center;
    min-height: 8px;
}}

QProgressBar::chunk {{
    background-color: {c.accent};
    border-radius: {r.small}px;
}}

/* ========================================================================
   MENUS / TOOLTIPS
   ======================================================================== */

QMenu {{
    background-color: {c.surface_alt};
    color: {c.text_secondary};
    border: 1px solid {c.border};
    border-radius: {r.medium}px;
    padding: 6px;
}}

QMenu::item {{
    padding: 8px 28px 8px 12px;
    border-radius: {r.small}px;
}}

QMenu::item:selected {{
    background-color: {c.accent_soft};
    color: {c.accent_hover};
}}

QMenu::separator {{
    height: 1px;
    background-color: {c.border_soft};
    margin: 5px 8px;
}}

QToolTip {{
    background-color: {c.surface_alt};
    color: {c.text_primary};
    border: 1px solid {c.border_strong};
    border-radius: {r.small}px;
    padding: 6px 8px;
}}

/* ========================================================================
   SCROLLBARS
   ======================================================================== */

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background-color: {c.border_strong};
    min-height: 28px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {c.text_muted};
}}

QScrollBar:add-line:vertical,
QScrollBar:sub-line:vertical,
QScrollBar:add-page:vertical,
QScrollBar:sub-page:vertical {{
    background: none;
    height: 0px;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    background-color: {c.border_strong};
    min-width: 28px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {c.text_muted};
}}

QScrollBar:add-line:horizontal,
QScrollBar:sub-line:horizontal,
QScrollBar:add-page:horizontal,
QScrollBar:sub-page:horizontal {{
    background: none;
    width: 0px;
}}

/* ========================================================================
   SPLITTER / GROUP BOX
   ======================================================================== */

QSplitter::handle {{
    background-color: {c.border_soft};
}}

QSplitter::handle:hover {{
    background-color: {c.accent_border};
}}

QGroupBox {{
    background-color: {c.surface};
    border: 1px solid {c.border_soft};
    border-radius: {r.large}px;
    margin-top: 12px;
    padding: 14px;
    font-weight: {t.weight_semibold};
    color: {c.text_primary};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {c.text_secondary};
}}

/* ========================================================================
   SPECIAL APPLICATION ELEMENTS
   ======================================================================== */

QFrame#TopBar {{
    background-color: {c.background_elevated};
    border-bottom: 1px solid {c.border_soft};
}}

QLabel#BrandMark {{
    color: {c.accent};
    font-size: {t.size_title}px;
    font-weight: {t.weight_bold};
}}

QLabel#BrandName {{
    color: {c.text_primary};
    font-size: {t.size_title}px;
    font-weight: {t.weight_bold};
}}

QFrame[role="step"] {{
    background: transparent;
    border: none;
}}

QLabel[role="stepCircle"] {{
    background-color: {c.surface_alt};
    color: {c.text_muted};
    border: 1px solid {c.border};
    border-radius: 12px;
    min-width: 24px;
    max-width: 24px;
    min-height: 24px;
    max-height: 24px;
    font-weight: {t.weight_bold};
}}

QLabel[role="stepCircle"][state="active"] {{
    background-color: {c.accent};
    color: {c.background};
    border-color: {c.accent};
}}

QLabel[role="stepCircle"][state="complete"] {{
    background-color: {c.success};
    color: {c.background};
    border-color: {c.success};
}}

QFrame[role="divider"] {{
    background-color: {c.border_soft};
    border: none;
    min-height: 1px;
    max-height: 1px;
}}

QFrame[role="verticalDivider"] {{
    background-color: {c.border_soft};
    border: none;
    min-width: 1px;
    max-width: 1px;
}}
"""


# Materialize once for callers that prefer a constant.
SENSORQA_STYLESHEET = build_stylesheet()


# ============================================================================
# Runtime application helpers
# ============================================================================


def apply_theme(application: Any) -> None:
    """Apply the SensorQA visual theme to a ``QApplication`` instance.

    PySide6 is imported lazily so backend-only installations can still import
    ``sensorqa.ui.theme`` without installing Qt.
    """

    try:
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - depends on GUI install
        raise RuntimeError(
            "PySide6 is required to launch the SensorQA desktop UI. "
            "Install the GUI dependencies before calling apply_theme()."
        ) from exc

    if not isinstance(application, QApplication):
        raise TypeError(
            "application must be a PySide6.QtWidgets.QApplication."
        )

    application.setStyle("Fusion")
    application.setStyleSheet(SENSORQA_STYLESHEET)

    # A palette is set in addition to QSS so native dialogs and widgets that
    # do not fully honor style sheets remain consistent with the dark theme.
    palette = QPalette()

    palette.setColor(
        QPalette.ColorRole.Window,
        QColor(THEME.colors.background),
    )
    palette.setColor(
        QPalette.ColorRole.WindowText,
        QColor(THEME.colors.text_primary),
    )
    palette.setColor(
        QPalette.ColorRole.Base,
        QColor(THEME.colors.background_elevated),
    )
    palette.setColor(
        QPalette.ColorRole.AlternateBase,
        QColor(THEME.colors.surface),
    )
    palette.setColor(
        QPalette.ColorRole.ToolTipBase,
        QColor(THEME.colors.surface_alt),
    )
    palette.setColor(
        QPalette.ColorRole.ToolTipText,
        QColor(THEME.colors.text_primary),
    )
    palette.setColor(
        QPalette.ColorRole.Text,
        QColor(THEME.colors.text_primary),
    )
    palette.setColor(
        QPalette.ColorRole.Button,
        QColor(THEME.colors.surface_alt),
    )
    palette.setColor(
        QPalette.ColorRole.ButtonText,
        QColor(THEME.colors.text_primary),
    )
    palette.setColor(
        QPalette.ColorRole.BrightText,
        QColor(THEME.colors.danger),
    )
    palette.setColor(
        QPalette.ColorRole.Highlight,
        QColor(THEME.colors.accent_pressed),
    )
    palette.setColor(
        QPalette.ColorRole.HighlightedText,
        QColor(THEME.colors.text_primary),
    )
    palette.setColor(
        QPalette.ColorRole.PlaceholderText,
        QColor(THEME.colors.text_muted),
    )

    application.setPalette(palette)


def repolish(widget: Any) -> None:
    """Reapply QSS after changing a dynamic Qt property.

    Qt style sheets do not always refresh immediately after a property such as
    ``active`` or ``status`` changes.  UI components can call this helper after
    updating one of those properties.
    """

    style = widget.style()

    if style is None:
        return

    style.unpolish(widget)
    style.polish(widget)
    widget.update()


__all__ = [
    "ColorTokens",
    "SpacingTokens",
    "RadiusTokens",
    "TypographyTokens",
    "ThemeTokens",
    "THEME",
    "STATUS_COLORS",
    "STATUS_BACKGROUND_COLORS",
    "SENSORQA_STYLESHEET",
    "status_color",
    "status_background",
    "chart_palette",
    "matplotlib_rc",
    "apply_matplotlib_theme",
    "build_stylesheet",
    "apply_theme",
    "repolish",
]
