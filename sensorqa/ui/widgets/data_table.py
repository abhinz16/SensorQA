from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from sensorqa.ui import humanize_identifier


class DataFrameTable(QTableWidget):
    """Read-only table used for dataset and result previews.

    Args:
        parent: Optional Qt parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the read-only table.

        Args:
            parent: Optional Qt parent widget.

        Returns:
            None.
        """

        super().__init__(parent)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)

    def set_dataframe(
        self,
        data: pd.DataFrame | None,
        max_rows: int = 30,
        *,
        humanize_headers: bool = False,
    ) -> None:
        """Populate the table from a pandas DataFrame.

        Args:
            data: DataFrame to show. ``None`` clears the table.
            max_rows: Maximum number of rows shown in the preview.
            humanize_headers: If ``True``, normalized SensorQA column names
                are converted to readable labels such as ``Accel X``.

        Returns:
            None.
        """

        self.clear()
        if data is None:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        frame = data.head(max_rows)
        self.setColumnCount(len(frame.columns))
        self.setRowCount(len(frame))

        labels = [
            humanize_identifier(column) if humanize_headers else str(column)
            for column in frame.columns
        ]
        self.setHorizontalHeaderLabels(labels)

        for row_index, (_, row) in enumerate(frame.iterrows()):
            for column_index, value in enumerate(row.tolist()):
                text = "" if pd.isna(value) else str(value)
                self.setItem(row_index, column_index, QTableWidgetItem(text))
