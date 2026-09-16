#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO

import pandas as pd


@dataclass
class DatasetColumnSummary:
    """
    Basic summary of one column in an uploaded dataset.
    """

    name: str
    dtype: str

    non_null_count: int
    null_count: int

    unique_count: int

    is_completely_empty: bool
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class CSVMetadata:
    """
    Metadata describing a successfully loaded CSV dataset.
    """

    file_name: str | None
    file_path: str | None

    delimiter: str

    row_count: int
    column_count: int

    columns: list[str]

    duplicate_columns: list[str] = field(default_factory=list)
    empty_columns: list[str] = field(default_factory=list)

    column_summaries: list[DatasetColumnSummary] = field(
        default_factory=list
    )


@dataclass
class CSVLoadResult:
    """
    Complete result returned by CSVLoader.

    A load operation may succeed while still generating warnings.
    """

    success: bool

    data: pd.DataFrame | None = None
    metadata: CSVMetadata | None = None

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class CSVLoader:
    """
    Loads CSV sensor datasets and performs basic structural inspection.

    This class only handles file-level and table-level concerns.

    It does NOT perform detailed sensor-data quality analysis.
    """

    SUPPORTED_ENCODINGS = (
        "utf-8-sig",
        "utf-8",
        "latin-1",
    )

    DEFAULT_PREVIEW_VALUES = 5

    def load(
        self,
        source: str | Path | IO,
        delimiter: str | None = None,
    ) -> CSVLoadResult:
        """Load a CSV file from a path or file-like object.
        
        Parameters
        ----------
        source:
            File path or readable file-like object.
        
        delimiter:
            Optional delimiter override.
            If omitted, SensorQA attempts to detect it.
        
        Args:
            source: Input source or source path.
            delimiter: Value for `delimiter`.
        
        Returns:
            CSVLoadResult returned by the function.
        """

        warnings: list[str] = []
        errors: list[str] = []

        file_name: str | None = None
        file_path: str | None = None

        # Resolve source information

        if isinstance(source, (str, Path)):

            path = Path(source)

            file_name = path.name
            file_path = str(path.resolve())

            if not path.exists():
                return CSVLoadResult(
                    success=False,
                    errors=[
                        f"CSV file does not exist: {path}"
                    ],
                )

            if not path.is_file():
                return CSVLoadResult(
                    success=False,
                    errors=[
                        f"CSV path is not a file: {path}"
                    ],
                )

            if path.stat().st_size == 0:
                return CSVLoadResult(
                    success=False,
                    errors=[
                        "CSV file is empty."
                    ],
                )

        else:

            file_name = getattr(
                source,
                "name",
                None,
            )

        # Detect delimiter if user did not specify one

        if delimiter is None:

            try:
                delimiter = self._detect_delimiter(
                    source
                )

            except Exception as exc:

                warnings.append(
                    "Could not confidently detect the CSV delimiter. "
                    f"Falling back to comma. "
                    f"Reason: {type(exc).__name__}: {exc}"
                )

                delimiter = ","

        # Read dataset

        dataframe: pd.DataFrame | None = None
        encoding_used: str | None = None

        for encoding in self.SUPPORTED_ENCODINGS:

            try:

                self._rewind_if_possible(
                    source
                )

                dataframe = pd.read_csv(
                    source,
                    sep=delimiter,
                    encoding=encoding,
                    low_memory=False,
                )

                encoding_used = encoding

                break

            except UnicodeDecodeError:
                continue

            except pd.errors.EmptyDataError:

                return CSVLoadResult(
                    success=False,
                    errors=[
                        "CSV file does not contain readable tabular data."
                    ],
                )

            except pd.errors.ParserError as exc:

                return CSVLoadResult(
                    success=False,
                    errors=[
                        f"CSV parsing failed: {exc}"
                    ],
                )

            except Exception as exc:

                errors.append(
                    f"CSV loading failed with encoding "
                    f"'{encoding}': "
                    f"{type(exc).__name__}: {exc}"
                )

        if dataframe is None:

            return CSVLoadResult(
                success=False,
                warnings=warnings,
                errors=errors or [
                    "SensorQA could not read the CSV file "
                    "using the supported encodings."
                ],
            )

        # Basic table validation

        if len(dataframe.columns) == 0:

            return CSVLoadResult(
                success=False,
                warnings=warnings,
                errors=[
                    "CSV dataset contains no columns."
                ],
            )

        if dataframe.empty:

            warnings.append(
                "CSV contains column headers but no data rows."
            )

        # Preserve original headers before normalization

        original_columns = [
            str(column)
            for column in dataframe.columns
        ]

        duplicate_columns = self._detect_duplicate_headers(
            original_columns
        )

        if duplicate_columns:

            warnings.append(
                "Duplicate column names were detected in the "
                "original CSV header: "
                + ", ".join(
                    sorted(duplicate_columns)
                )
            )

        # Normalize header whitespace

        cleaned_columns = self._clean_column_names(
            original_columns
        )

        dataframe.columns = cleaned_columns

        if cleaned_columns != original_columns:

            warnings.append(
                "Leading or trailing whitespace was removed "
                "from one or more column names."
            )

        # Detect duplicate names after cleaning

        cleaned_duplicates = self._detect_duplicate_headers(
            cleaned_columns
        )

        if cleaned_duplicates:

            warnings.append(
                "Duplicate column names exist after header cleanup: "
                + ", ".join(
                    sorted(cleaned_duplicates)
                )
            )

        all_duplicate_columns = sorted(
            set(duplicate_columns)
            | set(cleaned_duplicates)
        )

        # Completely empty columns

        empty_columns = [
            str(column)
            for column in dataframe.columns
            if dataframe[column].isna().all()
        ]

        if empty_columns:

            warnings.append(
                "Completely empty columns detected: "
                + ", ".join(
                    empty_columns
                )
            )

        # Build column summaries

        column_summaries = self._summarize_columns(
            dataframe
        )

        metadata = CSVMetadata(
            file_name=file_name,
            file_path=file_path,
            delimiter=delimiter,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            columns=[
                str(column)
                for column in dataframe.columns
            ],
            duplicate_columns=all_duplicate_columns,
            empty_columns=empty_columns,
            column_summaries=column_summaries,
        )

        if encoding_used not in {
            "utf-8",
            "utf-8-sig",
        }:

            warnings.append(
                f"CSV was loaded using encoding "
                f"'{encoding_used}'."
            )

        return CSVLoadResult(
            success=True,
            data=dataframe,
            metadata=metadata,
            warnings=warnings,
            errors=[],
        )

    def preview(
        self,
        result: CSVLoadResult,
        rows: int = 10,
    ) -> pd.DataFrame:
        """Return the first rows of a successfully loaded dataset.
        
        Intended for the future upload / column-mapping interface.
        
        Args:
            result: Result object to process.
            rows: Value for `rows`.
        
        Returns:
            DataFrame containing the requested data.
        """

        if not result.success or result.data is None:
            raise ValueError(
                "Cannot preview an unsuccessfully loaded dataset."
            )

        if rows <= 0:
            raise ValueError(
                "Preview row count must be greater than zero."
            )

        return result.data.head(rows).copy()

    # Delimiter handling

    def _detect_delimiter(
        self,
        source: str | Path | IO,
    ) -> str:
        """Attempt to detect the delimiter from a small sample.
        
        Common delimiters considered:
            comma
            semicolon
            tab
            pipe
        
        Args:
            source: Input source or source path.
        
        Returns:
            String representation.
        """

        sample = self._read_text_sample(
            source,
            size=8192,
        )

        if not sample.strip():
            raise ValueError(
                "File sample is empty."
            )

        sniffer = csv.Sniffer()

        dialect = sniffer.sniff(
            sample,
            delimiters=",;\t|",
        )

        return dialect.delimiter

    def _read_text_sample(
        self,
        source: str | Path | IO,
        size: int,
    ) -> str:
        """Read a small text sample without permanently consuming
        a file-like object.
        
        Args:
            source: Input source or source path.
            size: Value for `size`.
        
        Returns:
            String representation.
        """

        if isinstance(source, (str, Path)):

            path = Path(source)

            for encoding in self.SUPPORTED_ENCODINGS:

                try:

                    with path.open(
                        "r",
                        encoding=encoding,
                    ) as handle:

                        return handle.read(
                            size
                        )

                except UnicodeDecodeError:
                    continue

            raise UnicodeDecodeError(
                "unknown",
                b"",
                0,
                1,
                "Unable to decode file sample.",
            )

        # File-like object

        original_position = None

        try:
            original_position = source.tell()
        except Exception:
            pass

        content = source.read(size)

        if isinstance(content, bytes):

            decoded = None

            for encoding in self.SUPPORTED_ENCODINGS:

                try:
                    decoded = content.decode(
                        encoding
                    )
                    break
                except UnicodeDecodeError:
                    continue

            if decoded is None:
                raise ValueError(
                    "Unable to decode uploaded file sample."
                )

            content = decoded

        if original_position is not None:

            try:
                source.seek(
                    original_position
                )
            except Exception:
                pass

        return str(content)

    # Column inspection

    @staticmethod
    def _clean_column_names(
        columns: list[str],
    ) -> list[str]:
        """Remove leading and trailing whitespace from headers.
        
        Internal spaces are intentionally preserved.
        
        Args:
            columns: Value for `columns`.
        
        Returns:
            List of result values.
        """

        return [
            column.strip()
            for column in columns
        ]

    @staticmethod
    def _detect_duplicate_headers(
        columns: list[str],
    ) -> list[str]:
        """Return duplicate header names.
        
        Args:
            columns: Value for `columns`.
        
        Returns:
            List of result values.
        """

        seen: set[str] = set()
        duplicates: set[str] = set()

        for column in columns:

            if column in seen:
                duplicates.add(
                    column
                )

            seen.add(
                column
            )

        return sorted(
            duplicates
        )

    def _summarize_columns(
        self,
        data: pd.DataFrame,
    ) -> list[DatasetColumnSummary]:
        """Generate lightweight summaries for the upload interface.
        
        Args:
            data: Input data to process.
        
        Returns:
            List of result values.
        """

        summaries: list[DatasetColumnSummary] = []

        for column in data.columns:

            series = data[column]

            non_null = series.dropna()

            sample_values = (
                non_null
                .drop_duplicates()
                .head(
                    self.DEFAULT_PREVIEW_VALUES
                )
                .tolist()
            )

            summaries.append(
                DatasetColumnSummary(
                    name=str(column),
                    dtype=str(series.dtype),
                    non_null_count=int(
                        series.notna().sum()
                    ),
                    null_count=int(
                        series.isna().sum()
                    ),
                    unique_count=int(
                        series.nunique(
                            dropna=True
                        )
                    ),
                    is_completely_empty=bool(
                        series.isna().all()
                    ),
                    sample_values=sample_values,
                )
            )

        return summaries

    # File-object handling

    @staticmethod
    def _rewind_if_possible(
        source: str | Path | IO,
    ) -> None:
        """Reset uploaded file-like objects before another read.
        
        File paths do not require rewinding.
        
        Args:
            source: Input source or source path.
        
        Returns:
            None.
        """

        if isinstance(
            source,
            (str, Path),
        ):
            return

        try:
            source.seek(0)
        except Exception:
            pass
