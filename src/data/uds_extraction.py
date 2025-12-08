"""
UDS (Uniform Data Set) table extraction and dataset utilities.

This module streamlines logic prototyped in notebooks/exploratory/uds_table_extraction.ipynb
into reusable functions so the same handling can be performed consistently from
scripts or notebooks.

Key capabilities:
- Extract tables from the UDS reference PDF using pdfplumber.
- Build a variable catalog (e.g., for C1/C2 Neuropsych Battery) from selected pages.
- Select dataset columns from a CSV that match the catalog variable names.
- Tag participants with MMSE/MOCA availability.
- Summarize data availability and optionally plot a heatmap.

Notes
-----
- Missing sentinel values are treated as -4 by default (as seen in the notebook).
- The variable name is assumed to be in column index 2 of extracted table rows, per the PDF layout used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    pdfplumber = None  # Lazy import check will raise with a helpful message


@dataclass
class CatalogConfig:
    pdf_path: str
    page_range: Tuple[int, int]
    forms_filter: Sequence[str] = ("C1", "C2")


def _ensure_pdfplumber():
    if pdfplumber is None:
        raise ImportError(
            "pdfplumber is required for PDF extraction. Install it (pip) or ensure the environment is set up."
        )


def extract_pdf_tables(pdf_path: str, pages: Iterable[int]) -> List[List[List[Optional[str]]]]:
    """Extract all tables for the given pages using pdfplumber.

    Returns a list (per-page) of tables; each table is a list of rows; each row is a list of cell strings.
    """
    _ensure_pdfplumber()
    all_pages: List[List[List[Optional[str]]]] = []
    with pdfplumber.open(pdf_path) as pdf:  # type: ignore[operator]
        for p in pages:
            page = pdf.pages[p]
            tables = page.extract_tables() or []
            # Flatten multiple tables on a page into one list of rows for that page
            rows: List[List[Optional[str]]] = []
            for tbl in tables:
                if not tbl:
                    continue
                rows.extend(tbl)
            all_pages.append(rows)
    return all_pages


def _iter_variable_rows(all_pages_rows: List[List[List[Optional[str]]]]) -> Iterable[List[Optional[str]]]:
    """Yield candidate data rows, skipping short rows and header rows."""
    for page_rows in all_pages_rows:
        for row in page_rows:
            if not row or len(row) < 3:
                continue
            # Skip header rows where third column is "Variable name"
            if isinstance(row[2], str) and row[2].strip().lower() == "variable name":
                continue
            # Only consider rows with a non-empty variable name (col index 2)
            if row[2] and str(row[2]).strip():
                yield row


def build_variable_catalog(
    pdf_path: str,
    page_range: Tuple[int, int],
    forms_filter: Sequence[str] = ("C1", "C2"),
) -> pd.DataFrame:
    """Build a variable catalog DataFrame from a PDF and page range.

    Parameters
    ----------
    pdf_path : str
        Path to the UDS PDF reference.
    page_range : (start, end)
        Inclusive zero-based page indices to scan (e.g., (23, 27) to read 23..27).
    forms_filter : sequence of str
        Only keep rows whose form/section mentions any of these tokens (e.g., "C1"/"C2").

    Returns
    -------
    DataFrame with columns: form_field, variable_name, label (optional), source_page.
    """
    start, end = page_range
    pages = range(start, end + 1)
    all_pages_rows = extract_pdf_tables(pdf_path, pages)

    records = []
    for page_idx, row in enumerate(_iter_variable_rows(all_pages_rows), start=start):
        # Heuristic: form/section appears in column 0 (or 1) depending on PDF layout
        form_field = (row[0] or row[1] or "").strip() if len(row) > 1 else (row[0] or "")
        if forms_filter:
            text = form_field or ""
            if not any(tok in text for tok in forms_filter):
                continue
        variable_name = str(row[2]).strip() if row[2] is not None else ""
        label = str(row[3]).strip() if len(row) > 3 and row[3] is not None else ""
        if variable_name:
            records.append(
                {
                    "form_field": form_field,
                    "variable_name": variable_name,
                    "label": label,
                    "source_page": page_idx,
                }
            )

    catalog = pd.DataFrame.from_records(records).drop_duplicates("variable_name")
    return catalog


def select_dataset_columns(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Return a view/subset of df with only variables from the catalog present in df columns."""
    vars_in_df = [v for v in catalog["variable_name"].tolist() if v in df.columns]
    return df.loc[:, vars_in_df]


def tag_assessments(
    df: pd.DataFrame,
    mmse_cols: Sequence[str],
    moca_cols: Sequence[str],
    *,
    missing_sentinel: int | float = -4,
) -> pd.DataFrame:
    """Add boolean columns has_MMSE and has_MOCA based on presence of any valid data across provided columns."""
    out = df.copy()
    def _is_valid(series: pd.Series) -> pd.Series:
        return series.notna() & (series != missing_sentinel)

    out["has_MMSE"] = out[mmse_cols].apply(lambda s: _is_valid(s).any(), axis=1) if mmse_cols else False
    out["has_MOCA"] = out[moca_cols].apply(lambda s: _is_valid(s).any(), axis=1) if moca_cols else False
    return out


def compute_empty_rows_mask(
    df: pd.DataFrame,
    *,
    exclude: Sequence[str] = ("has_MMSE", "has_MOCA"),
    missing_sentinel: int | float = -4,
) -> pd.Series:
    """Return a boolean mask where True indicates rows with all excluded columns removed are empty/missing."""
    data_cols = [c for c in df.columns if c not in set(exclude)]
    def _row_empty(row: pd.Series) -> bool:
        vals = row[data_cols]
        return not ((vals.notna()) & (vals != missing_sentinel)).any()
    return df.apply(_row_empty, axis=1)


def summarize_availability(
    df: pd.DataFrame,
    *,
    exclude: Sequence[str] = ("has_MMSE", "has_MOCA"),
    missing_sentinel: int | float = -4,
) -> pd.DataFrame:
    """Summarize valid data counts and percentages per column (excluding helper columns)."""
    rows = []
    n = len(df)
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        valid = (s.notna()) & (s != missing_sentinel)
        count = int(valid.sum())
        rows.append({"Column": col, "Valid_Data_Count": count, "Percentage": (count / n * 100 if n else 0.0)})
    summary = pd.DataFrame(rows).sort_values("Valid_Data_Count", ascending=False)
    return summary


def plot_availability_heatmap(
    df: pd.DataFrame,
    *,
    exclude: Sequence[str] = ("has_MMSE", "has_MOCA"),
    sample_n: int = 100,
    seed: int = 42,
    missing_sentinel: int | float = -4,
    out_path: Optional[str] = None,
):
    """Plot a heatmap of data availability (1=valid, 0=missing/sentinel) for up to sample_n rows.

    If out_path is provided, saves the figure to that path; otherwise shows it interactively.
    """
    import numpy as np  # local import to keep base import light
    import matplotlib.pyplot as plt

    sample_size = min(sample_n, len(df))
    df_sample = df.sample(n=sample_size, random_state=seed) if sample_size else df
    data_cols = [c for c in df_sample.columns if c not in set(exclude)]
    binary = df_sample[data_cols].apply(lambda s: ((s.notna()) & (s != missing_sentinel)).astype(int))

    plt.figure(figsize=(15, 8))
    plt.imshow(binary.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Has Data (1) / No Data (0)')
    plt.xlabel('Sample Participants')
    plt.ylabel('Variables')
    plt.title(f'Data Availability Heatmap (Sample of {sample_size} participants)')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()


def load_nacc_csv(path: str) -> pd.DataFrame:
    """Convenience wrapper for reading the investigator NACC CSV (no special parsing yet)."""
    return pd.read_csv(path)


def align_dataset_to_catalog(
    df: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    mmse_cols: Optional[Sequence[str]] = None,
    moca_cols: Optional[Sequence[str]] = None,
    missing_sentinel: int | float = -4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a cleaned subset aligned to catalog and return (tagged_df, availability_summary).

    This mirrors the notebook flow in a compact, reusable way.
    """
    subset = select_dataset_columns(df, catalog)
    tagged = tag_assessments(subset, mmse_cols or [], moca_cols or [], missing_sentinel=missing_sentinel)
    summary = summarize_availability(tagged, exclude=("has_MMSE", "has_MOCA"), missing_sentinel=missing_sentinel)
    return tagged, summary


__all__ = [
    "CatalogConfig",
    "extract_pdf_tables",
    "build_variable_catalog",
    "select_dataset_columns",
    "tag_assessments",
    "compute_empty_rows_mask",
    "summarize_availability",
    "plot_availability_heatmap",
    "load_nacc_csv",
    "align_dataset_to_catalog",
]
