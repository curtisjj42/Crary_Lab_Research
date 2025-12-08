"""Data utilities package for the project."""

from .uds_extraction import (
    CatalogConfig,
    extract_pdf_tables,
    build_variable_catalog,
    select_dataset_columns,
    tag_assessments,
    compute_empty_rows_mask,
    summarize_availability,
    plot_availability_heatmap,
    load_nacc_csv,
    align_dataset_to_catalog,
)

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
