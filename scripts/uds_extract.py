#!/usr/bin/env python
"""
CLI to run the UDS table extraction and dataset alignment end-to-end.

Example:
  python scripts/uds_extract.py \
    --csv data-files/investigator_nacc67.csv \
    --pdf data-files/rdd_uds.pdf \
    --pages 23 27 \
    --out outputs/uds_extraction \
    --mmse-cols C1SCORE C1MMSE \
    --moca-cols C1MOCA C2MOCA

Notes:
- Page indices are zero-based in the code. If you pass 23 27, it will read pages 23..27 inclusive.
- If you are unsure of MMSE/MOCA columns, omit them and add later; the script will still produce catalog and availability.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.data.uds_extraction import (
    build_variable_catalog,
    load_nacc_csv,
    align_dataset_to_catalog,
    compute_empty_rows_mask,
    plot_availability_heatmap,
)


def _split_cols(values: List[str] | None) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for v in values:
        out.extend([x for x in v.split(',') if x])
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def main():
    p = argparse.ArgumentParser(description="UDS table extraction and dataset alignment")
    p.add_argument("--csv", required=True, help="Path to investigator NACC CSV")
    p.add_argument("--pdf", required=True, help="Path to UDS PDF spec (for catalog)")
    p.add_argument("--pages", nargs=2, type=int, metavar=("START", "END"), required=True,
                   help="Zero-based inclusive page range, e.g., 23 27")
    p.add_argument("--out", default="outputs/uds_extraction", help="Output directory")
    p.add_argument("--mmse-cols", nargs='*', help="MMSE column names (space- or comma-separated)")
    p.add_argument("--moca-cols", nargs='*', help="MOCA column names (space- or comma-separated)")
    p.add_argument("--heatmap", action="store_true", help="Also save a data availability heatmap PNG")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build catalog
    catalog = build_variable_catalog(args.pdf, (args.pages[0], args.pages[1]))
    catalog_path = out_dir / "variable_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    # 2) Load CSV and align
    df = load_nacc_csv(args.csv)
    mmse_cols = _split_cols(args.mmse_cols)
    moca_cols = _split_cols(args.moca_cols)
    cleaned, summary = align_dataset_to_catalog(df, catalog, mmse_cols=mmse_cols, moca_cols=moca_cols)

    cleaned_path = out_dir / "cleaned_subset.parquet"

    # Fix for ArrowKeyError: "A type extension with name pandas.period already defined"
    # This prevents crashes when pandas attempts to re-register types already in pyarrow.
    try:
        import pyarrow
        try:
            pyarrow.unregister_extension_type("pandas.period")
            pyarrow.unregister_extension_type("pandas.interval")
        except Exception:
            pass
    except ImportError:
        pass

    cleaned.to_parquet(cleaned_path, index=False, engine='pyarrow')
    summary_path = out_dir / "availability_summary.csv"
    summary.to_csv(summary_path, index=False)

    # 3) Empty rows stats
    empty_mask = compute_empty_rows_mask(cleaned)
    stats_txt = (
        f"Rows total: {len(cleaned)}\n"
        f"Completely empty (all -4/NaN): {int(empty_mask.sum())}\n"
        f"With some data: {int((~empty_mask).sum())}\n"
    )
    (out_dir / "stats.txt").write_text(stats_txt)

    # 4) Optional heatmap
    if args.heatmap:
        plot_availability_heatmap(cleaned, out_path=str(out_dir / "availability_heatmap.png"))

    print("Saved:")
    print(f"  - {catalog_path}")
    print(f"  - {cleaned_path}")
    print(f"  - {summary_path}")
    print(f"  - {(out_dir / 'stats.txt')}")
    if args.heatmap:
        print(f"  - {(out_dir / 'availability_heatmap.png')}")


if __name__ == "__main__":
    main()
