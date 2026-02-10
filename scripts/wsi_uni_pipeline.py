"""
WSI UNI Embedding Pipeline

This script loads the MahmoodLab/UNI model from Hugging Face (with optional
credential login), tiles whole-slide images (WSIs) using OpenSlide, performs
simple quality checks, extracts tile embeddings, and saves outputs for later
use.

Outputs per WSI:
- QC PNG: overlay of tile positions on the slide thumbnail (tissue vs background)
- QC CSV: per-tile summary with coordinates and tissue_ratio
- Embeddings Parquet: columns [x, y, level, tissue_ratio, embedding]
- Embeddings NPZ: arrays X (N,2), L (N,), T (N,), E (N,D)

Usage (examples):
  python scripts/wsi_uni_pipeline.py \
    --wsi C:\\path\\to\\slide.svs \
    --outdir outputs\\uni_embeddings \
    --hf-token %HUGGINGFACE_TOKEN% \
    --tile-size 224 --overlap 0 --level 0 --batch-size 256

Environment variables:
- HUGGINGFACE_TOKEN: HF token (optional if model is public or already logged in)
- OPENSLIDE_DLL_DIR (Windows): directory containing OpenSlide DLLs (optional)
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

os.add_dll_directory(r'C:\Users\curti\OpenSlide\bin')
def _maybe_add_openslide_dll_dir() -> None:
    # Windows-specific: allow user to set path to OpenSlide DLLs
    dll_dir = os.getenv("OPENSLIDE_DLL_DIR")
    if dll_dir and os.name == "nt":
        try:
            os.add_dll_directory(dll_dir)
        except Exception:
            # Best effort only; will fail later if OpenSlide cannot be found
            pass


def login_hf(token: str | None) -> None:
    """Login to Hugging Face if a token is provided.

    If token is None, no-op (assumes public access or prior login).
    """
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token=token)
    except Exception as e:
        print(f"Warning: Hugging Face login failed: {e}")


def load_uni_model(device: str = "auto"):
    """Load the UNI model via timm's Hugging Face integration.

    Returns: (model, transform, torch_device)
    """
    import torch
    import timm
    from torchvision import transforms

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    model = timm.create_model(
        "hf-hub:mahmoodLab/uni",  # note: HF repo is case-insensitive here
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    ).to(torch_device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, torch_device


def open_slide(wsi_path: str):
    import openslide
    slide = openslide.OpenSlide(wsi_path)
    return slide


def tile_wsi(slide, tile_size: int = 224, level: int = 0, overlap: int = 0) -> List[Dict[str, Any]]:
    """Tile a WSI into smaller patches.

    Returns a list of dicts: { 'image': PIL.Image, 'x': int, 'y': int, 'level': int }
    Coordinates x,y are at level 0 space.
    """
    width, height = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    tiles: List[Dict[str, Any]] = []
    stride = tile_size - overlap

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x_level0 = int(x * downsample)
            y_level0 = int(y * downsample)
            tile = slide.read_region((x_level0, y_level0), level, (tile_size, tile_size))
            tiles.append({
                'image': tile.convert('RGB'),
                'x': x_level0,
                'y': y_level0,
                'level': level,
            })
    return tiles


def evaluate_tiles(tiles: List[Dict[str, Any]], slide, save_png: Path | None) -> List[Dict[str, Any]]:
    """Compute simple tissue detection and optionally save a QC PNG overlay.

    Adds 'tissue_ratio' to each tile dict. Returns tiles (same objects) for convenience.
    """
    import matplotlib.pyplot as plt
    # Basic tissue detection via grayscale threshold proportion
    for t in tiles:
        gray = np.array(t['image'].convert('L'))
        tissue_ratio = float(np.sum(gray < 220) / gray.size)
        t['tissue_ratio'] = tissue_ratio

    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        thumbnail = slide.get_thumbnail((1000, 1000))
        ax1.imshow(thumbnail)
        ax1.set_title('Slide Thumbnail')

        thumb_w, thumb_h = thumbnail.size
        slide_w, slide_h = slide.dimensions
        for t in tiles:
            x = t['x'] / slide_w * thumb_w
            y = t['y'] / slide_h * thumb_h
            color = 'green' if t.get('tissue_ratio', 0.0) > 0.3 else 'red'
            ax1.plot(x, y, 'o', color=color, markersize=2, alpha=0.5)
        ax1.legend(['Tissue', 'Background'])

        tr = [float(tt.get('tissue_ratio', 0.0)) for tt in tiles]
        ax2.hist(tr, bins=50, edgecolor='black')
        ax2.axvline(0.3, color='red', linestyle='--', label='Threshold')
        ax2.set_xlabel('Tissue Ratio')
        ax2.set_ylabel('Number of Tiles')
        ax2.set_title('Tissue Content Distribution')
        ax2.legend()

        plt.tight_layout()
        fig.savefig(str(save_png), dpi=150)
        plt.close(fig)

    return tiles


def filter_tissue_tiles(tiles: List[Dict[str, Any]], min_tissue_ratio: float = 0.3) -> List[Dict[str, Any]]:
    return [t for t in tiles if float(t.get('tissue_ratio', 0.0)) >= min_tissue_ratio]


def extract_embeddings(tiles: List[Dict[str, Any]], model, transform, torch_device, batch_size: int = 256) -> List[Dict[str, Any]]:
    import torch
    out: List[Dict[str, Any]] = []
    if not tiles:
        return out
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            batch_images = [transform(t['image']) for t in batch_tiles]
            batch_tensors = torch.stack(batch_images).to(torch_device)
            batch_embeddings = model(batch_tensors)
            # Ensure tensor shape (N, D)
            emb = batch_embeddings.detach().cpu().numpy()
            for j, t in enumerate(batch_tiles):
                out.append({
                    'embedding': emb[j],
                    'x': int(t['x']),
                    'y': int(t['y']),
                    'level': int(t.get('level', 0)),
                    'tissue_ratio': float(t.get('tissue_ratio', 0.0)),
                })
    return out


def save_embeddings(outputs: List[Dict[str, Any]], base_path: Path) -> None:
    """Save outputs to Parquet (if available) and NPZ; also write a compact CSV of metadata.

    Files:
      - {base}.parquet (if pyarrow is available)
      - {base}.npz
      - {base}_tiles.csv (x,y,level,tissue_ratio)
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)

    if not outputs:
        # Still create empty marker files
        (base_path.parent / (base_path.name + "_EMPTY.json")).write_text(json.dumps({"empty": True}))
        return

    # Stack arrays
    E = np.stack([o['embedding'] for o in outputs], axis=0)
    X = np.stack([[o['x'], o['y']] for o in outputs], axis=0)
    L = np.array([o['level'] for o in outputs], dtype=np.int32)
    T = np.array([o.get('tissue_ratio', 0.0) for o in outputs], dtype=np.float32)

    # Try Parquet with array column
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame({
            'x': X[:, 0],
            'y': X[:, 1],
            'level': L,
            'tissue_ratio': T,
        })
        # Store embeddings as a fixed-size list using Arrow
        arr = paFixedSizeList_from_numpy(E)
        table = pa.Table.from_pandas(df)
        table = table.append_column('embedding', arr)
        pq.write_table(table, str(base_path.with_suffix('.parquet')))
    except Exception as e:
        # Fallback: skip Parquet
        print(f"Parquet save skipped ({e}).")

    # NPZ for universal Python loading
    np.savez_compressed(str(base_path.with_suffix('.npz')), E=E, X=X, L=L, T=T)

    # CSV of tile metadata
    try:
        import pandas as pd
        df_meta = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'level': L, 'tissue_ratio': T})
        df_meta.to_csv(str(base_path.parent / f"{base_path.name}_tiles.csv"), index=False)
    except Exception:
        # Minimal CSV via numpy
        np.savetxt(str(base_path.parent / f"{base_path.name}_tiles.csv"),
                   np.column_stack([X, L, T]), delimiter=",",
                   header="x,y,level,tissue_ratio", comments="", fmt=['%d', '%d', '%d', '%.6f'])


def paFixedSizeList_from_numpy(E: np.ndarray):
    """Helper to build a FixedSizeListArray column from a 2D numpy array."""
    import pyarrow as pa
    n, d = E.shape
    values = pa.array(E.reshape(-1), type=pa.float32()) if E.dtype != np.float64 else pa.array(E.reshape(-1), type=pa.float64())
    return pa.FixedSizeListArray.from_arrays(values, list_size=d)


def save_qc_csv(tiles_all: List[Dict[str, Any]], base_path: Path) -> None:
    try:
        import pandas as pd
        df = pd.DataFrame([
            {
                'x': int(t['x']), 'y': int(t['y']), 'level': int(t.get('level', 0)),
                'tissue_ratio': float(t.get('tissue_ratio', 0.0))
            } for t in tiles_all
        ])
        df.to_csv(str(base_path.with_suffix('.csv')), index=False)
    except Exception:
        # Minimal CSV via numpy if pandas missing
        X = np.array([[int(t['x']), int(t['y']), int(t.get('level', 0)), float(t.get('tissue_ratio', 0.0))] for t in tiles_all])
        np.savetxt(str(base_path.with_suffix('.csv')), X, delimiter=",",
                   header="x,y,level,tissue_ratio", comments="", fmt=['%d', '%d', '%d', '%.6f'])


def process_wsi(
    wsi_path: str,
    outdir: str,
    hf_token: str | None = None,
    tile_size: int = 224,
    overlap: int = 0,
    level: int = 0,
    batch_size: int = 256,
    min_tissue_ratio: float = 0.3,
    device: str = "auto",
) -> None:
    _maybe_add_openslide_dll_dir()
    login_hf(hf_token or os.getenv("HUGGINGFACE_TOKEN"))

    model, transform, torch_device = load_uni_model(device=device)
    slide = open_slide(wsi_path)

    print("Creating tiles...")
    tiles = tile_wsi(slide, tile_size=tile_size, level=level, overlap=overlap)
    print(f"Total tiles: {len(tiles)}")

    wsi_name = Path(wsi_path).stem
    out_base = Path(outdir) / wsi_name
    qc_png = Path(outdir) / "qc" / f"{wsi_name}_tiles.png"
    qc_csv_base = Path(outdir) / "qc" / f"{wsi_name}_tiles"

    print("Evaluating tiles (tissue detection) and generating QC plot...")
    evaluate_tiles(tiles, slide, qc_png)
    save_qc_csv(tiles, qc_csv_base)

    tiles_kept = filter_tissue_tiles(tiles, min_tissue_ratio=min_tissue_ratio)
    print(f"Tiles with tissue >= {min_tissue_ratio:.2f}: {len(tiles_kept)}")

    print("Extracting embeddings with UNI...")
    outputs = extract_embeddings(tiles_kept, model, transform, torch_device, batch_size=batch_size)

    print("Saving embeddings and metadata...")
    save_embeddings(outputs, out_base)

    print("Done.")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WSI tiling + UNI embedding pipeline")
    p.add_argument('--wsi', type=str, required=True, help='Path to a single WSI file (.svs, .tif, etc.)')
    p.add_argument('--outdir', type=str, default='outputs/uni_embeddings', help='Output directory')
    p.add_argument('--hf-token', type=str, default=None, help='Hugging Face token (optional)')
    p.add_argument('--tile-size', type=int, default=224)
    p.add_argument('--overlap', type=int, default=0)
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--min-tissue-ratio', type=float, default=0.3)
    p.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda')
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    process_wsi(
        wsi_path=args.wsi,
        outdir=args.outdir,
        hf_token=args.hf_token,
        tile_size=args.tile_size,
        overlap=args.overlap,
        level=args.level,
        batch_size=args.batch_size,
        min_tissue_ratio=args.min_tissue_ratio,
        device=args.device,
    )


if __name__ == '__main__':
    main()
