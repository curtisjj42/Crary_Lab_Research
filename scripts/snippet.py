"""
Embedding Loader Utilities

Lightweight utilities for loading and working with WSI tile embeddings
on machines without GPU/model requirements. Designed for downstream analysis.

Author: Curtis Crary Lab
Date: 2026-01-29
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class EmbeddingDataset:
    """Container for slide embeddings with metadata"""
    
    slide_name: str
    embeddings: np.ndarray  # Shape: (n_tiles, embedding_dim)
    coordinates: np.ndarray  # Shape: (n_tiles, 2) - [x, y]
    tissue_ratios: Optional[np.ndarray] = None  # Shape: (n_tiles,)
    metadata: Optional[Dict] = None
    
    @property
    def n_tiles(self) -> int:
        """Number of tiles"""
        return self.embeddings.shape[0]
    
    @property
    def embedding_dim(self) -> int:
        """Embedding dimension"""
        return self.embeddings.shape[1]
    
    def __repr__(self) -> str:
        return (f"EmbeddingDataset(slide='{self.slide_name}', "
                f"n_tiles={self.n_tiles}, dim={self.embedding_dim})")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with embeddings as separate columns"""
        data = {
            'slide_name': [self.slide_name] * self.n_tiles,
            'x': self.coordinates[:, 0],
            'y': self.coordinates[:, 1],
        }
        
        if self.tissue_ratios is not None:
            data['tissue_ratio'] = self.tissue_ratios
        
        # Add embedding dimensions as separate columns
        for i in range(self.embedding_dim):
            data[f'emb_{i}'] = self.embeddings[:, i]
        
        return pd.DataFrame(data)
    
    def get_slide_level_embedding(self, method: str = 'mean') -> np.ndarray:
        """
        Aggregate tile embeddings to slide-level representation
        
        Args:
            method: Aggregation method ('mean', 'max', 'median', 'weighted_mean')
        
        Returns:
            Slide-level embedding vector
        """
        if method == 'mean':
            return self.embeddings.mean(axis=0)
        elif method == 'max':
            return self.embeddings.max(axis=0)
        elif method == 'median':
            return np.median(self.embeddings, axis=0)
        elif method == 'weighted_mean' and self.tissue_ratios is not None:
            # Weight by tissue content
            weights = self.tissue_ratios / self.tissue_ratios.sum()
            return (self.embeddings.T @ weights).T
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def filter_by_tissue_ratio(self, min_ratio: float = 0.5) -> 'EmbeddingDataset':
        """Return new dataset with tiles filtered by tissue ratio"""
        if self.tissue_ratios is None:
            raise ValueError("No tissue ratios available")
        
        mask = self.tissue_ratios >= min_ratio
        
        return EmbeddingDataset(
            slide_name=self.slide_name,
            embeddings=self.embeddings[mask],
            coordinates=self.coordinates[mask],
            tissue_ratios=self.tissue_ratios[mask],
            metadata=self.metadata
        )


# ============================================================================
# Loading Functions
# ============================================================================

def load_slide_embeddings(
    slide_name: str,
    embeddings_dir: Union[str, Path] = "outputs/wsi_embeddings",
    format: str = "auto"
) -> EmbeddingDataset:
    """
    Load embeddings for a single slide
    
    Args:
        slide_name: Name of the slide (without extension)
        embeddings_dir: Directory containing embedding files
        format: File format to load ('numpy', 'pickle', 'auto')
    
    Returns:
        EmbeddingDataset object
    
    Examples:
        >>> dataset = load_slide_embeddings("41998")
        >>> print(f"Loaded {dataset.n_tiles} tiles")
    """
    embeddings_dir = Path(embeddings_dir)
    slide_dir = embeddings_dir / slide_name
    
    if not slide_dir.exists():
        raise FileNotFoundError(f"Slide directory not found: {slide_dir}")
    
    # Auto-detect format
    if format == "auto":
        if (slide_dir / f"{slide_name}_embeddings_matrix.npy").exists():
            format = "numpy"
        elif (slide_dir / f"{slide_name}_embeddings_full.pkl").exists():
            format = "pickle"
        else:
            raise FileNotFoundError(f"No embedding files found in {slide_dir}")
    
    # Load based on format
    if format == "numpy":
        return _load_numpy_format(slide_name, slide_dir)
    elif format == "pickle":
        return _load_pickle_format(slide_name, slide_dir)
    else:
        raise ValueError(f"Unknown format: {format}")


def _load_numpy_format(slide_name: str, slide_dir: Path) -> EmbeddingDataset:
    """Load embeddings from NumPy format"""
    embeddings = np.load(slide_dir / f"{slide_name}_embeddings_matrix.npy")
    coordinates = np.load(slide_dir / f"{slide_name}_coordinates.npy")
    
    # Try to load metadata
    metadata_path = slide_dir / f"{slide_name}_metadata.txt"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    return EmbeddingDataset(
        slide_name=slide_name,
        embeddings=embeddings,
        coordinates=coordinates,
        metadata=metadata
    )


def _load_pickle_format(slide_name: str, slide_dir: Path) -> EmbeddingDataset:
    """Load embeddings from pickle format (includes tissue ratios)"""
    with open(slide_dir / f"{slide_name}_embeddings_full.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Extract arrays
    embeddings = np.array([item['embedding'] for item in data])
    coordinates = np.array([[item['x'], item['y']] for item in data])
    tissue_ratios = np.array([item.get('tissue_ratio', 0.0) for item in data])
    
    return EmbeddingDataset(
        slide_name=slide_name,
        embeddings=embeddings,
        coordinates=coordinates,
        tissue_ratios=tissue_ratios
    )


def load_multiple_slides(
    slide_names: List[str],
    embeddings_dir: Union[str, Path] = "outputs/wsi_embeddings",
    format: str = "auto",
    verbose: bool = True
) -> Dict[str, EmbeddingDataset]:
    """
    Load embeddings for multiple slides
    
    Args:
        slide_names: List of slide names
        embeddings_dir: Directory containing embedding files
        format: File format to load
        verbose: Print progress
    
    Returns:
        Dictionary mapping slide_name -> EmbeddingDataset
    
    Examples:
        >>> datasets = load_multiple_slides(["41998", "42054", "42056"])
        >>> for name, ds in datasets.items():
        ...     print(f"{name}: {ds.n_tiles} tiles")
    """
    datasets = {}
    
    for slide_name in slide_names:
        if verbose:
            print(f"Loading {slide_name}...", end=" ")
        
        try:
            dataset = load_slide_embeddings(slide_name, embeddings_dir, format)
            datasets[slide_name] = dataset
            if verbose:
                print(f"✓ ({dataset.n_tiles} tiles)")
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            continue
    
    return datasets


def create_slide_level_features(
    datasets: Dict[str, EmbeddingDataset],
    labels: Optional[Dict[str, int]] = None,
    aggregation: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create slide-level feature matrix for ML
    
    Args:
        datasets: Dictionary of EmbeddingDataset objects
        labels: Optional dictionary mapping slide_name -> label
        aggregation: Method to aggregate tile embeddings
    
    Returns:
        (X, y, slide_names) where:
            X: Feature matrix (n_slides, embedding_dim)
            y: Label array (n_slides,) or None
            slide_names: List of slide names in same order
    
    Examples:
        >>> datasets = load_multiple_slides(["41998", "42054"])
        >>> labels = {"41998": 0, "42054": 1}  # 0=control, 1=disease
        >>> X, y, names = create_slide_level_features(datasets, labels)
        >>> # Now train a classifier
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier()
        >>> clf.fit(X, y)
    """
    slide_names = sorted(datasets.keys())
    
    # Aggregate embeddings
    X = np.vstack([
        datasets[name].get_slide_level_embedding(aggregation)
        for name in slide_names
    ])
    
    # Get labels if provided
    if labels is not None:
        y = np.array([labels[name] for name in slide_names])
    else:
        y = None
    
    return X, y, slide_names


# ============================================================================
# Batch Operations
# ============================================================================

def load_all_slides_from_directory(
    embeddings_dir: Union[str, Path] = "outputs/wsi_embeddings",
    format: str = "auto",
    verbose: bool = True
) -> Dict[str, EmbeddingDataset]:
    """
    Automatically discover and load all slides in directory
    
    Args:
        embeddings_dir: Directory containing embedding subdirectories
        format: File format to load
        verbose: Print progress
    
    Returns:
        Dictionary mapping slide_name -> EmbeddingDataset
    """
    embeddings_dir = Path(embeddings_dir)
    
    # Find all subdirectories
    slide_dirs = [d for d in embeddings_dir.iterdir() if d.is_dir()]
    slide_names = [d.name for d in slide_dirs]
    
    if verbose:
        print(f"Found {len(slide_names)} slides in {embeddings_dir}")
    
    return load_multiple_slides(slide_names, embeddings_dir, format, verbose)


def export_to_csv(
    dataset: EmbeddingDataset,
    output_path: Union[str, Path],
    include_embeddings: bool = False
):
    """
    Export dataset to CSV for easy inspection
    
    Args:
        dataset: EmbeddingDataset to export
        output_path: Path to save CSV
        include_embeddings: Whether to include full embeddings (large file!)
    """
    if include_embeddings:
        df = dataset.to_dataframe()
    else:
        data = {
            'slide_name': [dataset.slide_name] * dataset.n_tiles,
            'x': dataset.coordinates[:, 0],
            'y': dataset.coordinates[:, 1],
        }
        if dataset.tissue_ratios is not None:
            data['tissue_ratio'] = dataset.tissue_ratios
        df = pd.DataFrame(data)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Exported to {output_path}")


def compute_statistics(dataset: EmbeddingDataset) -> Dict:
    """Compute summary statistics for a dataset"""
    stats = {
        'slide_name': dataset.slide_name,
        'n_tiles': dataset.n_tiles,
        'embedding_dim': dataset.embedding_dim,
        'embedding_mean': float(dataset.embeddings.mean()),
        'embedding_std': float(dataset.embeddings.std()),
        'embedding_min': float(dataset.embeddings.min()),
        'embedding_max': float(dataset.embeddings.max()),
        'l2_norm_mean': float(np.linalg.norm(dataset.embeddings, axis=1).mean()),
        'l2_norm_std': float(np.linalg.norm(dataset.embeddings, axis=1).std()),
    }
    
    if dataset.tissue_ratios is not None:
        stats.update({
            'tissue_ratio_mean': float(dataset.tissue_ratios.mean()),
            'tissue_ratio_std': float(dataset.tissue_ratios.std()),
            'tissue_ratio_min': float(dataset.tissue_ratios.min()),
            'tissue_ratio_max': float(dataset.tissue_ratios.max()),
        })
    
    # Spatial extent
    stats.update({
        'x_min': int(dataset.coordinates[:, 0].min()),
        'x_max': int(dataset.coordinates[:, 0].max()),
        'y_min': int(dataset.coordinates[:, 1].min()),
        'y_max': int(dataset.coordinates[:, 1].max()),
    })
    
    return stats
