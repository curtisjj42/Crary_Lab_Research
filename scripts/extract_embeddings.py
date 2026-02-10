
"""
WSI Embedding Extraction Pipeline

This script processes whole slide images (WSI) using the UNI foundation model to:
1. Load and authenticate with Hugging Face
2. Tile WSI slides at configurable resolution
3. Filter tiles based on tissue content
4. Extract embeddings using the UNI model
5. Perform quality checks and spatial coherence analysis
6. Save embeddings and metadata in a structured format

Configuration is loaded from .env file. No command-line arguments required.

Author: Curtis Crary Lab
Date: 2026-01-29
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import torch
import timm
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from PIL import Image, ImageStat
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms

# ============================================================================
# Configuration Loading
# ============================================================================

def load_configuration():
    """Load configuration from .env file"""
    load_dotenv()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    config = {
        # Authentication
        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN'),
        'openslide_dll_dir': os.getenv('OPENSLIDE_DLL_DIR'),
        
        # Paths
        'slides_dir': project_root / 'slides',
        'output_dir': project_root / 'outputs/wsi_embeddings',
        'eval_dir': project_root / 'outputs/tiling_evals',
        
        # Tiling parameters
        'tile_size': 224,
        'pyramid_level': 0,
        'tile_overlap': 0,
        'tissue_threshold': 0.3,  # Minimum tissue ratio to keep tile
        
        # Model parameters
        'model_name': 'hf-hub:mahmoodLab/uni',
        'batch_size': 256,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Quality check parameters
        'n_clusters': 8,
        'spatial_coherence_samples': 1000,
        'spatial_coherence_radius': 1000,  # pixels
    }
    
    # Validate required fields
    if not config['huggingface_token']:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
    
    if not config['openslide_dll_dir']:
        raise ValueError("OPENSLIDE_DLL_DIR not found in .env file")
    
    # Create output directories
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['eval_dir'].mkdir(parents=True, exist_ok=True)
    
    return config


# ============================================================================
# OpenSlide Setup
# ============================================================================

def setup_openslide(dll_dir: str):
    """Configure OpenSlide DLL directory"""
    os.add_dll_directory(dll_dir)
    import openslide
    return openslide


# ============================================================================
# Hugging Face Authentication
# ============================================================================

def authenticate_huggingface(token: str):
    """Authenticate with Hugging Face and verify UNI model access"""
    login(token=token)
    
    # Verify access
    api = HfApi()
    try:
        model_info = api.model_info("MahmoodLab/UNI")
        print("✓ Successfully authenticated with Hugging Face")
        print(f"  Model ID: {model_info.modelId}")
        print(f"  Gated: {model_info.gated}")
    except Exception as e:
        print("✗ Cannot access UNI model")
        print(f"  Error: {e}")
        raise


# ============================================================================
# Model Loading
# ============================================================================

def load_uni_model(model_name: str, device: str):
    """Load UNI foundation model and create transform pipeline"""
    print(f"Loading UNI model on {device}...")
    
    model = timm.create_model(
        model_name,
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True
    )
    model.eval()
    model = model.to(device)
    
    # Create preprocessing transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("✓ Model loaded successfully")
    return model, transform


# ============================================================================
# WSI Tiling
# ============================================================================

def tile_wsi(slide, tile_size: int = 224, level: int = 0, overlap: int = 0) -> List[Dict]:
    """
    Tile a WSI into smaller patches
    
    Args:
        slide: OpenSlide object
        tile_size: Size of each tile (pixels)
        level: Pyramid level to use (0 = highest resolution)
        overlap: Overlap between tiles (pixels)
    
    Returns:
        List of tiles with their coordinates and images
    """
    width, height = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    
    tiles = []
    stride = tile_size - overlap
    
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Convert coordinates to level 0 (base resolution)
            x_level0 = int(x * downsample)
            y_level0 = int(y * downsample)
            
            # Read the tile from the slide
            tile = slide.read_region(
                (x_level0, y_level0), 
                level, 
                (tile_size, tile_size)
            )
            
            # Store tile with metadata
            tiles.append({
                'image': tile.convert('RGB'),
                'x': x_level0,
                'y': y_level0,
                'level': level
            })
    
    return tiles


# ============================================================================
# Tissue Detection
# ============================================================================

def filter_tissue_tiles(tiles: List[Dict], tissue_threshold: float = 0.3) -> List[Dict]:
    """
    Filter tiles to keep only those with sufficient tissue content
    
    Args:
        tiles: List of tile dictionaries
        tissue_threshold: Minimum tissue ratio (0-1) to keep tile
    
    Returns:
        Filtered list of tiles with tissue_ratio added
    """
    tissue_tiles = []
    
    for tile_data in tiles:
        tile = tile_data['image']
        # Convert to grayscale and check if mostly white (background)
        gray = np.array(tile.convert('L'))
        tissue_ratio = np.sum(gray < 220) / gray.size
        
        tile_data['tissue_ratio'] = tissue_ratio
        if tissue_ratio > tissue_threshold:
            tissue_tiles.append(tile_data)
    
    print(f"Total tiles: {len(tiles)}")
    print(f"Tissue tiles: {len(tissue_tiles)} ({len(tissue_tiles)/len(tiles)*100:.1f}%)")
    
    return tissue_tiles


# ============================================================================
# Embedding Extraction
# ============================================================================

def extract_embeddings(tiles: List[Dict], model, transform, device: str, batch_size: int = 256) -> List[Dict]:
    """
    Extract embeddings for all tiles using UNI model
    
    Args:
        tiles: List of tile dictionaries
        model: UNI model
        transform: Image preprocessing transform
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        List of embeddings with spatial coordinates
    """
    embeddings = []
    
    print(f"Extracting embeddings for {len(tiles)} tiles...")
    
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            
            # Preprocess batch
            batch_images = [transform(tile_data['image']) for tile_data in batch_tiles]
            batch_tensors = torch.stack(batch_images).to(device)
            
            # Extract embeddings
            batch_embeddings = model(batch_tensors)
            
            # Store with metadata
            for j, tile_data in enumerate(batch_tiles):
                embeddings.append({
                    'embedding': batch_embeddings[j].cpu().numpy(),
                    'x': tile_data['x'],
                    'y': tile_data['y'],
                    'tissue_ratio': tile_data.get('tissue_ratio', None)
                })
            
            if (i + batch_size) % (batch_size * 4) == 0:
                print(f"  Processed {min(i + batch_size, len(tiles))}/{len(tiles)} tiles")
    
    print("✓ Embedding extraction complete")
    return embeddings


# ============================================================================
# Quality Checks
# ============================================================================

def run_quality_checks(embeddings: List[Dict], save_dir: Path):
    """Run comprehensive quality checks on embeddings"""
    print("\n" + "="*50)
    print("EMBEDDING QUALITY REPORT")
    print("="*50)
    
    # Extract embedding matrix
    embeddings_only = np.array([item['embedding'] for item in embeddings])
    
    # Shape and basic stats
    print(f"\n📊 Shape: {embeddings_only.shape}")
    print(f"   Tiles: {embeddings_only.shape[0]:,}")
    print(f"   Dimensions: {embeddings_only.shape[1]}")
    
    # Distribution statistics
    print(f"\n📈 Distribution:")
    print(f"   Mean: {embeddings_only.mean():.4f}")
    print(f"   Std:  {embeddings_only.std():.4f}")
    print(f"   Min:  {embeddings_only.min():.4f}")
    print(f"   Max:  {embeddings_only.max():.4f}")
    
    # Data quality
    print(f"\n✓ Data Quality:")
    print(f"   NaN values: {np.isnan(embeddings_only).sum()}")
    print(f"   Inf values: {np.isinf(embeddings_only).sum()}")
    
    # L2 norms
    norms = np.linalg.norm(embeddings_only, axis=1)
    print(f"\n📏 L2 Norms:")
    print(f"   Mean: {norms.mean():.4f}")
    print(f"   Std:  {norms.std():.4f}")
    print(f"   Range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    # Check for degenerate embeddings
    unique_rows = np.unique(embeddings_only, axis=0).shape[0]
    print(f"\n🔍 Diversity:")
    print(f"   Unique embeddings: {unique_rows:,} / {embeddings_only.shape[0]:,}")
    
    # Save quality report
    report_path = save_dir / 'quality_report.txt'
    with open(report_path, 'w') as f:
        f.write("EMBEDDING QUALITY REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Shape: {embeddings_only.shape}\n")
        f.write(f"Mean: {embeddings_only.mean():.4f}\n")
        f.write(f"Std: {embeddings_only.std():.4f}\n")
        f.write(f"L2 Norm Mean: {norms.mean():.4f}\n")
        f.write(f"Unique embeddings: {unique_rows:,}\n")
    
    return embeddings_only, norms


# ============================================================================
# Visualization
# ============================================================================

def visualize_tiling(tiles: List[Dict], slide, save_path: Path):
    """Visualize tile locations and tissue content"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show thumbnail
    thumbnail = slide.get_thumbnail((1000, 1000))
    ax1.imshow(thumbnail)
    ax1.set_title('Slide Thumbnail with Tile Locations')
    
    # Overlay tile positions
    thumb_w, thumb_h = thumbnail.size
    slide_w, slide_h = slide.dimensions
    
    for tile_data in tiles:
        x = tile_data['x'] / slide_w * thumb_w
        y = tile_data['y'] / slide_h * thumb_h
        tissue_ratio = tile_data.get('tissue_ratio', 0)
        color = 'green' if tissue_ratio > 0.3 else 'red'
        ax1.plot(x, y, 'o', color=color, markersize=2, alpha=0.5)
    
    ax1.legend(['Tissue', 'Background'])
    
    # Histogram of tissue ratios
    tissue_ratios = [t['tissue_ratio'] for t in tiles]
    ax2.hist(tissue_ratios, bins=50, edgecolor='black')
    ax2.axvline(0.3, color='red', linestyle='--', label='Threshold')
    ax2.set_xlabel('Tissue Ratio')
    ax2.set_ylabel('Number of Tiles')
    ax2.set_title('Tissue Content Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Tiling visualization saved to {save_path}")


def visualize_embedding_space(embeddings_only: np.ndarray, embeddings: List[Dict], 
                              norms: np.ndarray, save_dir: Path):
    """Visualize embeddings in reduced dimensions"""
    print("\nGenerating embedding visualizations...")
    
    # PCA
    print("  Running PCA...")
    pca = PCA(n_components=50)
    pca_embeddings = pca.fit_transform(embeddings_only)
    print(f"  Variance explained by first 50 components: {pca.explained_variance_ratio_.sum():.2%}")
    
    # t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(pca_embeddings)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. PCA variance explained
    axes[0, 0].plot(np.cumsum(pca.explained_variance_ratio_)[:50])
    axes[0, 0].set_xlabel('Number of Components')
    axes[0, 0].set_ylabel('Cumulative Variance Explained')
    axes[0, 0].set_title('PCA: Cumulative Variance')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0.9, color='red', linestyle='--', label='90% variance')
    axes[0, 0].legend()
    
    # 2. PCA 2D
    scatter = axes[0, 1].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1],
                                alpha=0.5, s=10, c=norms, cmap='viridis')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('PCA: First 2 Components (colored by L2 norm)')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # 3. t-SNE
    y_coords = np.array([item['y'] for item in embeddings])
    scatter = axes[1, 0].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1],
                                alpha=0.6, s=10, c=y_coords, cmap='coolwarm')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    axes[1, 0].set_title('t-SNE: Colored by Y-coordinate')
    plt.colorbar(scatter, ax=axes[1, 0], label='Y position')
    
    # 4. Embedding norm distribution
    axes[1, 1].hist(norms, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(norms.mean(), color='red', linestyle='--',
                      label=f'Mean: {norms.mean():.2f}')
    axes[1, 1].set_xlabel('L2 Norm')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Embedding Norms')
    axes[1, 1].legend()
    
    plt.tight_layout()
    save_path = save_dir / 'embedding_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Embedding analysis saved to {save_path}")
    
    return pca_embeddings, tsne_embeddings


def spatial_coherence_analysis(embeddings_only: np.ndarray, embeddings: List[Dict],
                               save_dir: Path, n_samples: int = 1000, 
                               radius: float = 1000):
    """Check if spatially close tiles have similar embeddings"""
    print("\nPerforming spatial coherence analysis...")
    
    np.random.seed(42)
    sample_idx = np.random.choice(len(embeddings), min(n_samples, len(embeddings)), replace=False)
    
    spatial_distances = []
    embedding_similarities = []
    
    for i in sample_idx:
        x1, y1 = embeddings[i]['x'], embeddings[i]['y']
        
        # Find nearby tiles
        nearby_tiles = []
        for j, item in enumerate(embeddings):
            if j == i:
                continue
            x2, y2 = item['x'], item['y']
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < radius:
                nearby_tiles.append((j, dist))
        
        if nearby_tiles:
            j, spatial_dist = min(nearby_tiles, key=lambda x: x[1])
            sim = 1 - cosine(embeddings_only[i], embeddings_only[j])
            spatial_distances.append(spatial_dist)
            embedding_similarities.append(sim)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(spatial_distances, embedding_similarities, alpha=0.5, s=20)
    plt.xlabel('Spatial Distance (pixels)')
    plt.ylabel('Embedding Cosine Similarity')
    plt.title('Spatial Coherence: Do nearby tiles have similar embeddings?')
    
    # Add correlation
    corr, p_val = spearmanr(spatial_distances, embedding_similarities)
    plt.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np-value = {p_val:.2e}',
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_dir / 'spatial_coherence.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Correlation: {corr:.3f} (p={p_val:.2e})")
    print(f"✓ Spatial coherence saved to {save_path}")


def cluster_analysis(embeddings_only: np.ndarray, embeddings: List[Dict],
                    save_dir: Path, n_clusters: int = 8):
    """Find distinct tile patterns through clustering"""
    print(f"\nClustering embeddings into {n_clusters} groups...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_only)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Spatial distribution
    x_coords = np.array([item['x'] for item in embeddings])
    y_coords = np.array([item['y'] for item in embeddings])
    
    scatter = axes[0].scatter(x_coords, y_coords, c=labels, cmap='tab10',
                             s=20, alpha=0.6)
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    axes[0].set_title(f'Spatial Distribution of {n_clusters} Clusters')
    axes[0].invert_yaxis()
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    norm = mcolors.Normalize(vmin=unique.min(), vmax=unique.max())
    cmap = cm.tab10(norm(unique))
    
    axes[1].bar(unique, counts, color=cmap)
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Number of Tiles')
    axes[1].set_title('Cluster Sizes')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'cluster_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n🔬 Cluster Analysis:")
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id}: {count:,} tiles ({count / len(labels) * 100:.1f}%)")
    
    print(f"✓ Cluster analysis saved to {save_path}")
    return labels


# ============================================================================
# Data Persistence
# ============================================================================

def save_embeddings(embeddings: List[Dict], slide_name: str, output_dir: Path):
    """Save embeddings in multiple formats for easy loading"""
    
    # Create slide-specific directory
    slide_dir = output_dir / slide_name
    slide_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Full data with metadata (pickle)
    full_path = slide_dir / f'{slide_name}_embeddings_full.pkl'
    with open(full_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"\n✓ Saved full embeddings to {full_path}")
    
    # 2. Embedding matrix only (numpy)
    embeddings_only = np.array([item['embedding'] for item in embeddings])
    matrix_path = slide_dir / f'{slide_name}_embeddings_matrix.npy'
    np.save(matrix_path, embeddings_only)
    print(f"✓ Saved embedding matrix to {matrix_path}")
    
    # 3. Coordinates (numpy)
    coords = np.array([[item['x'], item['y']] for item in embeddings])
    coords_path = slide_dir / f'{slide_name}_coordinates.npy'
    np.save(coords_path, coords)
    print(f"✓ Saved coordinates to {coords_path}")
    
    # 4. Metadata (text summary)
    metadata_path = slide_dir / f'{slide_name}_metadata.txt'
    with open(metadata_path, 'w') as f:
        f.write(f"Slide: {slide_name}\n")
        f.write(f"Number of tiles: {len(embeddings)}\n")
        f.write(f"Embedding dimension: {embeddings_only.shape[1]}\n")
        f.write(f"Mean tissue ratio: {np.mean([e.get('tissue_ratio', 0) for e in embeddings]):.3f}\n")
    print(f"✓ Saved metadata to {metadata_path}")
    
    return slide_dir


# ============================================================================
# Main Pipeline
# ============================================================================

def process_slide(slide_path: Path, config: Dict, openslide, model, transform):
    """Process a single WSI slide through the entire pipeline"""
    
    slide_name = slide_path.stem
    print(f"\n{'='*60}")
    print(f"Processing slide: {slide_name}")
    print(f"{'='*60}")
    
    # Load slide
    print(f"\nLoading slide from {slide_path}...")
    slide = openslide.OpenSlide(str(slide_path))
    print(f"  Dimensions: {slide.dimensions}")
    print(f"  Levels: {slide.level_count}")
    
    # Tile slide
    print(f"\nTiling slide (size={config['tile_size']}, level={config['pyramid_level']})...")
    tiles = tile_wsi(
        slide,
        tile_size=config['tile_size'],
        level=config['pyramid_level'],
        overlap=config['tile_overlap']
    )
    
    # Filter tissue tiles
    print("\nFiltering for tissue content...")
    tissue_tiles = filter_tissue_tiles(tiles, config['tissue_threshold'])
    
    # Visualize tiling
    print("\nGenerating tiling visualization...")
    tiling_viz_path = config['eval_dir'] / f'{slide_name}_tiling_evaluation.png'
    visualize_tiling(tissue_tiles, slide, tiling_viz_path)
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(
        tissue_tiles,
        model,
        transform,
        config['device'],
        config['batch_size']
    )
    
    # Quality checks
    print("\nRunning quality checks...")
    embeddings_only, norms = run_quality_checks(embeddings, config['eval_dir'])
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_embedding_space(embeddings_only, embeddings, norms, config['eval_dir'])
    spatial_coherence_analysis(embeddings_only, embeddings, config['eval_dir'],
                               config['spatial_coherence_samples'],
                               config['spatial_coherence_radius'])
    cluster_analysis(embeddings_only, embeddings, config['eval_dir'], config['n_clusters'])
    
    # Save embeddings
    print("\nSaving embeddings...")
    slide_dir = save_embeddings(embeddings, slide_name, config['output_dir'])
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete for {slide_name}")
    print(f"{'='*60}")
    
    return slide_dir


def main():
    """Main pipeline execution"""
    print("="*60)
    print("WSI Embedding Extraction Pipeline")
    print("="*60)
    
    # Load configuration
    config = load_configuration()
    print(f"\nConfiguration loaded:")
    print(f"  Slides directory: {config['slides_dir']}")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Device: {config['device']}")
    
    # Setup OpenSlide
    print("\nSetting up OpenSlide...")
    openslide = setup_openslide(config['openslide_dll_dir'])
    print("✓ OpenSlide configured")
    
    # Authenticate with Hugging Face
    print("\nAuthenticating with Hugging Face...")
    authenticate_huggingface(config['huggingface_token'])
    
    # Load model
    model, transform = load_uni_model(config['model_name'], config['device'])
    
    # Find all slides
    slide_paths = list(config['slides_dir'].glob('*.svs'))
    if not slide_paths:
        print(f"\n⚠ No .svs files found in {config['slides_dir']}")
        print("Please add WSI slides to the slides directory")
        return
    
    print(f"\nFound {len(slide_paths)} slides to process:")
    for path in slide_paths:
        print(f"  - {path.name}")
    
    # Process each slide
    for slide_path in slide_paths:
        try:
            process_slide(slide_path, config, openslide, model, transform)
        except Exception as e:
            print(f"\n✗ Error processing {slide_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("Pipeline execution complete!")
    print("="*60)


if __name__ == '__main__':
    main()
