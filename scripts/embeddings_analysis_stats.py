"""
Example script for downstream embedding analysis

This demonstrates how to load and work with embeddings on a
different computer without GPU/model requirements.

Author: Curtis Crary Lab
Date: 2026-01-29
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import (
    load_slide_embeddings,
    load_multiple_slides,
    create_slide_level_features,
    compute_statistics,
)


def example_1_single_slide():
    """Example: Load and inspect a single slide"""
    print("="*60)
    print("Example 1: Loading a Single Slide")
    print("="*60)
    
    # Load embeddings
    dataset = load_slide_embeddings("41998")
    
    print(f"\nSlide: {dataset.slide_name}")
    print(f"Number of tiles: {dataset.n_tiles}")
    print(f"Embedding dimension: {dataset.embedding_dim}")
    
    # Get slide-level representation
    slide_embedding = dataset.get_slide_level_embedding('mean')
    print(f"Slide-level embedding shape: {slide_embedding.shape}")
    
    # Compute statistics
    stats = compute_statistics(dataset)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_2_multiple_slides():
    """Example: Load multiple slides and aggregate"""
    print("\n" + "="*60)
    print("Example 2: Loading Multiple Slides")
    print("="*60)
    
    slide_names = ["41998", "42054", "42056"]
    datasets = load_multiple_slides(slide_names)
    
    print(f"\nLoaded {len(datasets)} slides")
    
    # Create slide-level features
    X, _, names = create_slide_level_features(datasets, aggregation='mean')
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Slides: {names}")


def example_3_classification():
    """Example: Train a classifier on slide embeddings"""
    print("\n" + "="*60)
    print("Example 3: Classification Example")
    print("="*60)
    
    # Load slides
    slide_names = ["41998", "42054", "42056"]
    datasets = load_multiple_slides(slide_names, verbose=False)
    
    # Create fake labels for demonstration
    # In practice, you'd load these from a CSV or database
    labels = {
        "41998": 0,  # Control
        "42054": 1,  # Disease
        "42056": 1,  # Disease
    }
    
    # Create feature matrix
    X, y, names = create_slide_level_features(datasets, labels, aggregation='mean')
    
    print(f"Feature matrix: {X.shape}")
    print(f"Labels: {y}")
    
    # Note: This is just a demonstration with 3 samples
    # In practice you'd need many more samples
    print("\n⚠️  Note: Need more samples for real classification")


def example_4_tile_level_analysis():
    """Example: Work with individual tiles"""
    print("\n" + "="*60)
    print("Example 4: Tile-Level Analysis")
    print("="*60)
    
    dataset = load_slide_embeddings("41998")
    
    # Filter high-quality tiles
    if dataset.tissue_ratios is not None:
        high_quality = dataset.filter_by_tissue_ratio(min_ratio=0.5)
        print(f"Original tiles: {dataset.n_tiles}")
        print(f"High-quality tiles (>50% tissue): {high_quality.n_tiles}")
    
    # Convert to DataFrame for analysis
    df = dataset.to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())


def example_5_batch_processing():
    """Example: Process all slides in directory"""
    print("\n" + "="*60)
    print("Example 5: Batch Processing All Slides")
    print("="*60)
    
    from src.embeddings.loader import load_all_slides_from_directory
    
    # Load all available slides
    datasets = load_all_slides_from_directory()
    
    # Compute statistics for all
    stats_list = []
    for name, ds in datasets.items():
        stats = compute_statistics(ds)
        stats_list.append(stats)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_list)
    print("\nStatistics for all slides:")
    print(stats_df)
    
    # Save to CSV
    stats_df.to_csv("outputs/embedding_statistics.csv", index=False)
    print("\n✓ Saved statistics to outputs/embedding_statistics.csv")


def example_6_similarity_analysis():
    """Example: Find similar tiles or slides"""
    print("\n" + "="*60)
    print("Example 6: Similarity Analysis")
    print("="*60)
    
    from scipy.spatial.distance import cosine
    
    # Load two slides
    ds1 = load_slide_embeddings("41998", verbose=False)
    ds2 = load_slide_embeddings("42054", verbose=False)
    
    # Compute slide-level embeddings
    emb1 = ds1.get_slide_level_embedding('mean')
    emb2 = ds2.get_slide_level_embedding('mean')
    
    # Compute similarity
    similarity = 1 - cosine(emb1, emb2)
    print(f"\nCosine similarity between slides: {similarity:.4f}")
    
    # Find most similar tiles within a slide
    tile1 = ds1.embeddings[0]
    similarities = [1 - cosine(tile1, tile2) for tile2 in ds1.embeddings[1:100]]
    most_similar_idx = np.argmax(similarities) + 1
    
    print(f"Most similar tile to tile 0: tile {most_similar_idx}")
    print(f"Similarity: {similarities[most_similar_idx-1]:.4f}")


def main():
    """Run all examples"""
    print("WSI Embedding Analysis Examples")
    print("="*60)
    
    try:
        example_1_single_slide()
        example_2_multiple_slides()
        example_3_classification()
        example_4_tile_level_analysis()
        example_5_batch_processing()
        example_6_similarity_analysis()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Error: {e}")
        print("\nMake sure you've run the embedding extraction pipeline first:")
        print("  python scripts/wsi_embedding_pipeline.py")


if __name__ == "__main__":
    main()
