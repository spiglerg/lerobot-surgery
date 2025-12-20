"""
Tests for merge_datasets functionality.
"""
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_surgery.v3 import merge_datasets


def test_merge_two_datasets(dataset_a, dataset_b, output_path):
    """Test merging two datasets."""
    merged = merge_datasets(
        source_paths=[dataset_a.root, dataset_b.root],
        output_path=output_path,
    )

    # Verify merged dataset properties
    assert merged.num_episodes == 4  # 2 + 2
    assert merged.num_frames == 120  # (2*30) + (2*30)
    assert merged.fps == 30


def test_merge_three_datasets(dataset_a, dataset_b, dataset_c, output_path):
    """Test merging three datasets."""
    merged = merge_datasets(
        source_paths=[dataset_a.root, dataset_b.root, dataset_c.root],
        output_path=output_path,
    )

    # Verify merged dataset properties
    assert merged.num_episodes == 7  # 2 + 2 + 3
    assert merged.num_frames == 195  # (2*30) + (2*30) + (3*25)
    assert merged.fps == 30


def test_merge_with_overwrite(dataset_a, dataset_b, output_path):
    """Test that overwrite flag works correctly."""
    # Create initial merged dataset
    merge_datasets(
        source_paths=[dataset_a.root],
        output_path=output_path,
    )

    # Merge again with overwrite=True
    merged = merge_datasets(
        source_paths=[dataset_a.root, dataset_b.root],
        output_path=output_path,
        overwrite=True,
    )

    assert merged.num_episodes == 4


def test_merge_without_overwrite_raises_error(dataset_a, dataset_b, output_path):
    """Test that merging to existing path without overwrite raises error."""
    # Create initial dataset
    merge_datasets(
        source_paths=[dataset_a.root],
        output_path=output_path,
    )

    # Try to merge again without overwrite
    with pytest.raises(FileExistsError):
        merge_datasets(
            source_paths=[dataset_a.root, dataset_b.root],
            output_path=output_path,
            overwrite=False,
        )


def test_merge_empty_source_list_raises_error(output_path):
    """Test that empty source list raises ValueError."""
    with pytest.raises(ValueError, match="source_paths cannot be empty"):
        merge_datasets(
            source_paths=[],
            output_path=output_path,
        )


def test_merge_preserves_features(dataset_a, dataset_b, output_path):
    """Test that merged dataset preserves features from source datasets."""
    merged = merge_datasets(
        source_paths=[dataset_a.root, dataset_b.root],
        output_path=output_path,
    )

    # Check that features match the source datasets
    assert merged.features == dataset_a.features
    assert merged.features == dataset_b.features


def test_merged_episodes_are_sequential(dataset_a, dataset_b, output_path):
    """Test that episode indices in merged dataset are sequential."""
    merge_datasets(
        source_paths=[dataset_a.root, dataset_b.root],
        output_path=output_path,
    )

    # Reload the dataset to access metadata
    merged = LeRobotDataset(output_path.name, root=output_path)

    # Verify episode indices are 0, 1, 2, 3
    episode_indices = [ep["episode_index"] for ep in merged.meta.episodes]
    assert episode_indices == list(range(4))


def test_merge_can_be_loaded_as_dataset(dataset_a, dataset_b, output_path):
    """Test that merged dataset can be loaded as a LeRobotDataset."""
    merge_datasets(
        source_paths=[dataset_a.root, dataset_b.root],
        output_path=output_path,
    )

    # Load the merged dataset
    loaded = LeRobotDataset(output_path.name, root=output_path)
    assert loaded.num_episodes == 4
    assert loaded.num_frames == 120
