"""
Tests for remove_episodes functionality.
"""
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_surgery.v3 import remove_episodes


def test_remove_single_episode(dataset_a, output_path):
    """Test removing a single episode."""
    filtered = remove_episodes(
        dataset_path=dataset_a.root,
        episode_indices=[0],
        output_path=output_path,
    )

    # Verify filtered dataset
    assert filtered.num_episodes == 1  # Started with 2, removed 1
    assert filtered.num_frames == 30  # 1 episode * 30 frames


def test_remove_multiple_episodes(dataset_c, output_path):
    """Test removing multiple episodes."""
    # dataset_c has 3 episodes
    filtered = remove_episodes(
        dataset_path=dataset_c.root,
        episode_indices=[0, 2],  # Remove first and last
        output_path=output_path,
    )

    # Verify filtered dataset
    assert filtered.num_episodes == 1  # Kept episode 1
    assert filtered.num_frames == 25  # 1 episode * 25 frames


def test_remove_with_overwrite(dataset_a, output_path):
    """Test that overwrite flag works correctly."""
    # Create initial filtered dataset
    remove_episodes(
        dataset_path=dataset_a.root,
        episode_indices=[0],
        output_path=output_path,
    )

    # Remove again with overwrite=True
    filtered = remove_episodes(
        dataset_path=dataset_a.root,
        episode_indices=[1],
        output_path=output_path,
        overwrite=True,
    )

    assert filtered.num_episodes == 1


def test_remove_without_overwrite_raises_error(dataset_a, output_path):
    """Test that removing to existing path without overwrite raises error."""
    # Create initial dataset
    remove_episodes(
        dataset_path=dataset_a.root,
        episode_indices=[0],
        output_path=output_path,
    )

    # Try to remove again without overwrite
    with pytest.raises(FileExistsError):
        remove_episodes(
            dataset_path=dataset_a.root,
            episode_indices=[1],
            output_path=output_path,
            overwrite=False,
        )


def test_remove_invalid_episode_raises_error(dataset_a, output_path):
    """Test that removing invalid episode index raises ValueError."""
    # dataset_a has 2 episodes (indices 0, 1)
    with pytest.raises(ValueError, match="Invalid episode indices"):
        remove_episodes(
            dataset_path=dataset_a.root,
            episode_indices=[5],  # Invalid index
            output_path=output_path,
        )


def test_remove_all_episodes_raises_error(dataset_a, output_path):
    """Test that removing all episodes raises ValueError."""
    with pytest.raises(ValueError, match="Cannot remove all episodes"):
        remove_episodes(
            dataset_path=dataset_a.root,
            episode_indices=[0, 1],  # All episodes
            output_path=output_path,
        )


def test_removed_episodes_are_reindexed(dataset_c, output_path):
    """Test that remaining episodes are re-indexed sequentially."""
    # dataset_c has 3 episodes (0, 1, 2)
    # Remove episode 1
    remove_episodes(
        dataset_path=dataset_c.root,
        episode_indices=[1],
        output_path=output_path,
    )

    # Reload the dataset to access metadata
    filtered = LeRobotDataset(output_path.name, root=output_path)

    # Verify episode indices are now 0, 1 (originally 0, 2)
    episode_indices = [ep["episode_index"] for ep in filtered.meta.episodes]
    assert episode_indices == [0, 1]


def test_remove_preserves_features(dataset_a, output_path):
    """Test that filtered dataset preserves features from source dataset."""
    filtered = remove_episodes(
        dataset_path=dataset_a.root,
        episode_indices=[0],
        output_path=output_path,
    )

    # Check that features match the source dataset
    assert filtered.features == dataset_a.features
    assert filtered.fps == dataset_a.fps


def test_removed_dataset_can_be_loaded(dataset_a, output_path):
    """Test that filtered dataset can be loaded as a LeRobotDataset."""
    remove_episodes(
        dataset_path=dataset_a.root,
        episode_indices=[0],
        output_path=output_path,
    )

    # Load the filtered dataset
    loaded = LeRobotDataset(output_path.name, root=output_path)
    assert loaded.num_episodes == 1
    assert loaded.num_frames == 30
