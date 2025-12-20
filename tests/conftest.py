"""
Pytest configuration and shared fixtures for lerobot-surgery tests.
"""
import tempfile
from pathlib import Path

import pytest

from tests.fixtures.dataset_factory import create_fake_dataset


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dataset_a(temp_dir):
    """Create a fake dataset 'dataset_a' with blue-tinted images."""
    return create_fake_dataset(
        repo_id="dataset_a",
        root_dir=temp_dir / "dataset_a",
        value_offset=0.0,
        num_episodes=2,
        frames_per_episode=30,
    )


@pytest.fixture
def dataset_b(temp_dir):
    """Create a fake dataset 'dataset_b' with red-tinted images."""
    return create_fake_dataset(
        repo_id="dataset_b",
        root_dir=temp_dir / "dataset_b",
        value_offset=10.0,
        num_episodes=2,
        frames_per_episode=30,
    )


@pytest.fixture
def dataset_c(temp_dir):
    """Create a third fake dataset for multi-dataset tests."""
    return create_fake_dataset(
        repo_id="dataset_c",
        root_dir=temp_dir / "dataset_c",
        value_offset=20.0,
        num_episodes=3,
        frames_per_episode=25,
    )


@pytest.fixture
def output_path(temp_dir):
    """Provide a clean output path for test results."""
    return temp_dir / "output"
