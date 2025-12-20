"""
Factory for creating fake LeRobot datasets for testing.

Based on _create_fake_datasets_v3.py from the original implementation.
"""
import shutil
from pathlib import Path
from typing import Union

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def create_fake_dataset(
    repo_id: str,
    root_dir: Union[str, Path],
    value_offset: float = 0.0,
    num_episodes: int = 2,
    frames_per_episode: int = 30,
) -> LeRobotDataset:
    """
    Create a fake LeRobot dataset for testing.

    Args:
        repo_id: Repository ID for the dataset
        root_dir: Root directory where the dataset will be created
        value_offset: Offset added to all values (for differentiating datasets)
        num_episodes: Number of episodes to create (default: 2)
        frames_per_episode: Number of frames per episode (default: 30)

    Returns:
        The created LeRobotDataset instance

    Dataset Structure:
        - observation.proprioception: 10-element float32 vector
        - observation.images.camera_front: 256×256×3 video (uint8)
        - action: 4-element float32 vector
        - task: String description

    Differentiation:
        - value_offset < 5: Blue-tinted images (channel 2 = 255)
        - value_offset >= 5: Red-tinted images (channel 0 = 255)
    """
    root_dir = Path(root_dir)

    # Clean up existing directory
    if root_dir.exists():
        shutil.rmtree(root_dir)

    # Define features
    obs_features = {
        "observation.proprioception": {"dtype": "float32", "shape": (10,)},
        "observation.images.camera_front": {"dtype": "video", "shape": (256, 256, 3)},
    }
    action_features = {"action": {"dtype": "float32", "shape": (4,)}}

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=root_dir,
        features={**obs_features, **action_features},
        image_writer_threads=12,
    )

    # Create frame templates
    action_frame = {"action": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) + value_offset}

    observation_frame = {
        "observation.proprioception": (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
            + value_offset
        ),
        "observation.images.camera_front": np.ones((256, 256, 3), dtype=np.uint8),
    }

    # Color differentiation based on offset
    if value_offset >= 5:
        # Red-tinted for higher offsets
        observation_frame["observation.images.camera_front"][:, :, 0] = 255
    else:
        # Blue-tinted for lower offsets
        observation_frame["observation.images.camera_front"][:, :, 2] = 255

    task_frame = {"task": "Task description placeholder."}

    # Add episodes
    for episode_idx in range(num_episodes):
        for frame_idx in range(frames_per_episode):
            frame = {
                **observation_frame.copy(),
                **action_frame.copy(),
                **task_frame,
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.finalize()
    return dataset
