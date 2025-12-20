"""
Remove specific episodes from a LeRobot v3.0 format dataset.
"""
import logging
import shutil
from pathlib import Path
from typing import List, Set, Union

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from tqdm import tqdm

logger = logging.getLogger(__name__)


def remove_episodes(
    dataset_path: Union[str, Path],
    episode_indices: List[int],
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> LeRobotDataset:
    """
    Remove specified episodes from a LeRobot dataset.

    Creates a new dataset containing only the episodes that were not removed.
    Remaining episodes are re-indexed sequentially starting from 0.

    Args:
        dataset_path: Path to the source LeRobot dataset
        episode_indices: List of episode indices to remove
        output_path: Path where the filtered dataset will be created
        overwrite: If True, remove existing output directory. Default: False

    Returns:
        The new LeRobotDataset instance with episodes removed

    Raises:
        ValueError: If episode_indices contains invalid indices or would remove all episodes
        FileExistsError: If output_path exists and overwrite=False
        RuntimeError: If dataset cannot be loaded (likely not v3.0 format)

    Example:
        >>> from lerobot_surgery.v3 import remove_episodes
        >>> filtered = remove_episodes(
        ...     dataset_path="my_dataset/",
        ...     episode_indices=[0, 5, 10],  # Remove these episodes
        ...     output_path="filtered_dataset/",
        ... )
        >>> print(f"Filtered dataset has {filtered.num_episodes} episodes")
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    episode_indices_set: Set[int] = set(episode_indices)

    # Validation
    if output_path.exists():
        if overwrite:
            logger.info(f"Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output path {output_path} already exists. Use overwrite=True to replace it."
            )

    # Load source dataset
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        ds = LeRobotDataset(dataset_path.name, root=dataset_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset. Ensure it is a valid LeRobot v3.0 format dataset. "
            f"Error: {e}"
        ) from e

    # Determine which episodes to keep
    all_episodes = set(range(ds.num_episodes))
    invalid_episodes = episode_indices_set - all_episodes

    if invalid_episodes:
        raise ValueError(
            f"Invalid episode indices {sorted(invalid_episodes)}. "
            f"Dataset has {ds.num_episodes} episodes (indices 0-{ds.num_episodes - 1})"
        )

    episodes_to_keep = sorted(list(all_episodes - episode_indices_set))

    if not episodes_to_keep:
        raise ValueError(
            "Cannot remove all episodes. At least one episode must remain."
        )

    logger.info(
        f"Removing {len(episode_indices_set)} episodes, "
        f"keeping {len(episodes_to_keep)} out of {ds.num_episodes} total"
    )

    # Create new dataset
    logger.info(f"Creating filtered dataset at {output_path}")
    new_ds = LeRobotDataset.create(
        repo_id=output_path.name,
        root=output_path,
        features=ds.features,
        fps=ds.fps,
    )

    # Copy episodes to keep
    with tqdm(total=len(episodes_to_keep), desc="Copying episodes", unit="ep") as pbar:
        for new_ep_idx, old_ep_idx in enumerate(episodes_to_keep):
            ep_info = ds.meta.episodes[old_ep_idx]
            start_frame = ep_info["dataset_from_index"]
            end_frame = ep_info["dataset_to_index"]
            num_frames = end_frame - start_frame

            # Process frames with nested progress bar
            with tqdm(
                total=num_frames,
                desc=f"  Episode {old_ep_idx}→{new_ep_idx}",
                unit="frame",
                leave=False,
            ) as frame_pbar:
                for frame_idx in range(start_frame, end_frame):
                    frame = ds[frame_idx]

                    # Fix image shape from (C, H, W) to (H, W, C) for validation
                    for key in ds.meta.camera_keys:
                        if key in frame and isinstance(frame[key], torch.Tensor):
                            image_tensor = frame[key]
                            frame[key] = image_tensor.permute(1, 2, 0)

                    # Remove features that are added automatically by add_frame
                    frame_to_add = {
                        k: v for k, v in frame.items() if k not in DEFAULT_FEATURES
                    }

                    new_ds.add_frame(frame_to_add)
                    frame_pbar.update(1)

            new_ds.save_episode()
            pbar.update(1)

    # Finalize the new dataset
    logger.info("Finalizing filtered dataset...")
    new_ds.finalize()

    logger.info(
        f"✓ Successfully created filtered dataset with {len(episodes_to_keep)} episodes at {output_path}"
    )
    return new_ds
