"""
Merge multiple LeRobot v3.0 format datasets into a single dataset.
"""
import logging
import shutil
from pathlib import Path
from typing import List, Union

import torch
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES, write_stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


def merge_datasets(
    source_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> LeRobotDataset:
    """
    Merge multiple LeRobot datasets into a single dataset.

    This function combines multiple datasets with identical structures (features, fps)
    into a single consolidated dataset. Episodes are re-indexed sequentially starting
    from 0.

    Args:
        source_paths: List of paths to source LeRobot datasets to merge
        output_path: Path where the merged dataset will be created
        overwrite: If True, remove existing output directory. Default: False

    Returns:
        The merged LeRobotDataset instance

    Raises:
        ValueError: If source_paths is empty or datasets are incompatible
        FileExistsError: If output_path exists and overwrite=False
        RuntimeError: If datasets cannot be loaded (likely not v3.0 format)

    Example:
        >>> from lerobot_surgery.v3 import merge_datasets
        >>> merged = merge_datasets(
        ...     source_paths=["dataset_a/", "dataset_b/"],
        ...     output_path="merged_dataset/",
        ... )
        >>> print(f"Merged dataset has {merged.num_episodes} episodes")
    """
    output_path = Path(output_path)
    source_paths = [Path(p) for p in source_paths]

    # Validation
    if not source_paths:
        raise ValueError("source_paths cannot be empty")

    if output_path.exists():
        if overwrite:
            logger.info(f"Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output path {output_path} already exists. Use overwrite=True to replace it."
            )

    # Load source datasets
    logger.info(f"Loading {len(source_paths)} source datasets...")
    try:
        source_datasets = [
            LeRobotDataset(path.name, root=path) for path in source_paths
        ]
    except Exception as e:
        raise RuntimeError(
            f"Failed to load source datasets. Ensure they are valid LeRobot v3.0 format datasets. "
            f"Error: {e}"
        ) from e

    # Get features and fps from the first dataset
    base_ds = source_datasets[0]
    features = base_ds.features
    fps = base_ds.fps

    # Validate compatibility
    for i, ds in enumerate(source_datasets[1:], start=1):
        if ds.fps != fps:
            raise ValueError(
                f"Dataset {source_paths[i]} has fps={ds.fps}, expected {fps}"
            )
        if ds.features != features:
            raise ValueError(
                f"Dataset {source_paths[i]} has incompatible features. "
                f"Expected: {features}, Got: {ds.features}"
            )

    # Create merged dataset
    logger.info(f"Creating merged dataset at {output_path}")
    merged_ds = LeRobotDataset.create(
        repo_id=output_path.name,
        root=output_path,
        features=features,
        fps=fps,
    )

    # Count total episodes for progress bar
    total_episodes = sum(ds.num_episodes for ds in source_datasets)
    logger.info(f"Merging {total_episodes} episodes from {len(source_datasets)} datasets")

    # Merge all datasets
    episode_counter = 0
    with tqdm(total=total_episodes, desc="Merging episodes", unit="ep") as pbar:
        for ds_idx, ds in enumerate(source_datasets):
            if ds.num_episodes == 0:
                logger.warning(f"Dataset {source_paths[ds_idx]} is empty, skipping")
                continue

            for ep_idx in range(ds.num_episodes):
                ep_info = ds.meta.episodes[ep_idx]
                start_frame = ep_info["dataset_from_index"]
                end_frame = ep_info["dataset_to_index"]
                num_frames = end_frame - start_frame

                # Process frames with nested progress bar
                with tqdm(
                    total=num_frames,
                    desc=f"  Episode {episode_counter}",
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

                        merged_ds.add_frame(frame_to_add)
                        frame_pbar.update(1)

                merged_ds.save_episode()
                episode_counter += 1
                pbar.update(1)

    # Aggregate and save statistics
    logger.info("Aggregating statistics...")
    all_stats = [ds.meta.stats for ds in source_datasets if ds.meta.stats is not None]
    if all_stats:
        merged_stats = aggregate_stats(all_stats)
        write_stats(merged_stats, output_path)
    else:
        logger.warning("No statistics available from source datasets")

    # Finalize the merged dataset
    logger.info("Finalizing merged dataset...")
    merged_ds.finalize()

    logger.info(
        f"✓ Successfully merged {total_episodes} episodes into {output_path}"
    )
    return merged_ds
