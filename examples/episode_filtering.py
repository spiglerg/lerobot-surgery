"""
Example: Episode filtering (removal)

This example shows how to remove specific episodes from a dataset.
"""
from pathlib import Path

from lerobot_surgery import remove_episodes

# Example dataset path (update this to your actual dataset path)
DATASET_PATH = "path/to/my_dataset/"

# Episodes to remove (by index)
EPISODES_TO_REMOVE = [0, 5, 10]

OUTPUT_PATH = "filtered_dataset/"


def main():
    """Remove specific episodes from a dataset."""
    print("🔬 lerobot-surgery: Episode Filtering Example\n")

    # Check if dataset exists
    if not Path(DATASET_PATH).exists():
        print(f"⚠️  Dataset not found: {DATASET_PATH}")
        print("Please update DATASET_PATH in this script to point to a valid dataset.")
        return

    print(f"Dataset: {DATASET_PATH}")
    print(f"Removing episodes: {EPISODES_TO_REMOVE}")
    print(f"Output: {OUTPUT_PATH}\n")

    # Remove episodes
    try:
        filtered = remove_episodes(
            dataset_path=DATASET_PATH,
            episode_indices=EPISODES_TO_REMOVE,
            output_path=OUTPUT_PATH,
            overwrite=True,  # Overwrite if output exists
        )

        print(f"\n✓ Success!")
        print(f"  Removed: {len(EPISODES_TO_REMOVE)} episodes")
        print(f"  Remaining: {filtered.num_episodes} episodes")
        print(f"  Total frames: {filtered.num_frames}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
