"""
Example: Basic dataset merging

This example shows how to merge multiple LeRobot datasets using the Python API.
"""
from pathlib import Path

from lerobot_surgery import merge_datasets

# Example dataset paths (update these to your actual dataset paths)
DATASET_PATHS = [
    "path/to/dataset_a/",
    "path/to/dataset_b/",
    "path/to/dataset_c/",
]

OUTPUT_PATH = "merged_dataset/"


def main():
    """Merge multiple datasets into one."""
    print("🔬 lerobot-surgery: Dataset Merge Example\n")

    # Check if datasets exist
    missing = [p for p in DATASET_PATHS if not Path(p).exists()]
    if missing:
        print(f"⚠️  Missing datasets: {missing}")
        print("Please update DATASET_PATHS in this script to point to valid datasets.")
        return

    print(f"Merging {len(DATASET_PATHS)} datasets:")
    for path in DATASET_PATHS:
        print(f"  - {path}")
    print(f"\nOutput: {OUTPUT_PATH}\n")

    # Merge datasets
    try:
        merged = merge_datasets(
            source_paths=DATASET_PATHS,
            output_path=OUTPUT_PATH,
            overwrite=True,  # Overwrite if output exists
        )

        print(f"\n✓ Success!")
        print(f"  Episodes: {merged.num_episodes}")
        print(f"  Frames: {merged.num_frames}")
        print(f"  FPS: {merged.fps}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
