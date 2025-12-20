"""
Command-line interface for lerobot-surgery.
"""
import logging
import sys
from pathlib import Path

import click

from .__version__ import __version__
from .v3 import merge_datasets, remove_episodes


def setup_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.group()
@click.version_option(version=__version__, prog_name="lerobot-surgery")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx, verbose):
    """
    🔬 lerobot-surgery - Precision toolkit for LeRobot dataset manipulation

    \b
    ⚠️  LeRobotDataset v3.0 Format Only
    This tool works exclusively with LeRobotDataset v3.0.

    \b
    Examples:
      lerobot-surgery merge dataset1/ dataset2/ -o merged/
      lerobot-surgery remove dataset/ 0 5 10 -o filtered/
      lerobot-surgery info dataset/
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@main.command()
@click.argument("source_datasets", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output path for the merged dataset",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite output directory if it exists",
)
@click.pass_context
def merge(ctx, source_datasets, output_path, overwrite):
    """
    Merge multiple datasets into one.

    \b
    Combines multiple LeRobot datasets with identical structures (features, fps)
    into a single consolidated dataset. Episodes are re-indexed sequentially.

    \b
    Example:
      lerobot-surgery merge dataset_a/ dataset_b/ dataset_c/ -o merged_dataset/
    """
    try:
        merged_ds = merge_datasets(
            source_paths=list(source_datasets),
            output_path=output_path,
            overwrite=overwrite,
        )
        click.echo(
            click.style(
                f"\n✓ Successfully merged {merged_ds.num_episodes} episodes",
                fg="green",
                bold=True,
            )
        )
        click.echo(f"  Output: {output_path}")
        click.echo(f"  Total frames: {merged_ds.num_frames}")
        click.echo(f"  FPS: {merged_ds.fps}")
    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg="red", bold=True), err=True)
        if ctx.obj.get("verbose"):
            raise
        sys.exit(1)


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("episodes", nargs=-1, required=True, type=int)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output path for the filtered dataset",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite output directory if it exists",
)
@click.pass_context
def remove(ctx, dataset_path, episodes, output_path, overwrite):
    """
    Remove specific episodes from a dataset.

    \b
    Creates a new dataset with the specified episodes removed.
    Remaining episodes are re-indexed sequentially starting from 0.

    \b
    Example:
      lerobot-surgery remove my_dataset/ 0 5 10 -o filtered_dataset/

    This removes episodes 0, 5, and 10 from my_dataset/.
    """
    try:
        filtered_ds = remove_episodes(
            dataset_path=dataset_path,
            episode_indices=list(episodes),
            output_path=output_path,
            overwrite=overwrite,
        )
        click.echo(
            click.style(
                f"\n✓ Successfully removed {len(episodes)} episodes",
                fg="green",
                bold=True,
            )
        )
        click.echo(f"  Output: {output_path}")
        click.echo(f"  Remaining episodes: {filtered_ds.num_episodes}")
        click.echo(f"  Total frames: {filtered_ds.num_frames}")
    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg="red", bold=True), err=True)
        if ctx.obj.get("verbose"):
            raise
        sys.exit(1)


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def info(dataset_path):
    """
    Display information about a dataset.

    \b
    Shows basic statistics and metadata about a LeRobot dataset.

    \b
    Example:
      lerobot-surgery info my_dataset/
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        dataset_path = Path(dataset_path)
        ds = LeRobotDataset(dataset_path.name, root=dataset_path)

        click.echo(click.style(f"\n📊 Dataset: {dataset_path.name}", fg="cyan", bold=True))
        click.echo(f"  Path: {dataset_path}")
        click.echo(f"  Episodes: {ds.num_episodes}")
        click.echo(f"  Total frames: {ds.num_frames}")
        click.echo(f"  FPS: {ds.fps}")
        click.echo(f"\n  Features:")
        for feature_name, feature_info in ds.features.items():
            click.echo(f"    - {feature_name}: {feature_info}")

        if hasattr(ds.meta, "camera_keys") and ds.meta.camera_keys:
            click.echo(f"\n  Camera keys: {', '.join(ds.meta.camera_keys)}")

    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg="red", bold=True), err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
