# 🔬 lerobot-surgery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **⚠️ IMPORTANT: LeRobotDataset v3.0 Only**
> It may work with future v3.x versions, but the code has only been tested on v3.0 so far.
> For v2.1 datasets, migrate to v3.0 first.

Dataset surgery utils (e.g., merge, episode remove) for LeRobotDataset v3.

---

## ✨ Features

- 🔀 **Merge** multiple datasets seamlessly into a single dataset
- ✂️ **Remove** specific episodes with automatic re-indexing
- 🖥️ **CLI & Python API** for flexible usage

---

## 📋 Requirements

- **Python**: 3.8 or higher
- **LeRobot**: LeRobotDataset version 3.0
- **Dependencies**: Automatically installed with the package

---

## 🚀 Installation

### From PyPI

```bash
pip install lerobot-surgery
```

### From source

```bash
git clone https://github.com/spiglerg/lerobot-surgery.git
cd lerobot-surgery
pip install -e .
```

### Development installation

```bash
git clone https://github.com/spiglerg/lerobot-surgery.git
cd lerobot-surgery
pip install -e ".[dev]"
```

### Verify installation:

```bash
lerobot-surgery --version
# Output: lerobot-surgery, version 0.1.0

lerobot-surgery --help
# Shows CLI help
```

---

## 📖 Quick Start

### Command Line Interface

```bash
# Merge multiple datasets
lerobot-surgery merge dataset_a/ dataset_b/ dataset_c/ -o merged_dataset/

# Remove specific episodes (by index)
lerobot-surgery remove my_dataset/ 0 5 10 -o filtered_dataset/

# Display dataset information
lerobot-surgery info my_dataset/

# Show help
lerobot-surgery --help
```

### Python API

```python
from lerobot_surgery import merge_datasets, remove_episodes

# Merge datasets
merged = merge_datasets(
    source_paths=["dataset_a/", "dataset_b/", "dataset_c/"],
    output_path="merged_dataset/",
)
print(f"Merged {merged.num_episodes} episodes, {merged.num_frames} frames")

# Remove episodes
filtered = remove_episodes(
    dataset_path="my_dataset/",
    episode_indices=[0, 5, 10],  # Episodes to remove
    output_path="filtered_dataset/",
)
print(f"Remaining: {filtered.num_episodes} episodes")
```

---

## 📚 Detailed Usage

### Merging Datasets

Combine multiple LeRobot datasets with identical structures (features, fps) into a single consolidated dataset.

**Requirements:**
- All datasets must have the same FPS
- All datasets must have identical features
- Datasets must be in v3.0 format

**Episode Re-indexing:**
Episodes are automatically re-indexed sequentially starting from 0 in the merged dataset.


### Removing Episodes

Remove specific episodes from a dataset while maintaining data integrity and re-indexing remaining episodes.

**Episode Re-indexing:**
Remaining episodes are automatically re-indexed sequentially starting from 0.



## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=lerobot_surgery --cov-report=html

# Run specific test file
pytest tests/test_merge.py -v
```




## 🙏 Acknowledgments

Built for the [LeRobot](https://github.com/huggingface/lerobot) framework.
