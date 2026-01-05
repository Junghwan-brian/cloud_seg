"""
Path utilities for handling multiple NAS mount points.

Supports automatic detection of NAS paths across different servers:
- Server 1: /home/telepix_nas/junghwan/cloud_seg
- Server 2: /nas/junghwan/cloud_seg
"""

import os
from pathlib import Path
from typing import Optional

# Possible NAS base paths (ordered by priority)
NAS_BASE_PATHS = [
    '/home/telepix_nas/junghwan/cloud_seg',
    '/nas/junghwan/cloud_seg',
]

_detected_nas_base: Optional[str] = None


def detect_nas_base() -> Optional[str]:
    """
    Detect which NAS base path is available.

    Returns:
        str: The detected NAS base path, or None if not found.
    """
    global _detected_nas_base

    if _detected_nas_base is not None:
        return _detected_nas_base

    for base_path in NAS_BASE_PATHS:
        if os.path.exists(base_path):
            _detected_nas_base = base_path
            return _detected_nas_base

    return None


def get_nas_path(relative_path: str = '') -> str:
    """
    Get the full NAS path for a relative path.

    Args:
        relative_path: Path relative to the NAS base (e.g., 'vim_models/vim_t.pth')

    Returns:
        str: Full path to the resource

    Raises:
        FileNotFoundError: If no NAS base path is available
    """
    nas_base = detect_nas_base()

    if nas_base is None:
        raise FileNotFoundError(
            f"[paths] No NAS base path found. Checked:\n"
            f"  {NAS_BASE_PATHS}\n"
            "Please ensure one of these paths is mounted."
        )

    if relative_path:
        return os.path.join(nas_base, relative_path)
    return nas_base


def resolve_path(path: str) -> str:
    """
    Resolve a path that may contain NAS references.

    If the path starts with a known NAS base, it will be replaced with
    the currently available NAS base.

    Args:
        path: Path that may need resolution

    Returns:
        str: Resolved path
    """
    # Check if path starts with any known NAS base
    for nas_base in NAS_BASE_PATHS:
        if path.startswith(nas_base):
            # Replace with currently available NAS base
            relative = path[len(nas_base):].lstrip('/')
            return get_nas_path(relative)

    # Path doesn't start with NAS base, return as-is
    return path


def get_data_path(dataset_name: str) -> str:
    """
    Get the data directory path for a specific dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'l8biome', 'cloudsen12-l1c')

    Returns:
        str: Path to the dataset directory
    """
    return get_nas_path(dataset_name)


def get_pretrained_path(model_name: str) -> str:
    """
    Get the path to pretrained model weights.

    Args:
        model_name: Model filename (e.g., 'vim_t_midclstok_ft_78p3acc.pth')

    Returns:
        str: Full path to the pretrained weights
    """
    return get_nas_path(os.path.join('vim_models', model_name))


# Dataset path mappings
DATASET_PATHS = {
    'l8biome': 'l8biome_extracted/l8biome',
    'cloudsen12_l1c': 'cloudsen12-l1c',
    'cloudsen12_l2a': 'cloudsen12-l2a',
    'cloud38': '38-cloud',
    'cloud95_38': '38-cloud',
    'cloud95_95': '95-cloud',
}


def get_dataset_path(dataset_key: str) -> str:
    """
    Get the path for a dataset by its key.

    Args:
        dataset_key: Dataset identifier (e.g., 'l8biome', 'cloudsen12_l1c')

    Returns:
        str: Full path to the dataset

    Raises:
        ValueError: If dataset_key is not recognized
    """
    if dataset_key not in DATASET_PATHS:
        raise ValueError(
            f"[paths] Unknown dataset: {dataset_key}. "
            f"Available: {list(DATASET_PATHS.keys())}"
        )

    return get_nas_path(DATASET_PATHS[dataset_key])


# Pretrained model path mappings
PRETRAINED_PATHS = {
    'vim_tiny': 'vim_models/vim_t_midclstok_ft_78p3acc.pth',
    'vim_small': 'vim_models/vim_s_midclstok_ft_81p6acc.pth',
    'vim_base': 'vim_models/vim_b_midclstok_81p9acc.pth',
}


def get_vim_pretrained_path(backbone: str) -> str:
    """
    Get the path for VisionMamba pretrained weights.

    Args:
        backbone: Backbone name ('tiny', 'small', 'base')

    Returns:
        str: Full path to pretrained weights

    Raises:
        ValueError: If backbone is not recognized
    """
    key = f'vim_{backbone}'
    if key not in PRETRAINED_PATHS:
        raise ValueError(
            f"[paths] Unknown VIM backbone: {backbone}. "
            f"Available: tiny, small, base"
        )

    return get_nas_path(PRETRAINED_PATHS[key])


def print_path_info():
    """Print information about detected paths."""
    nas_base = detect_nas_base()
    print(f"[paths] NAS base path: {nas_base}")
    if nas_base:
        print(f"[paths] Available datasets:")
        for key, rel_path in DATASET_PATHS.items():
            full_path = os.path.join(nas_base, rel_path)
            exists = "✓" if os.path.exists(full_path) else "✗"
            print(f"  {exists} {key}: {full_path}")
