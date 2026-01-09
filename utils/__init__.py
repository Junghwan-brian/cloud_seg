"""
Utility modules for cloud segmentation.
"""

from .paths import (
    detect_nas_base,
    get_nas_path,
    resolve_path,
    get_data_path,
    get_pretrained_path,
    get_dataset_path,
    get_vim_pretrained_path,
    print_path_info,
    NAS_BASE_PATHS,
    DATASET_PATHS,
    PRETRAINED_PATHS,
)

from .losses import (
    FocalLoss,
    DiceLoss,
    CombinedLoss,
    OHEMLoss,
    ClassBalancedLoss,
    get_class_weights,
    get_loss_function,
)

__all__ = [
    # paths
    'detect_nas_base',
    'get_nas_path',
    'resolve_path',
    'get_data_path',
    'get_pretrained_path',
    'get_dataset_path',
    'get_vim_pretrained_path',
    'print_path_info',
    'NAS_BASE_PATHS',
    'DATASET_PATHS',
    'PRETRAINED_PATHS',
    # losses
    'FocalLoss',
    'DiceLoss',
    'CombinedLoss',
    'OHEMLoss',
    'ClassBalancedLoss',
    'get_class_weights',
    'get_loss_function',
]

