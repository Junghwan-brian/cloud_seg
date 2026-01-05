"""
Cloud Segmentation Models

다양한 분할 모델 제공
- UNet (Pretrained ResNet50 백본)
- DeepLabV3+ (Pretrained ResNet50 백본)
- CDNetV1 (Pretrained ResNet50 백본)
- CDNetV2 (Pretrained ResNet50 백본)
- HRCloudNet (High-Resolution Cloud Network)
- VisionMamba (Vim) for Segmentation with optional EDL head
"""

from .modeling import (
    get_model,
    list_models,
    unet_resnet50,
    deeplabv3plus_resnet50,
    deeplabv3plus_resnet101,
    cdnetv1_resnet50,
    cdnetv2_resnet50,
    hrcloudnet,
    vim_tiny,
    vim_small,
    vim_base,
    VIM_AVAILABLE,
)
from .unet import ResNetUNet
from .cdnetv1 import CDnetV1
from .cdnetv2 import CDnetV2
from .hrcloudnet import HRcloudNet
from ._deeplab import DeepLabV3, DeepLabHead, DeepLabHeadV3Plus

# VisionMamba (optional)
if VIM_AVAILABLE:
    from .vim_seg import VimSeg, EDLLoss, edl_uncertainty


__all__ = [
    # Factory functions
    'get_model',
    'list_models',
    # UNet
    'unet_resnet50',
    'ResNetUNet',
    # DeepLabV3+
    'deeplabv3plus_resnet50',
    'deeplabv3plus_resnet101',
    'DeepLabV3',
    'DeepLabHead',
    'DeepLabHeadV3Plus',
    # CDNet
    'cdnetv1_resnet50',
    'cdnetv2_resnet50',
    'CDnetV1',
    'CDnetV2',
    # HRCloudNet
    'hrcloudnet',
    'HRcloudNet',
    # VisionMamba
    'vim_tiny',
    'vim_small',
    'vim_base',
    'VIM_AVAILABLE',
]
