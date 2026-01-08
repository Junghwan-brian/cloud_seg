"""
Cloud Segmentation Models

모든 모델에 대한 통합 인터페이스를 제공합니다.
지원 모델: UNet, DeepLabV3+, CDNetV1, CDNetV2, VisionMamba, ViT
"""

from .vit_seg import ViTSeg, vit_seg_nano, vit_seg_small
from .vim_seg import VimSeg, vim_seg_tiny, vim_seg_small, vim_seg_base, EDLLoss, edl_uncertainty
import torch.nn as nn

from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet
from .unet import ResNetUNet
from .cdnetv1 import CDnetV1
from .cdnetv2 import CDnetV2
from .hrcloudnet import HRcloudNet

# VisionMamba models
VIM_AVAILABLE = True
VIM_IMPORT_ERROR = None

# ViT models


def _adapt_resnet_input(backbone, in_channels, pretrained):
    """
    ResNet 백본의 입력 채널을 변경합니다.

    Args:
        backbone: ResNet 백본 모델
        in_channels: 새로운 입력 채널 수
        pretrained: pretrained 가중치 사용 여부
    """
    if in_channels != 3:
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Average pretrained weights across channels
            pretrained_weight = old_conv.weight.data
            new_weight = pretrained_weight.mean(
                dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            new_conv.weight.data = new_weight

        backbone.conv1 = new_conv

    return backbone


def _segm_resnet(name, backbone_name, in_channels, num_classes, output_stride, pretrained_backbone):
    """ResNet 백본을 사용하는 DeepLab 모델 생성"""

    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation
    )

    # Adapt input channels
    backbone = _adapt_resnet_input(backbone, in_channels, pretrained_backbone)

    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(arch_type, backbone, in_channels, num_classes, output_stride, pretrained_backbone):
    """모델 로드 함수"""

    if backbone.startswith('resnet'):
        model = _segm_resnet(
            arch_type, backbone, in_channels, num_classes,
            output_stride=output_stride, pretrained_backbone=pretrained_backbone
        )
    else:
        raise NotImplementedError(f"Backbone {backbone} not supported")
    return model


# =============================================================================
# DeepLabV3+ Models
# =============================================================================

def deeplabv3plus_resnet50(in_channels=3, num_classes=21, output_stride=16, pretrained_backbone=True):
    """
    DeepLabV3+ with ResNet-50 backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        output_stride: 출력 stride (8 또는 16)
        pretrained_backbone: ImageNet pretrained 사용 여부
    """
    return _load_model(
        'deeplabv3plus', 'resnet50', in_channels, num_classes,
        output_stride=output_stride, pretrained_backbone=pretrained_backbone
    )


def deeplabv3plus_resnet101(in_channels=3, num_classes=21, output_stride=16, pretrained_backbone=True):
    """
    DeepLabV3+ with ResNet-101 backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        output_stride: 출력 stride (8 또는 16)
        pretrained_backbone: ImageNet pretrained 사용 여부
    """
    return _load_model(
        'deeplabv3plus', 'resnet101', in_channels, num_classes,
        output_stride=output_stride, pretrained_backbone=pretrained_backbone
    )


# =============================================================================
# UNet Models
# =============================================================================

def unet_resnet50(in_channels=3, num_classes=21, pretrained_backbone=True):
    """
    UNet with ResNet-50 backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained_backbone: ImageNet pretrained 사용 여부
    """
    return ResNetUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained_backbone
    )


# =============================================================================
# CDNet Models
# =============================================================================

def cdnetv1_resnet50(in_channels=3, num_classes=21, pretrained_backbone=True):
    """
    CDNetV1 with ResNet-50 backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained_backbone: ImageNet pretrained 사용 여부
    """
    return CDnetV1(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained_backbone
    )


def cdnetv2_resnet50(in_channels=3, num_classes=21, pretrained_backbone=True, aux=True):
    """
    CDNetV2 with ResNet-50 backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained_backbone: ImageNet pretrained 사용 여부
        aux: Auxiliary loss 사용 여부
    """
    return CDnetV2(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained_backbone,
        aux=aux
    )


# =============================================================================
# HRCloudNet Models
# =============================================================================

def hrcloudnet(in_channels=3, num_classes=21, **kwargs):
    """
    HRCloudNet (High-Resolution Cloud Network)

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
    """
    return HRcloudNet(
        in_channels=in_channels,
        num_classes=num_classes,
    )


# =============================================================================
# VisionMamba Models
# =============================================================================

def vim_tiny(in_channels=3, num_classes=21, pretrained_backbone=True,
             decoder_type='unet', head_type='standard', **kwargs):
    """
    VisionMamba Tiny for Segmentation

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained_backbone: ImageNet pretrained 사용 여부
        decoder_type: 디코더 타입 ('unet', 'deeplab')
        head_type: 헤드 타입 ('standard', 'edl')
    """
    if not VIM_AVAILABLE:
        error_msg = "[vim_tiny] VisionMamba is not available.\n"
        if VIM_IMPORT_ERROR:
            error_msg += f"Import error: {VIM_IMPORT_ERROR}\n"
        error_msg += "Install required packages: pip install einops"
        raise ImportError(error_msg)
    return VimSeg(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone='tiny',
        decoder_type=decoder_type,
        head_type=head_type,
        pretrained=pretrained_backbone,
        **kwargs
    )


def vim_small(in_channels=3, num_classes=21, pretrained_backbone=True,
              decoder_type='unet', head_type='standard', **kwargs):
    """
    VisionMamba Small for Segmentation

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained_backbone: ImageNet pretrained 사용 여부
        decoder_type: 디코더 타입 ('unet', 'deeplab')
        head_type: 헤드 타입 ('standard', 'edl')
    """
    if not VIM_AVAILABLE:
        error_msg = "[vim_small] VisionMamba is not available.\n"
        if VIM_IMPORT_ERROR:
            error_msg += f"Import error: {VIM_IMPORT_ERROR}\n"
        error_msg += "Install required packages: pip install einops"
        raise ImportError(error_msg)
    return VimSeg(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone='small',
        decoder_type=decoder_type,
        head_type=head_type,
        pretrained=pretrained_backbone,
        **kwargs
    )


def vim_base(in_channels=3, num_classes=21, pretrained_backbone=True,
             decoder_type='unet', head_type='standard', **kwargs):
    """
    VisionMamba Base for Segmentation

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained_backbone: ImageNet pretrained 사용 여부
        decoder_type: 디코더 타입 ('unet', 'deeplab')
        head_type: 헤드 타입 ('standard', 'edl')
    """
    if not VIM_AVAILABLE:
        error_msg = "[vim_base] VisionMamba is not available.\n"
        if VIM_IMPORT_ERROR:
            error_msg += f"Import error: {VIM_IMPORT_ERROR}\n"
        error_msg += "Install required packages: pip install einops"
        raise ImportError(error_msg)
    return VimSeg(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone='base',
        decoder_type=decoder_type,
        head_type=head_type,
        pretrained=pretrained_backbone,
        **kwargs
    )


# =============================================================================
# Model Factory
# =============================================================================

MODELS = {
    'unet': unet_resnet50,
    'deeplabv3plus': deeplabv3plus_resnet50,
    'cdnetv1': cdnetv1_resnet50,
    'cdnetv2': cdnetv2_resnet50,
    'hrcloudnet': hrcloudnet,
    # VisionMamba models
    'vim_tiny': vim_tiny,
    'vim_small': vim_small,
    'vim_base': vim_base,
    # ViT models
    'vit_nano': vit_seg_nano,
    'vit_small': vit_seg_small,
}


def get_model(model_name, in_channels=3, num_classes=21, pretrained_backbone=True, **kwargs):
    """
    모델 팩토리 함수

    Args:
        model_name: 모델 이름 ('unet', 'deeplabv3plus', 'cdnetv1', 'cdnetv2')
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained_backbone: pretrained 백본 사용 여부
        **kwargs: 모델별 추가 인자

    Returns:
        nn.Module: 생성된 모델

    Example:
        >>> model = get_model('unet', in_channels=4, num_classes=2)
        >>> model = get_model('deeplabv3plus', in_channels=13, num_classes=4)
        >>> model = get_model('cdnetv2', in_channels=11, num_classes=4, aux=True)
    """
    model_name = model_name.lower()

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODELS.keys())}"
        )

    model_fn = MODELS[model_name]

    # CDNetV2의 경우 aux 인자 전달
    if 'cdnetv2' in model_name:
        aux = kwargs.get('aux', True)
        return model_fn(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            aux=aux
        )

    # DeepLabV3+의 경우 output_stride 인자 전달
    if 'deeplabv3plus' in model_name:
        output_stride = kwargs.get('output_stride', 16)
        return model_fn(
            in_channels=in_channels,
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone
        )

    # VisionMamba의 경우 decoder_type, head_type 인자 전달
    if model_name.startswith('vim_'):
        decoder_type = kwargs.get('decoder_type', 'unet')
        head_type = kwargs.get('head_type', 'standard')
        return model_fn(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            decoder_type=decoder_type,
            head_type=head_type,
            **{k: v for k, v in kwargs.items() if k not in ['decoder_type', 'head_type']}
        )

    # ViT의 경우 decoder_type 인자만 전달
    if model_name.startswith('vit_'):
        decoder_type = kwargs.get('decoder_type', 'unet')
        return model_fn(
            in_channels=in_channels,
            num_classes=num_classes,
            decoder_type=decoder_type,
        )

    return model_fn(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone
    )


def list_models():
    """사용 가능한 모델 목록 반환"""
    return list(MODELS.keys())
