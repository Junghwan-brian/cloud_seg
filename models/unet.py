"""
UNet with Pretrained ResNet50 Backbone for Cloud Segmentation

ResNet50을 인코더(백본)로 사용하는 UNet 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet


class ConvBlock(nn.Module):
    """Double Convolution Block"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling Block with Skip Connection"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.size() != skip.size():
            x = F.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """
    UNet with Pretrained ResNet50 Backbone

    ResNet50의 각 stage 출력을 skip connection으로 사용하는 UNet 구조

    Args:
        in_channels: 입력 이미지 채널 수
        num_classes: 출력 클래스 수
        pretrained: ImageNet pretrained 가중치 사용 여부
    """

    def __init__(self, in_channels=3, num_classes=21, pretrained=True):
        super(ResNetUNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load pretrained ResNet50
        backbone = resnet.resnet50(pretrained=pretrained)

        # Input adaptation layer for non-3-channel inputs
        if in_channels != 3:
            self.input_conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            # Initialize from pretrained weights if possible
            if pretrained:
                # Conv1 가중치 적응: 채널 평균을 새 채널 수만큼 반복
                pretrained_weight = backbone.conv1.weight.data
                new_weight = pretrained_weight.mean(
                    dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                self.input_conv[0].weight.data = new_weight
                
                # BatchNorm1 가중치 복사 (중요!)
                self.input_conv[1].weight.data = backbone.bn1.weight.data.clone()
                self.input_conv[1].bias.data = backbone.bn1.bias.data.clone()
                self.input_conv[1].running_mean = backbone.bn1.running_mean.clone()
                self.input_conv[1].running_var = backbone.bn1.running_var.clone()
                self.input_conv[1].num_batches_tracked = backbone.bn1.num_batches_tracked.clone()
        else:
            self.input_conv = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu
            )

        self.maxpool = backbone.maxpool

        # Encoder (ResNet layers)
        self.encoder1 = backbone.layer1  # 256 channels, 1/4
        self.encoder2 = backbone.layer2  # 512 channels, 1/8
        self.encoder3 = backbone.layer3  # 1024 channels, 1/16
        self.encoder4 = backbone.layer4  # 2048 channels, 1/32

        # Decoder
        self.up4 = UpBlock(2048, 1024, 512)  # 1/16
        self.up3 = UpBlock(512, 512, 256)    # 1/8
        self.up2 = UpBlock(256, 256, 128)    # 1/4
        self.up1 = UpBlock(128, 64, 64)      # 1/2

        # Skip connection from input conv
        self.skip_conv = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final upsampling and classification
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Initialize decoder weights"""
        for m in [self.up4, self.up3, self.up2, self.up1,
                  self.skip_conv, self.final_up, self.final_conv]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]

        # Initial convolution (1/2)
        x0 = self.input_conv(x)
        skip0 = self.skip_conv(x0)

        # Encoder path
        x1 = self.maxpool(x0)
        x1 = self.encoder1(x1)  # 1/4, 256 channels

        x2 = self.encoder2(x1)  # 1/8, 512 channels
        x3 = self.encoder3(x2)  # 1/16, 1024 channels
        x4 = self.encoder4(x3)  # 1/32, 2048 channels

        # Decoder path with skip connections
        d4 = self.up4(x4, x3)   # 1/16
        d3 = self.up3(d4, x2)   # 1/8
        d2 = self.up2(d3, x1)   # 1/4
        d1 = self.up1(d2, skip0)  # 1/2

        # Final upsampling and classification
        out = self.final_up(d1)  # 1/1
        out = self.final_conv(out)

        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size,
                                mode='bilinear', align_corners=False)

        return out


def unet_resnet50(in_channels=3, num_classes=21, pretrained=True):
    """
    UNet with ResNet50 backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained: ImageNet pretrained 사용 여부
    """
    return ResNetUNet(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
