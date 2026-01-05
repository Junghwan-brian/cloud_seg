"""
CDNetV1: CNN-Based Cloud Detection Network

Pretrained ResNet50 백본을 사용하는 CDNetV1 구현
원본 논문: "CDnet: CNN-Based Cloud Detection for Remote Sensing Imagery"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer=nn.BatchNorm2d):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer=nn.BatchNorm2d):
        super(_ASPP, self).__init__()
        out_channels = 512
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = _AsppPooling(in_channels, out_channels,
                               norm_layer=norm_layer)

        self.dropout2d = nn.Dropout2d(0.3)

    def forward(self, x):
        feat1 = self.dropout2d(self.b0(x))
        feat2 = self.dropout2d(self.b1(x))
        feat3 = self.dropout2d(self.b2(x))
        feat4 = self.dropout2d(self.b3(x))
        feat5 = self.dropout2d(self.b4(x))
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        return x


class _FPM(nn.Module):
    """Feature Pyramid Module"""

    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d):
        super(_FPM, self).__init__()
        self.aspp = _ASPP(in_channels, [6, 12, 18], norm_layer=norm_layer)

    def forward(self, x):
        x = torch.cat((x, self.aspp(x)), dim=1)
        return x


class BR(nn.Module):
    """Boundary Refinement Module"""

    def __init__(self, num_classes, stride=1):
        super(BR, self).__init__()
        self.conv1 = conv3x3(num_classes, num_classes * 16, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_classes * 16, num_classes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class CDnetV1(nn.Module):
    """
    CDNetV1 with Pretrained ResNet50 Backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수  
        pretrained: ImageNet pretrained 백본 사용 여부
    """

    def __init__(self, in_channels=3, num_classes=21, pretrained=True):
        super(CDnetV1, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load pretrained ResNet50 backbone
        backbone = resnet.resnet50(
            pretrained=pretrained,
            # Use dilation in layer3, layer4
            replace_stride_with_dilation=[False, True, True]
        )

        # Input adaptation for non-3-channel inputs
        if in_channels != 3:
            self.input_conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            if pretrained:
                pretrained_weight = backbone.conv1.weight.data
                new_weight = pretrained_weight.mean(
                    dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                self.input_conv[0].weight.data = new_weight
        else:
            self.input_conv = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu
            )

        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

        # Combine layer3 and layer4 features
        self.res5_con1x1 = nn.Sequential(
            nn.Conv2d(1024 + 2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # Feature Pyramid Modules
        self.fpm1 = _FPM(512, num_classes)
        self.fpm2 = _FPM(512, num_classes)
        self.fpm3 = _FPM(256, num_classes)

        # Boundary Refinement Modules
        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)

        # Prediction layers
        self.predict1 = self._predict_layer(512 * 6, num_classes)
        self.predict2 = self._predict_layer(512 * 6, num_classes)
        self.predict3 = self._predict_layer(512 * 5 + 256, num_classes)

        self._init_weights()

    def _predict_layer(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=3,
                      stride=1, padding=1, bias=True)
        )

    def _init_weights(self):
        for m in [self.res5_con1x1, self.fpm1, self.fpm2, self.fpm3,
                  self.br1, self.br2, self.br3, self.br4, self.br5, self.br6, self.br7,
                  self.predict1, self.predict2, self.predict3]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        size = x.size()[2:]

        # Encoder
        x = self.input_conv(x)  # 1/2
        size_conv1 = x.size()[2:]
        x = self.maxpool(x)     # 1/4

        res2 = self.layer1(x)   # 1/4, 256 channels
        res3 = self.layer2(res2)  # 1/4 (dilated), 512 channels
        res4 = self.layer3(res3)  # 1/4 (dilated), 1024 channels
        res5 = self.layer4(res4)  # 1/4 (dilated), 2048 channels

        # Combine layer3 and layer4
        score1 = self.res5_con1x1(
            torch.cat([res5, res4], dim=1))  # 512 channels

        # FPM + Prediction for score1
        score1 = self.fpm1(score1)
        score1 = self.predict1(score1)
        score1 = self.br1(score1)

        # FPM + Prediction for res3
        score2 = self.fpm2(res3)
        score2 = self.predict2(score2)

        # First fusion
        score2 = self.br2(score2) + score1
        score2 = self.br3(score2)

        # FPM + Prediction for res2
        score3 = self.fpm3(res2)
        score3 = self.predict3(score3)
        score3 = self.br4(score3)

        # Second fusion
        size_score3 = score3.size()[2:]
        score3 = score3 + \
            F.interpolate(score2, size_score3,
                          mode='bilinear', align_corners=True)
        score3 = self.br5(score3)

        # Upsampling + BR
        score3 = F.interpolate(
            score3, size_conv1, mode='bilinear', align_corners=True)
        score3 = self.br6(score3)
        score3 = F.interpolate(
            score3, size, mode='bilinear', align_corners=True)
        score3 = self.br7(score3)

        return score3


def cdnetv1(in_channels=3, num_classes=21, pretrained=True):
    """
    CDNetV1 with pretrained ResNet50 backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained: ImageNet pretrained 사용 여부
    """
    return CDnetV1(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
