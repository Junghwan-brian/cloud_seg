"""
CDNetV2: CNN-Based Cloud Detection Network V2

Pretrained ResNet50 백본을 사용하는 CDNetV2 구현
원본 논문: "CDnetV2: CNN-Based Cloud Detection for Remote Sensing Imagery with Cloud-Snow Coexistence"
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


class _CARM(nn.Module):
    """Channel Attention Refinement Module"""

    def __init__(self, in_planes, ratio=8):
        super(_CARM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1_1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc1_2 = nn.Linear(in_planes // ratio, in_planes)

        self.fc2_1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc2_2 = nn.Linear(in_planes // ratio, in_planes)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc1_2(self.relu(self.fc1_1(avg_out)))

        max_out = self.max_pool(x)
        max_out = max_out.view(max_out.size(0), -1)
        max_out = self.fc2_2(self.relu(self.fc2_1(max_out)))

        max_out_size = max_out.size()[1]
        avg_out = torch.reshape(avg_out, (-1, max_out_size, 1, 1))
        max_out = torch.reshape(max_out, (-1, max_out_size, 1, 1))

        out = self.sigmoid(avg_out + max_out)
        return out * x


class FSFB_CH(nn.Module):
    """Feature Selection Fusion Block - Channel"""

    def __init__(self, in_planes, num, ratio=8):
        super(FSFB_CH, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1_1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc1_2 = nn.Linear(in_planes // ratio, num * in_planes)

        self.fc2_1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc2_2 = nn.Linear(in_planes // ratio, num * in_planes)
        self.relu = nn.ReLU(True)

        self.fc3 = nn.Linear(num * in_planes, 2 * num * in_planes)
        self.fc4 = nn.Linear(2 * num * in_planes, 2 * num * in_planes)
        self.fc5 = nn.Linear(2 * num * in_planes, num * in_planes)

        self.softmax = nn.Softmax(dim=3)

    def forward(self, x, num):
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc1_2(self.relu(self.fc1_1(avg_out)))

        max_out = self.max_pool(x)
        max_out = max_out.view(max_out.size(0), -1)
        max_out = self.fc2_2(self.relu(self.fc2_1(max_out)))

        out = avg_out + max_out
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.relu(self.fc5(out))

        out_size = out.size()[1]
        out = torch.reshape(out, (-1, out_size // num, 1, num))
        out = self.softmax(out)

        channel_scale = torch.chunk(out, num, dim=3)
        return channel_scale


class FSFB_SP(nn.Module):
    """Feature Selection Fusion Block - Spatial"""

    def __init__(self, num, norm_layer=nn.BatchNorm2d):
        super(FSFB_SP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 2 * num, kernel_size=3, padding=1, bias=False),
            norm_layer(2 * num),
            nn.ReLU(True),
            nn.Conv2d(2 * num, 4 * num, kernel_size=3, padding=1, bias=False),
            norm_layer(4 * num),
            nn.ReLU(True),
            nn.Conv2d(4 * num, 4 * num, kernel_size=3, padding=1, bias=False),
            norm_layer(4 * num),
            nn.ReLU(True),
            nn.Conv2d(4 * num, 2 * num, kernel_size=3, padding=1, bias=False),
            norm_layer(2 * num),
            nn.ReLU(True),
            nn.Conv2d(2 * num, num, kernel_size=3, padding=1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, num):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.softmax(x)
        spatial_scale = torch.chunk(x, num, dim=1)
        return spatial_scale


class _HFFM(nn.Module):
    """Hierarchical Feature Fusion Module"""

    def __init__(self, in_channels, atrous_rates, norm_layer=nn.BatchNorm2d):
        super(_HFFM, self).__init__()
        out_channels = 256

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = _AsppPooling(in_channels, out_channels,
                               norm_layer=norm_layer)

        self.carm = _CARM(in_channels)
        self.sa = FSFB_SP(4, norm_layer)
        self.ca = FSFB_CH(out_channels, 4, 8)

    def forward(self, x, num):
        x = self.carm(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        feat = feat1 + feat2 + feat3 + feat4

        spatial_atten = self.sa(feat, num)
        channel_atten = self.ca(feat, num)

        feat_ca = (channel_atten[0] * feat1 + channel_atten[1] * feat2 +
                   channel_atten[2] * feat3 + channel_atten[3] * feat4)
        feat_sa = (spatial_atten[0] * feat1 + spatial_atten[1] * feat2 +
                   spatial_atten[2] * feat3 + spatial_atten[3] * feat4)
        feat_sa = feat_sa + feat_ca

        return feat_sa


class _AFFM(nn.Module):
    """Adaptive Feature Fusion Module"""

    def __init__(self, in_channels=256, norm_layer=nn.BatchNorm2d):
        super(_AFFM, self).__init__()
        self.sa = FSFB_SP(2, norm_layer)
        self.ca = FSFB_CH(in_channels, 2, 8)
        self.carm = _CARM(in_channels)

    def forward(self, feat1, feat2, hffm, num):
        feat = feat1 + feat2
        spatial_atten = self.sa(feat, num)
        channel_atten = self.ca(feat, num)

        feat_ca = channel_atten[0] * feat1 + channel_atten[1] * feat2
        feat_sa = spatial_atten[0] * feat1 + spatial_atten[1] * feat2
        output = self.carm(feat_sa + feat_ca + hffm)

        return output, channel_atten, spatial_atten


class block_Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(block_Conv3x3, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class CDnetV2(nn.Module):
    """
    CDNetV2 with Pretrained ResNet50 Backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained: ImageNet pretrained 백본 사용 여부
        aux: Auxiliary loss 사용 여부
    """

    def __init__(self, in_channels=3, num_classes=21, pretrained=True, aux=True):
        super(CDnetV2, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.aux = aux

        # Load pretrained ResNet50 backbone
        backbone = resnet.resnet50(
            pretrained=pretrained,
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

        # Use layer1 for low-level features
        self.layer1 = backbone.layer1  # 256 channels, 1/4
        self.layer2 = backbone.layer2  # 512 channels, 1/8 or 1/4 with dilation
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

        # Feature adaptation layers
        self.con_layer1 = block_Conv3x3(256, 256)
        self.con_res2 = block_Conv3x3(256, 256)
        self.con_res3 = block_Conv3x3(512, 256)
        self.con_res4 = block_Conv3x3(1024, 256)
        self.con_res5 = block_Conv3x3(2048, 256)

        # HFFM and AFFM modules
        self.hffm = _HFFM(2048, [6, 12, 18])
        self.affm_1 = _AFFM()
        self.affm_2 = _AFFM()
        self.affm_3 = _AFFM()
        self.affm_4 = _AFFM()

        # Output layers
        self.dsn1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.dsn2 = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.con_layer1, self.con_res2, self.con_res3, self.con_res4, self.con_res5,
                  self.hffm, self.affm_1, self.affm_2, self.affm_3, self.affm_4,
                  self.dsn1, self.dsn2]:
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
        input_size = x.size()[2:]

        # Encoder - Initial convolution
        x = self.input_conv(x)  # 1/2
        x = self.maxpool(x)     # 1/4

        # Layer 1 features
        layer1_feat = self.layer1(x)  # 256 channels, 1/4
        layer1_0 = self.con_layer1(layer1_feat)
        size_layer1_0 = layer1_0.size()[2:]

        # Layer 2 features
        layer2_feat = self.layer2(layer1_feat)  # 512 channels
        res2 = self.con_res2(layer1_feat)  # From layer1 for skip connection
        res3 = self.con_res3(layer2_feat)
        size_res2 = res2.size()[2:]

        # Layer 3 and 4 features
        layer3_feat = self.layer3(layer2_feat)  # 1024 channels
        res4 = self.con_res4(layer3_feat)

        layer4_feat = self.layer4(layer3_feat)  # 2048 channels
        res5 = self.con_res5(layer4_feat)

        # HFFM
        hffm = self.hffm(layer4_feat, 4)
        res5 = res5 + hffm
        aux_feature = res5

        # AFFM cascade
        res5, _, _ = self.affm_1(res4, res5, hffm, 2)
        res5, _, _ = self.affm_2(res3, res5, hffm, 2)

        res5 = F.interpolate(
            res5, size_res2, mode='bilinear', align_corners=True)
        res5, _, _ = self.affm_3(
            res2, res5,
            F.interpolate(hffm, size_res2, mode='bilinear',
                          align_corners=True),
            2
        )

        res5 = F.interpolate(res5, size_layer1_0,
                             mode='bilinear', align_corners=True)
        res5, _, _ = self.affm_4(
            layer1_0, res5,
            F.interpolate(hffm, size_layer1_0,
                          mode='bilinear', align_corners=True),
            2
        )

        # Output
        output = self.dsn1(res5)
        output = F.interpolate(
            output, input_size, mode='bilinear', align_corners=True)

        if self.training and self.aux:
            auxout = self.dsn2(aux_feature)
            auxout = F.interpolate(
                auxout, input_size, mode='bilinear', align_corners=True)
            return output, auxout

        return output


def cdnetv2(in_channels=3, num_classes=21, pretrained=True, aux=True):
    """
    CDNetV2 with pretrained ResNet50 backbone

    Args:
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        pretrained: ImageNet pretrained 사용 여부
        aux: Auxiliary loss 사용 여부
    """
    return CDnetV2(in_channels=in_channels, num_classes=num_classes,
                   pretrained=pretrained, aux=aux)
