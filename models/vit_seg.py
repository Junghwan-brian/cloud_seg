"""
Vision Transformer (ViT) for Semantic Segmentation

Features:
- ViT backbone (nano, small variants)
- Flexible input channels adaptation
- Selectable decoder types (UNet, DeepLab)

Reference:
- ViT: https://arxiv.org/abs/2010.11929
- DeiT: https://arxiv.org/abs/2012.12877
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, List

from timm.models.layers import trunc_normal_, DropPath, to_2tuple


# ============================================================================
# ViT Components
# ============================================================================

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Flexible size handling for segmentation
        grid_h = H // self.patch_size[0]
        grid_w = W // self.patch_size[1]

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, (grid_h, grid_w)


class Attention(nn.Module):
    """Multi-head Self Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP as used in Vision Transformer"""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# ViT Encoder (Backbone)
# ============================================================================

class VisionTransformerEncoder(nn.Module):
    """Vision Transformer Encoder for Segmentation"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=None,
        feature_indices=None,
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            qkv_bias: Enable bias for qkv projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            norm_layer: Normalization layer
            feature_indices: Indices of layers to extract features from
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Default feature indices for 4-stage features
        if feature_indices is None:
            feature_indices = [depth // 4 - 1, depth // 2 - 1,
                               3 * depth // 4 - 1, depth - 1]
        self.feature_indices = feature_indices

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_channels, embed_dim=embed_dim)
        self.patch_size = patch_size

        # Position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_embed(self, x, grid_size):
        """Interpolate position embedding for different input sizes"""
        N = x.shape[1]  # number of patches
        npatch = grid_size[0] * grid_size[1]

        if npatch == self.patch_embed.num_patches:
            return self.pos_embed

        # Separate cls token and patch embeddings
        cls_pos = self.pos_embed[:, 0:1]
        patch_pos = self.pos_embed[:, 1:]

        # Reshape to 2D
        dim = patch_pos.shape[-1]
        orig_size = int(math.sqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)

        # Interpolate
        patch_pos = F.interpolate(
            patch_pos, size=grid_size, mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        original_size = (H, W)

        # Patch embedding
        x, grid_size = self.patch_embed(x)

        # Add cls token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Interpolate position embedding if needed
        pos_embed = self.interpolate_pos_embed(x, grid_size)
        x = x + pos_embed
        x = self.pos_drop(x)

        # Collect multi-scale features
        features = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.feature_indices:
                # Remove cls token and reshape to 2D feature map
                feat = x[:, 1:].permute(0, 2, 1).reshape(
                    B, self.embed_dim, grid_size[0], grid_size[1])
                features.append(feat)

        x = self.norm(x)

        return features, original_size


# ============================================================================
# Decoders (same as vim_seg.py)
# ============================================================================

class UNetDecoder(nn.Module):
    """UNet-style decoder for ViT"""

    def __init__(self, embed_dim, num_classes):
        super().__init__()

        # All features have same dimension (embed_dim)
        self.up4 = self._make_up_block(embed_dim, 512)
        self.up3 = self._make_up_block(512 + embed_dim, 256)
        self.up2 = self._make_up_block(256 + embed_dim, 128)
        self.up1 = self._make_up_block(128 + embed_dim, 64)

        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        self._init_weights()

    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, target_size):
        f1, f2, f3, f4 = features  # From shallow to deep

        x = self.up4(f4)
        x = torch.cat([x, F.interpolate(f3, size=x.shape[2:],
                      mode='bilinear', align_corners=False)], dim=1)
        x = self.up3(x)
        x = torch.cat([x, F.interpolate(f2, size=x.shape[2:],
                      mode='bilinear', align_corners=False)], dim=1)
        x = self.up2(x)
        x = torch.cat([x, F.interpolate(f1, size=x.shape[2:],
                      mode='bilinear', align_corners=False)], dim=1)
        x = self.up1(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=target_size,
                          mode='bilinear', align_corners=False)
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        out_channels = 256

        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )]

        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        res = []
        for conv in self.convs:
            out = conv(x)
            if out.shape[-2:] != size:
                out = F.interpolate(
                    out, size=size, mode='bilinear', align_corners=False)
            res.append(out)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabDecoder(nn.Module):
    """DeepLabV3+ style decoder for ViT"""

    def __init__(self, embed_dim, num_classes, aspp_dilate=[6, 12, 18]):
        super().__init__()

        # ASPP module
        self.aspp = ASPP(embed_dim, aspp_dilate)

        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(embed_dim, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, target_size):
        low_level = features[0]  # Early features
        high_level = features[-1]  # Deep features

        # ASPP
        x = self.aspp(high_level)

        # Upsample and concatenate
        x = F.interpolate(
            x, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        low_level = self.low_level_proj(low_level)
        x = torch.cat([x, low_level], dim=1)

        # Decode
        x = self.decoder(x)
        x = F.interpolate(x, size=target_size,
                          mode='bilinear', align_corners=False)
        return x


# ============================================================================
# ViT Segmentation Model
# ============================================================================

class ViTSeg(nn.Module):
    """
    Vision Transformer for Semantic Segmentation

    Combines ViT encoder with UNet or DeepLab decoder.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=21,
        backbone='small',
        decoder_type='unet',
        img_size=224,
        drop_rate=0.,
        drop_path_rate=0.1,
    ):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            backbone: ViT backbone variant ('nano', 'small')
            decoder_type: Decoder type ('unet', 'deeplab')
            img_size: Default image size (for position embedding)
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.decoder_type = decoder_type

        # ViT configurations
        configs = {
            'nano': {
                'embed_dim': 192,
                'depth': 12,
                'num_heads': 3,
            },
            'small': {
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 6,
            },
        }

        if backbone not in configs:
            raise ValueError(f"Unknown backbone: {backbone}. "
                           f"Available: {list(configs.keys())}")

        cfg = configs[backbone]
        embed_dim = cfg['embed_dim']

        # Encoder
        self.encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=16,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Decoder
        if decoder_type == 'unet':
            self.decoder = UNetDecoder(embed_dim, num_classes)
        elif decoder_type == 'deeplab':
            self.decoder = DeepLabDecoder(embed_dim, num_classes)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def forward(self, x):
        features, original_size = self.encoder(x)
        output = self.decoder(features, original_size)
        return output


# ============================================================================
# Model Factory Functions
# ============================================================================

def vit_seg_nano(in_channels=3, num_classes=21, decoder_type='unet', **kwargs):
    """
    ViT-Nano for Segmentation

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        decoder_type: Decoder type ('unet', 'deeplab')
    """
    return ViTSeg(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone='nano',
        decoder_type=decoder_type,
        **kwargs
    )


def vit_seg_small(in_channels=3, num_classes=21, decoder_type='unet', **kwargs):
    """
    ViT-Small for Segmentation

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        decoder_type: Decoder type ('unet', 'deeplab')
    """
    return ViTSeg(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone='small',
        decoder_type=decoder_type,
        **kwargs
    )
