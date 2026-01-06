"""
Vision Mamba (Vim) for Semantic Segmentation

Features:
- Pretrained ViM backbone (tiny, small, base)
- Flexible input channels adaptation
- Selectable decoder types (UNet, DeepLab)
- Optional EDL (Evidential Deep Learning) head for uncertainty estimation

Reference:
- Vim: https://arxiv.org/abs/2401.09417
- EDL: https://arxiv.org/abs/2303.02045
"""

import math
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Dict, Any, Union
from torch import Tensor

from timm.models.layers import trunc_normal_, lecun_normal_, DropPath, to_2tuple


# ============================================================================
# Mamba Components (simplified for segmentation, no CUDA dependency for loading)
# ============================================================================

MAMBA_AVAILABLE = False
MAMBA_IMPORT_ERROR = None
RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
Mamba = None

try:
    from .mamba_ssm.modules.mamba_simple import Mamba as _Mamba
    Mamba = _Mamba
    MAMBA_AVAILABLE = True
except ImportError as e:
    MAMBA_IMPORT_ERROR = e
    warnings.warn(
        f"[VimSeg] Failed to import Mamba: {e}. VisionMamba models will not be available.")

try:
    from .mamba_ssm.ops.triton.layer_norm import RMSNorm as _RMSNorm, layer_norm_fn as _layer_norm_fn, rms_norm_fn as _rms_norm_fn
    RMSNorm, layer_norm_fn, rms_norm_fn = _RMSNorm, _layer_norm_fn, _rms_norm_fn
except ImportError:
    # Use LayerNorm as fallback - this is expected behavior
    RMSNorm = nn.LayerNorm
    layer_norm_fn, rms_norm_fn = None, None


# ============================================================================
# VisionMamba Encoder (Backbone)
# ============================================================================

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3,
                 embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1,
                          (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Flexible size handling for segmentation
        grid_h = (H - self.patch_size[0]) // self.stride + 1
        grid_w = (W - self.patch_size[1]) // self.stride + 1

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, (grid_h, grid_w)


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm,
                 fused_add_norm=False, residual_in_fp32=False, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm))

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                inference_params=None) -> Tuple[Tensor, Tensor]:
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(
                residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(
                self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states, self.norm.weight, self.norm.bias,
                    residual=residual, prenorm=True,
                    residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps)
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(
                        hidden_states), self.norm.weight, self.norm.bias,
                    residual=residual, prenorm=True,
                    residual_in_fp32=self.residual_in_fp32, eps=self.norm.eps)

        hidden_states = self.mixer(
            hidden_states, inference_params=inference_params)
        return hidden_states, residual


def create_block(d_model, d_state=16, ssm_cfg=None, norm_epsilon=1e-5, drop_path=0.,
                 rms_norm=False, residual_in_fp32=False, fused_add_norm=False,
                 layer_idx=None, device=None, dtype=None, if_bimamba=False,
                 bimamba_type="none", if_divide_out=False, init_layer_scale=None):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx,
                        bimamba_type=bimamba_type, if_divide_out=if_divide_out,
                        init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm,
                       eps=norm_epsilon, **factory_kwargs)

    block = Block(d_model, mixer_cls, norm_cls=norm_cls, drop_path=drop_path,
                  fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02,
                  rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class VisionMambaEncoder(nn.Module):
    """Vision Mamba Encoder for Segmentation"""

    def __init__(self, img_size=224, patch_size=16, stride=16, depth=24,
                 embed_dim=192, d_state=16, in_channels=3,
                 drop_rate=0., drop_path_rate=0.1, norm_epsilon=1e-5,
                 rms_norm=False, fused_add_norm=False, residual_in_fp32=True,
                 if_abs_pos_embed=True, if_cls_token=True,
                 use_middle_cls_token=True, bimamba_type="v2", if_divide_out=True,
                 return_intermediate=True, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.stride = stride
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.return_intermediate = return_intermediate
        self.num_tokens = 1 if if_cls_token else 0

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride,
            in_chans=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token
        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(
                1, num_patches + self.num_tokens, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr

        # Mamba blocks
        self.layers = nn.ModuleList([
            create_block(
                embed_dim, d_state=d_state, norm_epsilon=norm_epsilon,
                rms_norm=rms_norm, residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm, layer_idx=i,
                bimamba_type=bimamba_type, drop_path=inter_dpr[i],
                if_divide_out=if_divide_out)
            for i in range(depth)
        ])

        # Final norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # Initialize
        self.patch_embed.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(partial(_init_weights, n_layer=depth))

    def interpolate_pos_embed(self, x, grid_size):
        """Interpolate position embeddings for different input sizes"""
        if not self.if_abs_pos_embed:
            return x

        num_patches = grid_size[0] * grid_size[1]
        N = self.pos_embed.shape[1] - self.num_tokens

        if num_patches == N:
            return x + self.pos_embed

        # Interpolate
        pos_embed = self.pos_embed
        if self.num_tokens > 0:
            cls_pos_embed = pos_embed[:, :self.num_tokens]
            pos_embed = pos_embed[:, self.num_tokens:]

        # Reshape to 2D
        orig_size = int(math.sqrt(N))
        pos_embed = pos_embed.reshape(
            1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed, size=grid_size, mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(
            0, 2, 3, 1).reshape(1, -1, self.embed_dim)

        if self.num_tokens > 0:
            # Insert cls token position at middle
            if self.use_middle_cls_token:
                token_pos = num_patches // 2
                pos_embed = torch.cat([
                    pos_embed[:, :token_pos], cls_pos_embed, pos_embed[:, token_pos:]
                ], dim=1)
            else:
                pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)

        return x + pos_embed

    def forward(self, x) -> Dict[str, Tensor]:
        B = x.shape[0]

        # Patch embed
        x, grid_size = self.patch_embed(x)
        M = x.shape[1]

        # Add cls token
        if self.if_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            if self.use_middle_cls_token:
                token_position = M // 2
                x = torch.cat([x[:, :token_position], cls_token,
                              x[:, token_position:]], dim=1)
            else:
                token_position = 0
                x = torch.cat([cls_token, x], dim=1)
            M = x.shape[1]
        else:
            token_position = None

        # Add position embedding
        x = self.interpolate_pos_embed(x, grid_size)
        if self.if_abs_pos_embed:
            x = self.pos_drop(x)

        # Mamba layers with intermediate outputs
        intermediate_features = []
        residual = None
        hidden_states = x

        layer_indices = [5, 11, 17, 23] if self.depth == 24 else [self.depth // 4 - 1, self.depth // 2 - 1,
                                                                  3 * self.depth // 4 - 1, self.depth - 1]

        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual)

            if self.return_intermediate and i in layer_indices:
                # Get spatial features (exclude cls token)
                feat = hidden_states
                if self.if_cls_token:
                    if self.use_middle_cls_token:
                        feat = torch.cat(
                            [feat[:, :token_position], feat[:, token_position+1:]], dim=1)
                    else:
                        feat = feat[:, 1:]
                # Reshape to spatial
                feat = feat.transpose(1, 2).reshape(
                    B, self.embed_dim, grid_size[0], grid_size[1])
                intermediate_features.append(feat)

        # Final norm
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(
                residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(
                self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(
                    hidden_states), self.norm_f.weight, self.norm_f.bias,
                eps=self.norm_f.eps, residual=residual, prenorm=False,
                residual_in_fp32=self.residual_in_fp32)

        # Final feature map
        final_feat = hidden_states
        if self.if_cls_token:
            if self.use_middle_cls_token:
                final_feat = torch.cat([final_feat[:, :token_position],
                                       final_feat[:, token_position+1:]], dim=1)
            else:
                final_feat = final_feat[:, 1:]
        final_feat = final_feat.transpose(1, 2).reshape(
            B, self.embed_dim, grid_size[0], grid_size[1])

        return {
            'features': intermediate_features,
            'final': final_feat,
            'grid_size': grid_size
        }


# ============================================================================
# Decoders
# ============================================================================

class UNetDecoder(nn.Module):
    """UNet-style decoder for VisionMamba"""

    def __init__(self, embed_dim, num_classes, feature_channels=None):
        super().__init__()

        if feature_channels is None:
            feature_channels = [embed_dim] * 4

        # Decoder blocks
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


class DeepLabDecoder(nn.Module):
    """DeepLabV3+ style decoder for VisionMamba"""

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


# ============================================================================
# EDL (Evidential Deep Learning) Head
# ============================================================================

class EDLHead(nn.Module):
    """
    Evidential Deep Learning Head for Uncertainty Estimation

    Based on: "Uncertainty Estimation by Fisher Information-based Evidential Deep Learning"
    https://arxiv.org/abs/2303.02045

    Outputs Dirichlet concentration parameters (evidence) instead of class probabilities.
    """

    def __init__(self, in_channels, num_classes, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Returns:
            evidence: (B, C, H, W) - evidence for each class (alpha - 1)
            alpha: (B, C, H, W) - Dirichlet concentration parameters
        """
        logits = self.head(x)
        # Evidence is non-negative: use softplus or exp
        evidence = F.softplus(logits)
        # Dirichlet parameters: alpha = evidence + 1
        alpha = evidence + 1
        return {'evidence': evidence, 'alpha': alpha, 'logits': logits}


class EDLLoss(nn.Module):
    """
    Evidential Deep Learning Loss Function (Official Implementation)

    Combines:
    1. MSE Loss: (y - p)^2 where p = alpha / S
    2. Variance Loss: Var[p] under Dirichlet distribution
    3. KL Divergence Regularization with annealing

    Reference: https://github.com/deargen/MT-ENet
    """

    def __init__(self, num_classes, annealing_epochs=10, lambda_kl=1.0,
                 target_concentration=1.0, epsilon=1e-8):
        """
        Args:
            num_classes: Number of classes
            annealing_epochs: Total epochs for KL annealing (lambda_kl weight goes from 0 to 1)
            lambda_kl: KL divergence weight (default 1.0, -1.0 means use annealing only)
            target_concentration: Target concentration for KL (default 1.0 for uniform)
            epsilon: Small value for numerical stability
        """
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.lambda_kl = lambda_kl
        self.target_concentration = target_concentration
        self.epsilon = epsilon

    def forward(self, alpha: Union[Tensor, Dict[str, Tensor]], target: Tensor,
                epoch: int = 0, total_epochs: int = None,
                ignore_index: int = 255) -> Dict[str, Tensor]:
        """
        Args:
            alpha: (B, C, H, W) Dirichlet concentration parameters, or dict with 'alpha' key
            target: (B, H, W) Ground truth labels
            epoch: Current training epoch (for KL annealing)
            total_epochs: Total training epochs (for KL annealing, optional)
            ignore_index: Index to ignore in loss computation

        Returns:
            Dictionary containing total loss and individual components
        """
        # Handle dict input (from VimSeg EDL head)
        if isinstance(alpha, dict):
            alpha = alpha['alpha']

        B, C, H, W = alpha.shape

        # Reshape for computation
        alpha = alpha.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        target = target.reshape(-1)  # (B*H*W,)

        # Create mask for valid pixels
        valid_mask = target != ignore_index
        alpha = alpha[valid_mask]
        target = target[valid_mask]

        if alpha.numel() == 0:
            zero_loss = torch.tensor(
                0.0, device=alpha.device if alpha.numel() > 0 else 'cpu')
            return {'loss': zero_loss, 'mse_loss': zero_loss,
                    'var_loss': zero_loss, 'kl_loss': zero_loss}

        # One-hot encode target
        target_one_hot = F.one_hot(target.long(), num_classes=C).float()

        # Dirichlet strength (sum of alphas)
        S = alpha.sum(dim=-1, keepdim=True)  # (N, 1)

        # Expected probabilities: p = alpha / S
        p = alpha / S

        # =============================================================
        # 1. MSE Loss: (y - p)^2
        # =============================================================
        mse_loss = (target_one_hot - p).pow(2).sum(dim=-1).mean()

        # =============================================================
        # 2. Variance Loss: Var[p] = alpha * (S - alpha) / (S^2 * (S + 1))
        # =============================================================
        var_loss = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=-1).mean()

        # =============================================================
        # 3. KL Divergence Regularization
        # =============================================================
        # Adjust alpha for KL: remove evidence of correct class
        # alpha_adjusted = (alpha - target_c) * (1 - y) + target_c
        # This keeps alpha=target_c for correct class, adjusts others
        alpha_adjusted = (alpha - self.target_concentration) * \
            (1 - target_one_hot) + self.target_concentration

        # Compute KL divergence
        kl_loss = self._compute_kl_loss(
            alpha_adjusted, target,
            target_concentration=self.target_concentration,
            concentration=1.0
        )

        # Annealing coefficient
        if self.lambda_kl == -1.0:
            # Use epoch-based annealing only
            if total_epochs is not None:
                annealing_coef = epoch / total_epochs
            else:
                annealing_coef = min(
                    1.0, epoch / max(1, self.annealing_epochs))
            kl_weight = annealing_coef
        else:
            # Use fixed lambda_kl with optional annealing
            annealing_coef = min(1.0, epoch / max(1, self.annealing_epochs))
            kl_weight = annealing_coef * self.lambda_kl

        # Total loss: MSE + Variance + KL
        total_loss = mse_loss + var_loss + kl_weight * kl_loss

        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'var_loss': var_loss,
            'kl_loss': kl_loss,
            'annealing_coef': torch.tensor(annealing_coef, device=alpha.device)
        }

    def _compute_kl_loss(self, alphas: Tensor, labels: Tensor,
                         target_concentration: float = 1.0,
                         concentration: float = 1.0) -> Tensor:
        """
        Compute KL divergence loss between predicted Dirichlet and target Dirichlet.

        Args:
            alphas: (N, C) Adjusted Dirichlet parameters
            labels: (N,) Ground truth labels
            target_concentration: Concentration for correct class in target
            concentration: Base concentration for target Dirichlet

        Returns:
            KL divergence loss (scalar)
        """
        C = alphas.shape[-1]

        # Create target alphas: uniform Dirichlet with higher concentration for correct class
        target_alphas = torch.ones_like(alphas) * concentration
        target_alphas.scatter_add_(
            -1,
            labels.unsqueeze(-1),
            torch.full_like(labels.unsqueeze(-1),
                            target_concentration - 1, dtype=alphas.dtype)
        )

        return self._dirichlet_kl_divergence(alphas, target_alphas)

    def _dirichlet_kl_divergence(self, alphas: Tensor, target_alphas: Tensor) -> Tensor:
        """
        Compute KL divergence between two Dirichlet distributions.
        KL(Dir(alphas) || Dir(target_alphas))

        Args:
            alphas: (N, C) Source Dirichlet parameters
            target_alphas: (N, C) Target Dirichlet parameters

        Returns:
            Mean KL divergence (scalar)
        """
        eps = self.epsilon

        # Sum of alphas
        alp0 = alphas.sum(dim=-1, keepdim=True)  # (N, 1)
        target_alp0 = target_alphas.sum(dim=-1, keepdim=True)  # (N, 1)

        # First term: lgamma(sum(alphas)) - lgamma(sum(target_alphas))
        alp0_term = torch.lgamma(alp0 + eps) - torch.lgamma(target_alp0 + eps)
        alp0_term = torch.where(torch.isfinite(
            alp0_term), alp0_term, torch.zeros_like(alp0_term))

        # Second term: sum of individual lgamma differences and digamma terms
        alphas_term = torch.sum(
            torch.lgamma(target_alphas + eps) - torch.lgamma(alphas + eps)
            + (alphas - target_alphas) *
            (torch.digamma(alphas + eps) - torch.digamma(alp0 + eps)),
            dim=-1, keepdim=True
        )
        alphas_term = torch.where(torch.isfinite(
            alphas_term), alphas_term, torch.zeros_like(alphas_term))

        # Total KL divergence
        kl = (alp0_term + alphas_term).squeeze(-1)

        return kl.mean()


def edl_uncertainty(alpha: Tensor) -> Dict[str, Tensor]:
    """
    Compute various uncertainty measures from Dirichlet parameters

    Args:
        alpha: (B, C, H, W) Dirichlet concentration parameters

    Returns:
        Dictionary containing:
        - prob: Expected class probabilities
        - pred: Predicted class labels
        - entropy: Predictive entropy (total uncertainty)
        - data_uncertainty: Aleatoric uncertainty
        - model_uncertainty: Epistemic uncertainty (mutual information)
        - vacuity: Uncertainty due to lack of evidence
    """
    B, C, H, W = alpha.shape

    # Dirichlet strength
    S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)

    # Expected probabilities
    prob = alpha / S

    # Predictions
    pred = prob.argmax(dim=1)

    # Total uncertainty (predictive entropy)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

    # Data uncertainty (expected entropy)
    digamma_alpha = torch.digamma(alpha)
    digamma_S = torch.digamma(S)
    data_uncertainty = -torch.sum(prob * (digamma_alpha - digamma_S), dim=1)

    # Model uncertainty (mutual information = total - data)
    model_uncertainty = entropy - data_uncertainty

    # Vacuity (K / S where K is number of classes)
    vacuity = C / S.squeeze(1)

    return {
        'prob': prob,
        'pred': pred,
        'entropy': entropy,
        'data_uncertainty': data_uncertainty,
        'model_uncertainty': model_uncertainty,
        'vacuity': vacuity
    }


# ============================================================================
# Main Model: VisionMamba for Segmentation
# ============================================================================

class VimSeg(nn.Module):
    """
    Vision Mamba for Semantic Segmentation

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        backbone: Backbone size ('tiny', 'small', 'base')
        decoder_type: Type of decoder ('unet', 'deeplab')
        head_type: Type of classification head ('standard', 'edl')
        pretrained: Path to pretrained weights or boolean
        img_size: Expected input image size (for position embedding)
    """

    # Backbone configurations
    BACKBONE_CONFIGS = {
        'tiny': {'embed_dim': 192, 'depth': 24},
        'small': {'embed_dim': 384, 'depth': 24},
        'base': {'embed_dim': 768, 'depth': 24},
    }

    # Pretrained weight filenames (relative to NAS vim_models directory)
    PRETRAINED_FILENAMES = {
        'tiny': 'vim_models/vim_t_midclstok_ft_78p3acc.pth',
        'small': 'vim_models/vim_s_midclstok_ft_81p6acc.pth',
        'base': 'vim_models/vim_b_midclstok_81p9acc.pth',
    }

    @classmethod
    def get_pretrained_path(cls, backbone: str) -> Optional[str]:
        """Get pretrained weights path with automatic NAS detection."""
        if backbone not in cls.PRETRAINED_FILENAMES:
            return None

        try:
            # Import here to avoid circular imports
            import sys
            sys.path.insert(0, os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            from utils.paths import get_nas_path
            return get_nas_path(cls.PRETRAINED_FILENAMES[backbone])
        except (ImportError, FileNotFoundError) as e:
            warnings.warn(f"[VimSeg] Could not resolve pretrained path: {e}")
            return None

    def __init__(self, in_channels=3, num_classes=21, backbone='tiny',
                 decoder_type='unet', head_type='standard', pretrained=True,
                 img_size=224, **kwargs):
        super().__init__()

        if not MAMBA_AVAILABLE:
            error_msg = (
                "[VimSeg] Mamba SSM is required but not available.\n"
                "This could be due to:\n"
                "  1. Missing 'einops' package: pip install einops\n"
                "  2. Missing CUDA dependencies for mamba_ssm\n"
                "  3. Import error in mamba_ssm modules\n"
            )
            if MAMBA_IMPORT_ERROR is not None:
                error_msg += f"\nOriginal error: {MAMBA_IMPORT_ERROR}"
            raise ImportError(error_msg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.decoder_type = decoder_type
        self.head_type = head_type

        # Get backbone config
        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Unknown backbone: {backbone}. Choose from {list(self.BACKBONE_CONFIGS.keys())}")
        config = self.BACKBONE_CONFIGS[backbone]
        embed_dim = config['embed_dim']
        depth = config['depth']

        # Create encoder
        # rms_norm=True: pretrained weights가 RMSNorm으로 훈련됨
        self.encoder = VisionMambaEncoder(
            img_size=img_size, in_channels=in_channels,
            embed_dim=embed_dim, depth=depth,
            rms_norm=True, return_intermediate=True, **kwargs)

        # Create decoder
        if decoder_type == 'unet':
            self.decoder = UNetDecoder(
                embed_dim, num_classes if head_type == 'standard' else embed_dim)
        elif decoder_type == 'deeplab':
            self.decoder = DeepLabDecoder(
                embed_dim, num_classes if head_type == 'standard' else embed_dim)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # EDL head (optional)
        self.edl_head = None
        if head_type == 'edl':
            decoder_out_channels = num_classes if decoder_type == 'unet' else num_classes
            # For EDL, decoder outputs feature maps, then EDL head converts to evidence
            self.decoder = UNetDecoder(embed_dim, embed_dim // 2) if decoder_type == 'unet' else \
                DeepLabDecoder(embed_dim, embed_dim // 2)
            self.edl_head = EDLHead(embed_dim // 2, num_classes)

        # Load pretrained weights
        if pretrained:
            self._load_pretrained(pretrained, in_channels)

    def _load_pretrained(self, pretrained, in_channels):
        """Load pretrained weights with input channel adaptation"""
        import os

        if isinstance(pretrained, bool):
            if pretrained:
                pretrained_path = self.get_pretrained_path(self.backbone_name)
                if pretrained_path is None:
                    print(
                        f"[VimSeg] No pretrained weights available for backbone '{self.backbone_name}'")
                    return
            else:
                return
        else:
            pretrained_path = pretrained

        # Check if file exists
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"[VimSeg] Pretrained weights file not found: {pretrained_path}\n"
                f"Please download the pretrained weights or set pretrained=False."
            )

        print(f"[VimSeg] Loading pretrained weights from {pretrained_path}")

        try:
            try:
                checkpoint = torch.load(
                    pretrained_path, map_location='cpu', weights_only=False)
            except TypeError:
                # Older torch versions don't support weights_only
                checkpoint = torch.load(pretrained_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(
                f"[VimSeg] Failed to load pretrained weights from {pretrained_path}: {e}"
            ) from e

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError(
                f"[VimSeg] Invalid checkpoint format. Expected dict, got {type(state_dict)}"
            )

        # Filter out head weights (we have different num_classes)
        encoder_state = self.encoder.state_dict()
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith('head.') and k in encoder_state}

        if len(state_dict) == 0:
            raise ValueError(
                f"[VimSeg] No matching keys found in pretrained weights. "
                f"Expected keys like: {list(encoder_state.keys())[:5]}"
            )

        # Handle input channel adaptation
        if in_channels != 3:
            patch_embed_weight = state_dict.get(
                'patch_embed.proj.weight', None)
            if patch_embed_weight is not None:
                # Average across input channels and replicate
                new_weight = patch_embed_weight.mean(
                    dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                state_dict['patch_embed.proj.weight'] = new_weight
                print(
                    f"[VimSeg] Adapted patch_embed.proj.weight from 3 to {in_channels} channels")

        # Handle position embedding size mismatch
        if 'pos_embed' in state_dict:
            pos_embed = state_dict['pos_embed']
            encoder_pos_embed = self.encoder.pos_embed
            if pos_embed.shape != encoder_pos_embed.shape:
                print(
                    f"[VimSeg] Position embedding size mismatch: pretrained {pos_embed.shape} vs model {encoder_pos_embed.shape}")
                # Keep only matching dimensions or interpolate
                if pos_embed.shape[1] > encoder_pos_embed.shape[1]:
                    state_dict['pos_embed'] = pos_embed[:,
                                                        :encoder_pos_embed.shape[1], :]
                else:
                    # Interpolate
                    N_old = pos_embed.shape[1] - 1  # Exclude cls token
                    N_new = encoder_pos_embed.shape[1] - 1
                    cls_pos = pos_embed[:, :1, :]
                    patch_pos = pos_embed[:, 1:, :]

                    old_size = int(math.sqrt(N_old))
                    new_size = int(math.sqrt(N_new))

                    patch_pos = patch_pos.reshape(
                        1, old_size, old_size, -1).permute(0, 3, 1, 2)
                    patch_pos = F.interpolate(patch_pos, size=(
                        new_size, new_size), mode='bicubic', align_corners=False)
                    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
                        1, -1, pos_embed.shape[-1])

                    state_dict['pos_embed'] = torch.cat(
                        [cls_pos, patch_pos], dim=1)

        # Load state dict
        msg = self.encoder.load_state_dict(state_dict, strict=False)

        # Check for critical missing keys
        critical_missing = [
            k for k in msg.missing_keys if 'mixer' in k or 'layers' in k]
        if critical_missing:
            print(
                f"[VimSeg] Warning: Critical keys missing from pretrained weights: {critical_missing[:5]}...")

        print(f"[VimSeg] Loaded pretrained weights successfully")

    def forward(self, x) -> Dict[str, Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            For standard head: {'out': segmentation logits}
            For EDL head: {'out': logits, 'evidence': evidence, 'alpha': alpha}
        """
        input_size = x.shape[2:]

        # Encoder
        enc_out = self.encoder(x)
        features = enc_out['features']

        # Decoder
        if self.head_type == 'edl':
            feat = self.decoder(features, input_size)
            edl_out = self.edl_head(feat)

            # Compute probabilities from alpha
            alpha = edl_out['alpha']
            S = alpha.sum(dim=1, keepdim=True)
            prob = alpha / S

            return {
                'out': edl_out['logits'],
                'evidence': edl_out['evidence'],
                'alpha': alpha,
                'prob': prob
            }
        else:
            out = self.decoder(features, input_size)
            return {'out': out}


# ============================================================================
# Factory functions
# ============================================================================

def vim_seg_tiny(in_channels=3, num_classes=21, decoder_type='unet',
                 head_type='standard', pretrained=True, **kwargs):
    """VisionMamba Tiny for Segmentation"""
    return VimSeg(in_channels=in_channels, num_classes=num_classes, backbone='tiny',
                  decoder_type=decoder_type, head_type=head_type, pretrained=pretrained, **kwargs)


def vim_seg_small(in_channels=3, num_classes=21, decoder_type='unet',
                  head_type='standard', pretrained=True, **kwargs):
    """VisionMamba Small for Segmentation"""
    return VimSeg(in_channels=in_channels, num_classes=num_classes, backbone='small',
                  decoder_type=decoder_type, head_type=head_type, pretrained=pretrained, **kwargs)


def vim_seg_base(in_channels=3, num_classes=21, decoder_type='unet',
                 head_type='standard', pretrained=True, **kwargs):
    """VisionMamba Base for Segmentation"""
    return VimSeg(in_channels=in_channels, num_classes=num_classes, backbone='base',
                  decoder_type=decoder_type, head_type=head_type, pretrained=pretrained, **kwargs)
