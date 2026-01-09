"""
Cloud Segmentation Loss Functions

클래스 불균형 문제를 해결하기 위한 다양한 손실 함수 구현:
- Weighted CrossEntropyLoss: 클래스 가중치 적용
- Focal Loss: 어려운 샘플에 집중
- Dice Loss: 영역 기반 손실 (클래스 불균형에 강건)
- Combined Loss: CE + Dice 조합
- OHEM (Online Hard Example Mining): 어려운 샘플만 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for dense classification.
    
    논문: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - gamma > 0: 쉬운 샘플의 가중치를 낮춤 (보통 2.0 사용)
    - alpha: 클래스별 가중치 (클래스 불균형 보정)
    
    Args:
        alpha: 클래스별 가중치. None이면 균등 가중치
        gamma: focusing parameter (default: 2.0)
        ignore_index: 무시할 라벨 인덱스
        reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # alpha 처리
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
            elif isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha', alpha.float())
            else:
                self.register_buffer('alpha', torch.tensor([alpha], dtype=torch.float32))
        else:
            self.alpha = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) labels
        """
        B, C, H, W = inputs.shape
        
        # (B, C, H, W) -> (B*H*W, C)
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)
        targets = targets.reshape(-1)
        
        # ignore_index 마스킹
        valid_mask = targets != self.ignore_index
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            if alpha.dim() == 1 and alpha.size(0) == C:
                alpha_t = alpha.gather(0, targets)
            else:
                alpha_t = alpha[0]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice
    
    클래스 불균형에 강건하며, IoU 최적화에 효과적.
    
    Args:
        smooth: 0으로 나누는 것을 방지하기 위한 smoothing factor
        ignore_index: 무시할 라벨 인덱스
        reduction: 'mean', 'sum', 'none'
        class_weights: 클래스별 가중치
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        reduction: str = 'mean',
        class_weights: Optional[Union[List[float], torch.Tensor]] = None
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
            else:
                self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) labels
        """
        num_classes = inputs.size(1)
        
        # Softmax probabilities
        probs = F.softmax(inputs, dim=1)  # (B, C, H, W)
        
        # One-hot encoding for targets
        # ignore_index를 임시로 0으로 변경 (one_hot 인코딩을 위해)
        targets_for_onehot = targets.clone()
        valid_mask = targets != self.ignore_index
        targets_for_onehot[~valid_mask] = 0
        
        # (B, H, W) -> (B, C, H, W)
        targets_onehot = F.one_hot(targets_for_onehot, num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        
        # Apply valid mask (ignore_index 제외)
        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs = probs * valid_mask
        targets_onehot = targets_onehot * valid_mask
        
        # Flatten spatial dimensions
        probs = probs.flatten(2)  # (B, C, H*W)
        targets_onehot = targets_onehot.flatten(2)  # (B, C, H*W)
        
        # Dice per class
        intersection = (probs * targets_onehot).sum(dim=2)  # (B, C)
        union = probs.sum(dim=2) + targets_onehot.sum(dim=2)  # (B, C)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        dice_loss = 1.0 - dice  # (B, C)
        
        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights.to(inputs.device)
            dice_loss = dice_loss * weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss: CrossEntropy + Dice Loss
    
    CE는 픽셀 단위 학습에 효과적이고,
    Dice는 영역 단위 학습에 효과적입니다.
    두 손실을 결합하면 더 좋은 성능을 얻을 수 있습니다.
    
    Args:
        ce_weight: CrossEntropy 손실 가중치
        dice_weight: Dice 손실 가중치
        class_weights: 클래스별 가중치 (CE와 Dice 모두에 적용)
        focal_gamma: > 0이면 FocalLoss 사용, 0이면 CE 사용
        ignore_index: 무시할 라벨 인덱스
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None,
        focal_gamma: float = 0.0,
        ignore_index: int = 255
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        
        # CE or Focal
        if focal_gamma > 0:
            self.ce_loss = FocalLoss(
                alpha=class_weights,
                gamma=focal_gamma,
                ignore_index=ignore_index
            )
        else:
            if class_weights is not None:
                if isinstance(class_weights, (list, tuple)):
                    weight = torch.tensor(class_weights, dtype=torch.float32)
                else:
                    weight = class_weights.float()
            else:
                weight = None
            self.ce_loss = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_index
            )
        
        # Dice
        self.dice_loss = DiceLoss(
            ignore_index=ignore_index,
            class_weights=class_weights
        )
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining Loss
    
    가장 어려운 k%의 픽셀만 선택하여 학습합니다.
    쉬운 샘플(대부분 배경)을 무시하고 어려운 샘플에 집중합니다.
    
    Args:
        thresh: hard negative threshold (default: 0.7)
        min_kept: 최소 유지할 픽셀 수 또는 비율
        ignore_index: 무시할 라벨 인덱스
        class_weights: 클래스별 가중치
    """
    
    def __init__(
        self,
        thresh: float = 0.7,
        min_kept: Union[int, float] = 100000,
        ignore_index: int = 255,
        class_weights: Optional[Union[List[float], torch.Tensor]] = None
    ):
        super().__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
            else:
                self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inputs.shape
        
        # Pixel-wise CE loss
        if self.class_weights is not None:
            weight = self.class_weights.to(inputs.device)
        else:
            weight = None
        
        pixel_losses = F.cross_entropy(
            inputs, targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction='none'
        )  # (B, H, W)
        
        # Flatten
        pixel_losses = pixel_losses.view(-1)
        
        # Valid pixels
        valid_mask = targets.view(-1) != self.ignore_index
        pixel_losses = pixel_losses[valid_mask]
        
        if pixel_losses.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Get probabilities for threshold
        probs = F.softmax(inputs, dim=1)
        probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
        probs = probs[valid_mask]
        targets_flat = targets.view(-1)[valid_mask]
        
        # Get probability of correct class
        p_correct = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        
        # Hard examples: low confidence on correct class
        hard_mask = p_correct < self.thresh
        
        # Minimum kept
        if isinstance(self.min_kept, float):
            min_kept = int(self.min_kept * pixel_losses.numel())
        else:
            min_kept = self.min_kept
        
        min_kept = min(min_kept, pixel_losses.numel())
        
        if hard_mask.sum() < min_kept:
            # Sort by loss (descending) and take top-k
            _, topk_idx = pixel_losses.topk(min_kept)
            hard_mask = torch.zeros_like(hard_mask)
            hard_mask[topk_idx] = True
        
        hard_losses = pixel_losses[hard_mask]
        
        if hard_losses.numel() == 0:
            return pixel_losses.mean()
        
        return hard_losses.mean()


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss
    
    논문: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)
    
    각 클래스의 effective number of samples를 기반으로 가중치를 계산합니다.
    
    E_n = (1 - beta^n) / (1 - beta)
    weight = 1 / E_n
    
    Args:
        samples_per_class: 각 클래스의 샘플 수 리스트
        beta: hyperparameter (default: 0.9999)
        loss_type: 'focal', 'ce', 'dice', 'combined'
        focal_gamma: focal loss gamma
        ignore_index: 무시할 라벨 인덱스
    """
    
    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        loss_type: str = 'focal',
        focal_gamma: float = 2.0,
        ignore_index: int = 255
    ):
        super().__init__()
        
        # Effective number of samples
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(samples_per_class, dtype=torch.float32))
        effective_num = effective_num / (1.0 - beta)
        
        # Weights inversely proportional to effective number
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(samples_per_class)  # Normalize
        
        self.register_buffer('weights', weights)
        self.loss_type = loss_type
        self.ignore_index = ignore_index
        
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(
                alpha=weights,
                gamma=focal_gamma,
                ignore_index=ignore_index
            )
        elif loss_type == 'ce':
            self.loss_fn = nn.CrossEntropyLoss(
                weight=weights,
                ignore_index=ignore_index
            )
        elif loss_type == 'dice':
            self.loss_fn = DiceLoss(
                ignore_index=ignore_index,
                class_weights=weights
            )
        elif loss_type == 'combined':
            self.loss_fn = CombinedLoss(
                class_weights=weights,
                focal_gamma=focal_gamma,
                ignore_index=ignore_index
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(inputs, targets)


def get_class_weights(
    dataset_name: str,
    num_classes: int,
    method: str = 'inverse_freq',
    samples_per_class: Optional[List[int]] = None
) -> torch.Tensor:
    """
    데이터셋별 클래스 가중치 계산
    
    Args:
        dataset_name: 데이터셋 이름
        num_classes: 클래스 수
        method: 'inverse_freq', 'sqrt_inverse_freq', 'effective_num', 'manual'
        samples_per_class: 각 클래스의 샘플 수 (method='effective_num'일 때 필요)
    
    Returns:
        클래스 가중치 텐서
    """
    # 데이터셋별 수동 가중치 (분석 기반 추정치)
    # 실제 데이터 분포에 따라 조정 필요
    MANUAL_WEIGHTS = {
        'l8biome': {
            # clear, thin_cloud, cloud, cloud_shadow
            # thin_cloud와 cloud_shadow가 가장 적음
            'weights': [0.5, 3.0, 1.0, 2.0],
        },
        'cloudsen12_l1c': {
            # clear, thick_cloud, thin_cloud, cloud_shadow
            'weights': [0.5, 1.0, 2.5, 2.5],
        },
        'cloudsen12_l2a': {
            # clear, thick_cloud, thin_cloud, cloud_shadow
            'weights': [0.5, 1.0, 2.5, 2.5],
        },
        'cloud38': {
            # clear, cloud
            'weights': [0.5, 1.5],
        },
        'cloud95': {
            # clear, cloud
            'weights': [0.5, 1.5],
        },
    }
    
    if method == 'manual':
        if dataset_name in MANUAL_WEIGHTS:
            weights = torch.tensor(MANUAL_WEIGHTS[dataset_name]['weights'], dtype=torch.float32)
        else:
            weights = torch.ones(num_classes, dtype=torch.float32)
    elif method == 'effective_num':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for 'effective_num' method")
        beta = 0.9999
        effective_num = 1.0 - torch.pow(
            torch.tensor(beta),
            torch.tensor(samples_per_class, dtype=torch.float32)
        )
        effective_num = effective_num / (1.0 - beta)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * num_classes
    elif method == 'inverse_freq':
        if samples_per_class is None:
            # 수동 가중치 사용
            if dataset_name in MANUAL_WEIGHTS:
                weights = torch.tensor(MANUAL_WEIGHTS[dataset_name]['weights'], dtype=torch.float32)
            else:
                weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            total = sum(samples_per_class)
            weights = torch.tensor([total / (num_classes * s) for s in samples_per_class], dtype=torch.float32)
    elif method == 'sqrt_inverse_freq':
        if samples_per_class is None:
            if dataset_name in MANUAL_WEIGHTS:
                weights = torch.tensor(MANUAL_WEIGHTS[dataset_name]['weights'], dtype=torch.float32)
                weights = torch.sqrt(weights)
            else:
                weights = torch.ones(num_classes, dtype=torch.float32)
        else:
            total = sum(samples_per_class)
            weights = torch.tensor([total / (num_classes * s) for s in samples_per_class], dtype=torch.float32)
            weights = torch.sqrt(weights)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weights


def get_loss_function(
    loss_type: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = 255,
    **kwargs
) -> nn.Module:
    """
    손실 함수 팩토리
    
    Args:
        loss_type: 손실 함수 종류
            - 'ce': CrossEntropyLoss
            - 'weighted_ce': Weighted CrossEntropyLoss
            - 'focal': Focal Loss
            - 'dice': Dice Loss
            - 'ce_dice': CE + Dice Combined
            - 'focal_dice': Focal + Dice Combined
            - 'ohem': Online Hard Example Mining
        num_classes: 클래스 수
        class_weights: 클래스 가중치
        ignore_index: 무시할 인덱스
        **kwargs: 추가 인자 (gamma, ce_weight, dice_weight 등)
    
    Returns:
        손실 함수 모듈
    """
    gamma = kwargs.get('gamma', 2.0)
    ce_weight = kwargs.get('ce_weight', 1.0)
    dice_weight = kwargs.get('dice_weight', 1.0)
    ohem_thresh = kwargs.get('ohem_thresh', 0.7)
    ohem_min_kept = kwargs.get('ohem_min_kept', 100000)
    
    if loss_type == 'ce':
        if ignore_index is not None:
            return nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            return nn.CrossEntropyLoss()
    
    elif loss_type == 'weighted_ce':
        if class_weights is not None:
            if ignore_index is not None:
                return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
            else:
                return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index is not None else -100)
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            ignore_index=ignore_index if ignore_index is not None else 255
        )
    
    elif loss_type == 'dice':
        return DiceLoss(
            ignore_index=ignore_index if ignore_index is not None else 255,
            class_weights=class_weights
        )
    
    elif loss_type == 'ce_dice':
        return CombinedLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            class_weights=class_weights,
            focal_gamma=0.0,
            ignore_index=ignore_index if ignore_index is not None else 255
        )
    
    elif loss_type == 'focal_dice':
        return CombinedLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            class_weights=class_weights,
            focal_gamma=gamma,
            ignore_index=ignore_index if ignore_index is not None else 255
        )
    
    elif loss_type == 'ohem':
        return OHEMLoss(
            thresh=ohem_thresh,
            min_kept=ohem_min_kept,
            ignore_index=ignore_index if ignore_index is not None else 255,
            class_weights=class_weights
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
