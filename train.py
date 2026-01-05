"""
Cloud Segmentation Training Module

다양한 데이터셋과 모델을 지원하는 학습 모듈
사용법:
    python train.py --dataset l8biome --model unet --epochs 100
    python train.py --dataset cloudsen12 --model deeplabv3plus --bands 4 3 2
    python train.py --dataset cloud95 --model cdnetv2 --batch_size 16
"""

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Models
from models.modeling import get_model, list_models

# Datasets
from l8biome_dataset import L8BiomeDataset
from cloudsen12_dataset import CloudSEN12Dataset
from cloud38_95_dataset import Cloud38Dataset, Cloud95Dataset


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_CONFIG = {
    'l8biome': {
        'class': L8BiomeDataset,
        'num_classes': 4,
        'ignore_index': 255,
        'class_names': ['clear', 'thin_cloud', 'cloud', 'cloud_shadow'],
        'default_bands': [4, 3, 2],  # RGB
        'all_bands': list(range(1, 12)),  # 1-11
        'data_dir': '/home/telepix_nas/junghwan/cloud_seg/l8biome_extracted/l8biome',
    },
    'cloudsen12_l1c': {
        'class': CloudSEN12Dataset,
        'num_classes': 4,
        'ignore_index': None,
        'class_names': ['clear', 'thick_cloud', 'thin_cloud', 'cloud_shadow'],
        'default_bands': [4, 3, 2],  # RGB
        'all_bands': list(range(1, 14)),  # 1-13
        'data_dir': '/home/telepix_nas/junghwan/cloud_seg/cloudsen12-l1c',
        'level': 'l1c',
    },
    'cloudsen12_l2a': {
        'class': CloudSEN12Dataset,
        'num_classes': 4,
        'ignore_index': None,
        'class_names': ['clear', 'thick_cloud', 'thin_cloud', 'cloud_shadow'],
        'default_bands': [4, 3, 2],  # RGB
        'all_bands': list(range(1, 15)),  # 1-14
        'data_dir': '/home/telepix_nas/junghwan/cloud_seg/cloudsen12-l2a',
        'level': 'l2a',
    },
    'cloud38': {
        'class': Cloud38Dataset,
        'num_classes': 2,
        'ignore_index': None,
        'class_names': ['clear', 'cloud'],
        'default_bands': ['red', 'green', 'blue', 'nir'],
        'all_bands': ['red', 'green', 'blue', 'nir'],
        'data_dir': '/home/telepix_nas/junghwan/cloud_seg/38-cloud',
    },
    'cloud95': {
        'class': Cloud95Dataset,
        'num_classes': 2,
        'ignore_index': None,
        'class_names': ['clear', 'cloud'],
        'default_bands': ['red', 'green', 'blue', 'nir'],
        'all_bands': ['red', 'green', 'blue', 'nir'],
        'data_dir_38': '/home/telepix_nas/junghwan/cloud_seg/38-cloud',
        'data_dir_95': '/home/telepix_nas/junghwan/cloud_seg/95-cloud',
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(dataset_name, split, bands=None, patch_size=512, **kwargs):
    """
    데이터셋 생성

    Args:
        dataset_name: 데이터셋 이름
        split: 'train', 'val', 'test'
        bands: 사용할 밴드 리스트
        patch_size: 패치 크기 (l8biome에서 사용)
    """
    config = DATASET_CONFIG[dataset_name]

    if bands is None:
        bands = config['default_bands']

    if dataset_name == 'l8biome':
        dataset = L8BiomeDataset(
            data_dir=config['data_dir'],
            split=split,
            patch_size=patch_size,
            bands=bands,
            normalize=True,
            **kwargs
        )
    elif dataset_name.startswith('cloudsen12'):
        # CloudSEN12는 split 이름이 다름 (val -> validation)
        cs_split = 'validation' if split == 'val' else split
        dataset = CloudSEN12Dataset(
            taco_dir=config['data_dir'],
            split=cs_split,
            bands=bands,
            level=config['level'],
            normalize=True,
            **kwargs
        )
    elif dataset_name == 'cloud38':
        dataset = Cloud38Dataset(
            root=config['data_dir'],
            split=split,
            bands=bands,
            normalize=True,
            **kwargs
        )
    elif dataset_name == 'cloud95':
        dataset = Cloud95Dataset(
            root_38cloud=config['data_dir_38'],
            root_95cloud=config['data_dir_95'],
            split=split,
            bands=bands,
            normalize=True,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def create_dataloaders(dataset_name, batch_size, bands=None, patch_size=512,
                       num_workers=4, pin_memory=True):
    """학습/검증/테스트 데이터로더 생성"""

    train_dataset = get_dataset(dataset_name, 'train', bands, patch_size)
    val_dataset = get_dataset(dataset_name, 'val', bands, patch_size)
    test_dataset = get_dataset(dataset_name, 'test', bands, patch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


class AverageMeter:
    """평균 및 현재 값 추적"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(pred, target, num_classes, ignore_index=None):
    """
    분할 메트릭 계산 (IoU, Accuracy)

    Args:
        pred: 예측값 (B, H, W)
        target: 정답 (B, H, W)
        num_classes: 클래스 수
        ignore_index: 무시할 인덱스

    Returns:
        dict: iou, mean_iou, accuracy
    """
    pred = pred.flatten()
    target = target.flatten()

    # Ignore index 마스킹
    if ignore_index is not None:
        valid_mask = target != ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]

    # Confusion matrix
    confusion_matrix = torch.zeros(
        num_classes, num_classes, device=pred.device)
    for t, p in zip(target, pred):
        confusion_matrix[t.long(), p.long()] += 1

    # IoU per class
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
    iou = intersection / (union + 1e-10)

    # Mean IoU
    valid_classes = union > 0
    mean_iou = iou[valid_classes].mean()

    # Accuracy
    accuracy = intersection.sum() / (confusion_matrix.sum() + 1e-10)

    return {
        'iou': iou.cpu().numpy(),
        'mean_iou': mean_iou.item(),
        'accuracy': accuracy.item(),
    }


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    scaler=None, num_classes=4, ignore_index=None, aux_weight=0.4):
    """한 에폭 학습"""
    model.train()

    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(train_loader, desc='Training')

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)

                # CDNetV2는 auxiliary output 반환
                if isinstance(outputs, tuple):
                    main_out, aux_out = outputs
                    loss = criterion(main_out, targets)
                    if aux_out is not None:
                        loss += aux_weight * criterion(aux_out, targets)
                    pred = main_out.argmax(1)
                else:
                    loss = criterion(outputs, targets)
                    pred = outputs.argmax(1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)

            if isinstance(outputs, tuple):
                main_out, aux_out = outputs
                loss = criterion(main_out, targets)
                if aux_out is not None:
                    loss += aux_weight * criterion(aux_out, targets)
                pred = main_out.argmax(1)
            else:
                loss = criterion(outputs, targets)
                pred = outputs.argmax(1)

            loss.backward()
            optimizer.step()

        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(pred, targets, num_classes, ignore_index)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        iou_meter.update(metrics['mean_iou'], batch_size)
        acc_meter.update(metrics['accuracy'], batch_size)

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'mIoU': f'{iou_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })

    return {
        'loss': loss_meter.avg,
        'mean_iou': iou_meter.avg,
        'accuracy': acc_meter.avg,
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, num_classes=4, ignore_index=None):
    """검증"""
    model.eval()

    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    acc_meter = AverageMeter()
    class_iou_sum = np.zeros(num_classes)
    class_count = 0

    pbar = tqdm(val_loader, desc='Validation')

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, targets)
        pred = outputs.argmax(1)

        metrics = compute_metrics(pred, targets, num_classes, ignore_index)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        iou_meter.update(metrics['mean_iou'], batch_size)
        acc_meter.update(metrics['accuracy'], batch_size)
        class_iou_sum += metrics['iou']
        class_count += 1

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'mIoU': f'{iou_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })

    class_iou = class_iou_sum / class_count

    return {
        'loss': loss_meter.avg,
        'mean_iou': iou_meter.avg,
        'accuracy': acc_meter.avg,
        'class_iou': class_iou,
    }


def save_checkpoint(state, filename):
    """체크포인트 저장"""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


# =============================================================================
# Main Training Function
# =============================================================================

def main(args):
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset configuration
    dataset_config = DATASET_CONFIG[args.dataset]
    num_classes = dataset_config['num_classes']
    ignore_index = dataset_config['ignore_index']
    class_names = dataset_config['class_names']

    # Parse bands
    if args.bands is not None:
        if args.dataset in ['cloud38', 'cloud95']:
            bands = args.bands  # String bands for cloud38/95
        else:
            bands = [int(b) for b in args.bands]  # Integer bands for others
    else:
        bands = dataset_config['default_bands']

    in_channels = len(bands)

    print(f"\n{'='*60}")
    print(f"Cloud Segmentation Training")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Input channels: {in_channels} (bands: {bands})")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"{'='*60}\n")

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        bands=bands,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(
        args.model,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained_backbone=args.pretrained,
        aux=args.aux,
        output_stride=args.output_stride,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    if ignore_index is not None:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'poly':
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=args.epochs, power=0.9)
    else:
        scheduler = None

    # Mixed precision
    scaler = GradScaler() if args.amp else None

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / \
        f"{args.dataset}_{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_miou = 0.0
    history = {'train': [], 'val': []}

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Output directory: {output_dir}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, num_classes=num_classes, ignore_index=ignore_index,
            aux_weight=args.aux_weight,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device,
            num_classes=num_classes, ignore_index=ignore_index,
        )

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Log
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"mIoU: {train_metrics['mean_iou']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"mIoU: {val_metrics['mean_iou']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}")

        # Class-wise IoU
        print("Class-wise IoU:")
        for i, (name, iou) in enumerate(zip(class_names, val_metrics['class_iou'])):
            print(f"  {name}: {iou:.4f}")

        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Save best model
        if val_metrics['mean_iou'] > best_miou:
            best_miou = val_metrics['mean_iou']
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'config': {
                    'model': args.model,
                    'dataset': args.dataset,
                    'in_channels': in_channels,
                    'num_classes': num_classes,
                    'bands': bands,
                }
            }, output_dir / 'best_model.pth')

        # Save latest model
        if epoch % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(
        model, test_loader, criterion, device,
        num_classes=num_classes, ignore_index=ignore_index,
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  mIoU: {test_metrics['mean_iou']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"\nClass-wise IoU:")
    for name, iou in zip(class_names, test_metrics['class_iou']):
        print(f"  {name}: {iou:.4f}")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Test mIoU: {test_metrics['mean_iou']:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Cloud Segmentation Training')

    # Dataset
    parser.add_argument('--dataset', type=str, default='l8biome',
                        choices=['l8biome', 'cloudsen12_l1c', 'cloudsen12_l2a',
                                 'cloud38', 'cloud95'],
                        help='Dataset to use')
    parser.add_argument('--bands', nargs='+', default=None,
                        help='Bands to use (e.g., 4 3 2 for RGB)')
    parser.add_argument('--patch_size', type=int, default=512,
                        help='Patch size for L8Biome dataset')

    # Model
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'deeplabv3plus',
                                 'cdnetv1', 'cdnetv2'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Do not use pretrained backbone')
    parser.add_argument('--output_stride', type=int, default=16,
                        help='Output stride for DeepLabV3+')
    parser.add_argument('--aux', action='store_true', default=True,
                        help='Use auxiliary loss for CDNetV2')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                        help='Auxiliary loss weight')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine', 'poly'],
                        help='Learning rate scheduler')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')

    # Misc
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint frequency')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
