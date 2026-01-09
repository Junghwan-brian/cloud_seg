"""
Cloud Segmentation Training Module

다양한 데이터셋과 모델을 지원하는 학습 모듈
사용법:
    python train.py --dataset l8biome --model unet --epochs 100
    python train.py --dataset cloudsen12 --model deeplabv3plus --bands 4 3 2
    python train.py --dataset cloud95 --model cdnetv2 --batch_size 16
"""

import argparse
import json
import logging
import os
import random
import sys
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
from models.vim_seg import EDLLoss

# Datasets
from l8biome_dataset import L8BiomeDataset
from cloudsen12_dataset import CloudSEN12Dataset
from cloud38_95_dataset import Cloud38Dataset, Cloud95Dataset

# Path utilities
from utils.paths import get_nas_path, detect_nas_base

# Loss functions
from utils.losses import get_loss_function, get_class_weights


# =============================================================================
# Dataset Configuration
# =============================================================================

def _get_dataset_config():
    """
    Get dataset configuration with auto-detected NAS paths.
    """
    nas_base = detect_nas_base()
    if nas_base is None:
        raise RuntimeError(
            "[train] NAS path not found. Please ensure one of these paths is available:\n"
            "  - /home/telepix_nas/junghwan/cloud_seg\n"
            "  - /nas/junghwan/cloud_seg"
        )

    return {
        'l8biome': {
            'class': L8BiomeDataset,
            'num_classes': 4,
            'ignore_index': 255,
            'class_names': ['clear', 'thin_cloud', 'cloud', 'cloud_shadow'],
            'default_bands': list(range(1, 12)),  # 1-11 (전체 11채널)
            'data_dir': get_nas_path('l8biome_extracted/l8biome'),
        },
        'cloudsen12_l1c': {
            'class': CloudSEN12Dataset,
            'num_classes': 4,
            'ignore_index': 255,  # Invalid labels (4, 5, 6, 99 등)를 255로 매핑
            'class_names': ['clear', 'thick_cloud', 'thin_cloud', 'cloud_shadow'],
            'default_bands': list(range(1, 14)),  # 1-13 (전체 13채널)
            'data_dir': get_nas_path('cloudsen12-l1c'),
            'level': 'l1c',
        },
        'cloudsen12_l2a': {
            'class': CloudSEN12Dataset,
            'num_classes': 4,
            'ignore_index': 255,  # Invalid labels (4, 5, 6, 99 등)를 255로 매핑
            'class_names': ['clear', 'thick_cloud', 'thin_cloud', 'cloud_shadow'],
            'default_bands': list(range(1, 15)),  # 1-14 (전체 14채널)
            'data_dir': get_nas_path('cloudsen12-l2a'),
            'level': 'l2a',
        },
        'cloud38': {
            'class': Cloud38Dataset,
            'num_classes': 2,
            'ignore_index': None,
            'class_names': ['clear', 'cloud'],
            'default_bands': ['red', 'green', 'blue', 'nir'],  # 전체 4채널
            'data_dir': get_nas_path('38-cloud'),
        },
        'cloud95': {
            'class': Cloud95Dataset,
            'num_classes': 2,
            'ignore_index': None,
            'class_names': ['clear', 'cloud'],
            'default_bands': ['red', 'green', 'blue', 'nir'],  # 전체 4채널
            'data_dir_38': get_nas_path('38-cloud'),
            'data_dir_95': get_nas_path('95-cloud'),
        },
    }


# Lazy initialization of DATASET_CONFIG
DATASET_CONFIG = None


def get_dataset_config():
    """Get dataset configuration (lazy initialization)."""
    global DATASET_CONFIG
    if DATASET_CONFIG is None:
        DATASET_CONFIG = _get_dataset_config()
    return DATASET_CONFIG


# =============================================================================
# Helper Functions
# =============================================================================

def set_seed(seed, deterministic=False):
    """
    재현성을 위한 시드 설정

    Args:
        seed: 랜덤 시드
        deterministic: True면 완전한 재현성 보장 (느림), 
                       False면 성능 우선 (기본값)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 완전한 재현성 모드 (느림)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 성능 우선 모드 (빠름) - 입력 크기가 고정된 경우 최적
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_dataset(dataset_name, split, bands=None, patch_size=512, preload=False, **kwargs):
    """
    데이터셋 생성

    Args:
        dataset_name: 데이터셋 이름
        split: 'train', 'val', 'test'
        bands: 사용할 밴드 리스트
        patch_size: 패치 크기 (l8biome에서 사용)
        preload: 데이터를 메모리에 미리 로드할지 여부 (빠른 학습)
    """
    config = get_dataset_config()[dataset_name]

    if bands is None:
        bands = config['default_bands']

    if dataset_name == 'l8biome':
        dataset = L8BiomeDataset(
            data_dir=config['data_dir'],
            split=split,
            patch_size=patch_size,
            bands=bands,
            normalize=True,
            use_cache=True,  # 파일 핸들 캐싱 활성화
            preload=preload,  # 메모리 프리로드 옵션
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
            patch_size=patch_size,  # 패치 크기 추가
            use_cache=True,  # 파일 핸들 캐싱 활성화
            preload=preload,  # 메모리 프리로드 옵션
            **kwargs
        )
    elif dataset_name == 'cloud38':
        dataset = Cloud38Dataset(
            root=config['data_dir'],
            split=split,
            bands=bands,
            normalize=True,
            preload=preload,  # 메모리 프리로드 옵션
            **kwargs
        )
    elif dataset_name == 'cloud95':
        dataset = Cloud95Dataset(
            root_38cloud=config['data_dir_38'],
            root_95cloud=config['data_dir_95'],
            split=split,
            bands=bands,
            normalize=True,
            preload=preload,  # 메모리 프리로드 옵션
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def create_dataloaders(dataset_name, batch_size, bands=None, patch_size=512,
                       num_workers=4, pin_memory=True, preload=False,
                       prefetch_factor=4, persistent_workers=True):
    """
    학습/검증/테스트 데이터로더 생성

    Args:
        dataset_name: 데이터셋 이름
        batch_size: 배치 크기
        bands: 사용할 밴드 리스트
        patch_size: 패치 크기
        num_workers: 데이터 로딩 워커 수
        pin_memory: GPU 메모리 핀닝
        preload: 데이터를 메모리에 미리 로드 (빠른 학습)
        prefetch_factor: 각 워커당 미리 로드할 배치 수 (기본값 4)
        persistent_workers: 워커를 에폭 간 유지 (초기화 오버헤드 감소)
    """

    train_dataset = get_dataset(
        dataset_name, 'train', bands, patch_size, preload=preload)
    val_dataset = get_dataset(
        dataset_name, 'val', bands, patch_size, preload=preload)
    test_dataset = get_dataset(
        dataset_name, 'test', bands, patch_size, preload=False)  # 테스트는 프리로드 안함

    # 워커가 있을 때만 prefetch_factor와 persistent_workers 사용
    loader_kwargs = {
        'pin_memory': pin_memory,
        'num_workers': num_workers,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
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
    분할 메트릭 계산 (IoU, Accuracy) - GPU 최적화 버전

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

    # 빈 텐서 처리
    if pred.numel() == 0:
        return {
            'iou': np.zeros(num_classes),
            'mean_iou': 0.0,
            'accuracy': 0.0,
        }

    # GPU에서 직접 Confusion Matrix 계산 (scatter_add 사용)
    # 이 방식은 bincount보다 GPU에서 더 효율적
    confusion_matrix = torch.zeros(
        num_classes, num_classes,
        dtype=torch.float32,
        device=pred.device
    )

    # one-hot 인코딩 + 행렬 누적
    indices = target.long() * num_classes + pred.long()
    ones = torch.ones_like(indices, dtype=torch.float32)
    confusion_matrix.view(-1).scatter_add_(0, indices, ones)

    # IoU per class
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
    iou = intersection / (union + 1e-10)

    # Mean IoU (유효한 클래스만)
    valid_classes = union > 0
    if valid_classes.sum() > 0:
        mean_iou = iou[valid_classes].mean()
    else:
        mean_iou = torch.tensor(0.0, device=pred.device)

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
                    scaler=None, num_classes=4, ignore_index=None, aux_weight=0.4,
                    epoch=0, total_epochs=100, is_edl=False):
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
                elif isinstance(outputs, dict):
                    # EDL 모델 처리 (alpha 기반 loss)
                    if is_edl and 'alpha' in outputs:
                        loss_dict = criterion(
                            outputs, targets,
                            epoch=epoch, total_epochs=total_epochs,
                            ignore_index=ignore_index if ignore_index is not None else 255
                        )
                        loss = loss_dict['loss']
                        # EDL에서는 prob 또는 alpha를 사용하여 예측
                        if 'prob' in outputs:
                            pred = outputs['prob'].argmax(1)
                        else:
                            alpha = outputs['alpha']
                            pred = alpha.argmax(1)
                    else:
                        # VimSeg 등 일반 dict 반환 모델 처리
                        main_out = outputs['out']
                        loss = criterion(main_out, targets)
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
            elif isinstance(outputs, dict):
                # EDL 모델 처리 (alpha 기반 loss)
                if is_edl and 'alpha' in outputs:
                    loss_dict = criterion(
                        outputs, targets,
                        epoch=epoch, total_epochs=total_epochs,
                        ignore_index=ignore_index if ignore_index is not None else 255
                    )
                    loss = loss_dict['loss']
                    # EDL에서는 prob 또는 alpha를 사용하여 예측
                    if 'prob' in outputs:
                        pred = outputs['prob'].argmax(1)
                    else:
                        alpha = outputs['alpha']
                        pred = alpha.argmax(1)
                else:
                    # VimSeg 등 일반 dict 반환 모델 처리
                    main_out = outputs['out']
                    loss = criterion(main_out, targets)
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
def validate(model, val_loader, criterion, device, num_classes=4, ignore_index=None,
             epoch=0, total_epochs=100, is_edl=False):
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
        elif isinstance(outputs, dict):
            # EDL 모델 처리
            if is_edl and 'alpha' in outputs:
                loss_dict = criterion(
                    outputs, targets,
                    epoch=epoch, total_epochs=total_epochs,
                    ignore_index=ignore_index if ignore_index is not None else 255
                )
                loss = loss_dict['loss']
                # EDL에서는 prob 또는 alpha를 사용하여 예측
                if 'prob' in outputs:
                    pred = outputs['prob'].argmax(1)
                else:
                    alpha = outputs['alpha']
                    pred = alpha.argmax(1)
            else:
                outputs = outputs['out']
                loss = criterion(outputs, targets)
                pred = outputs.argmax(1)
        else:
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
    logging.info(f"Checkpoint saved: {filename}")


def setup_logging(output_dir: Path, log_name: str = 'training.log'):
    """
    로깅 설정 - 콘솔과 파일 모두에 로그 출력

    Args:
        output_dir: 로그 파일을 저장할 디렉토리
        log_name: 로그 파일 이름
    """
    log_file = output_dir / log_name

    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def generate_experiment_name(args):
    """
    Arguments에 기반한 실험 이름 생성

    Args:
        args: Parsed arguments

    Returns:
        실험 이름 문자열
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 기본 이름: dataset_model
    name_parts = [args.dataset, args.model]

    # 주요 하이퍼파라미터 추가
    name_parts.append(f"lr{args.lr}")
    name_parts.append(f"bs{args.batch_size}")
    name_parts.append(f"{args.optimizer}")

    # 모델별 특수 파라미터
    if args.model == 'deeplabv3plus':
        name_parts.append(f"os{args.output_stride}")
    elif args.model == 'cdnetv2':
        name_parts.append(f"aux{args.aux_weight}")
    elif args.model.startswith('vim_'):
        name_parts.append(f"{args.decoder_type}")
        name_parts.append(f"{args.head_type}")

    # Loss 정보 추가
    name_parts.append(args.loss_type)
    if args.use_class_weights:
        name_parts.append("weighted")

    # 타임스탬프 추가
    name_parts.append(timestamp)

    return "_".join(str(p) for p in name_parts)


def save_config(args, output_dir: Path, additional_info: dict = None):
    """
    학습 설정을 JSON 파일로 저장

    Args:
        args: Parsed arguments
        output_dir: 저장할 디렉토리
        additional_info: 추가 정보 (예: in_channels, num_classes)
    """
    config = vars(args).copy()

    # 추가 정보 병합
    if additional_info:
        config.update(additional_info)

    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    logging.info(f"Config saved: {config_file}")


def save_history(history: dict, output_dir: Path):
    """
    학습 히스토리를 JSON 파일로 저장

    Args:
        history: 학습/검증 메트릭 히스토리
        output_dir: 저장할 디렉토리
    """
    # numpy array를 list로 변환
    serializable_history = {}
    for split, metrics_list in history.items():
        serializable_history[split] = []
        for metrics in metrics_list:
            serializable_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, np.ndarray):
                    serializable_metrics[k] = v.tolist()
                else:
                    serializable_metrics[k] = v
            serializable_history[split].append(serializable_metrics)

    history_file = output_dir / 'history.json'
    with open(history_file, 'w') as f:
        json.dump(serializable_history, f, indent=2)

    logging.info(f"History saved: {history_file}")


# =============================================================================
# Main Training Function
# =============================================================================

def main(args):
    # Setup - 기본은 성능 우선 모드, --deterministic 옵션으로 재현성 모드 선택 가능
    set_seed(args.seed, deterministic=args.deterministic)

    # Device setup
    if args.gpu is not None:
        if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(args.gpu)
        else:
            raise ValueError(
                f"GPU {args.gpu} is not available. "
                f"Available GPUs: {torch.cuda.device_count()}"
            )
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset configuration
    dataset_config = get_dataset_config()[args.dataset]
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

    # Output directory 설정 (argument 기반 이름 생성)
    exp_name = generate_experiment_name(args)
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory 설정 (NAS 경로 또는 output_dir 사용)
    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir) / exp_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_dir = output_dir

    # 로깅 설정
    setup_logging(output_dir)

    logging.info(f"{'='*60}")
    logging.info(f"Cloud Segmentation Training")
    logging.info(f"{'='*60}")
    logging.info(f"Experiment: {exp_name}")
    logging.info(f"Log directory: {output_dir}")
    logging.info(f"Checkpoint directory: {checkpoint_dir}")
    logging.info(f"Device: {device}")
    logging.info(f"{'='*60}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Input channels: {in_channels} (bands: {bands})")
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Class names: {class_names}")
    logging.info(f"{'='*60}")
    logging.info(f"Hyperparameters:")
    logging.info(f"  - Learning rate: {args.lr}")
    logging.info(f"  - Batch size: {args.batch_size}")
    logging.info(f"  - Optimizer: {args.optimizer}")
    logging.info(f"  - Scheduler: {args.scheduler}")
    logging.info(f"  - Weight decay: {args.weight_decay}")
    logging.info(f"  - Epochs: {args.epochs}")
    logging.info(f"  - Patch size: {args.patch_size}")
    if args.model.startswith('vim_'):
        logging.info(f"  - Decoder type: {args.decoder_type}")
        logging.info(f"  - Head type: {args.head_type}")
    logging.info(f"{'='*60}")

    # Config 저장
    save_config(args, output_dir, {
        'in_channels': in_channels,
        'num_classes': num_classes,
        'class_names': class_names,
        'bands': bands,
        'device': str(device),
        'experiment_name': exp_name,
        'log_dir': str(output_dir),
        'checkpoint_dir': str(checkpoint_dir),
    })

    # Create dataloaders
    logging.info("Loading datasets...")
    if args.preload:
        logging.info("Preloading data to memory for faster training...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        bands=bands,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        preload=args.preload,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.num_workers > 0,
    )
    logging.info(f"Train batches: {len(train_loader)}")
    logging.info(f"Val batches: {len(val_loader)}")
    logging.info(f"Test batches: {len(test_loader)}")

    # Create model
    logging.info(f"Creating model: {args.model}")
    model = get_model(
        args.model,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained_backbone=args.pretrained,
        aux=args.aux,
        output_stride=args.output_stride,
        decoder_type=args.decoder_type,
        head_type=args.head_type,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    is_edl = args.model.startswith('vim_') and args.head_type == 'edl'
    if is_edl:
        # EDL Loss for uncertainty estimation
        criterion = EDLLoss(
            num_classes=num_classes,
            annealing_epochs=args.edl_annealing_epochs,
            lambda_kl=args.edl_lambda_kl,
        )
        logging.info(
            f"Using EDL Loss (annealing_epochs={args.edl_annealing_epochs}, lambda_kl={args.edl_lambda_kl})")
    else:
        # 클래스 가중치 계산
        class_weights = None
        if args.use_class_weights:
            class_weights = get_class_weights(
                dataset_name=args.dataset,
                num_classes=num_classes,
                method=args.class_weight_method,
            )
            logging.info(
                f"Class weights ({args.class_weight_method}): {class_weights.tolist()}")

        # 손실 함수 생성
        criterion = get_loss_function(
            loss_type=args.loss_type,
            num_classes=num_classes,
            class_weights=class_weights,
            ignore_index=ignore_index,
            gamma=args.focal_gamma,
            ce_weight=args.ce_weight,
            dice_weight=args.dice_weight,
            ohem_thresh=args.ohem_thresh,
            ohem_min_kept=args.ohem_min_kept,
        )

        logging.info(f"Loss function: {args.loss_type}")
        if args.loss_type in ['focal', 'focal_dice']:
            logging.info(f"  - Focal gamma: {args.focal_gamma}")
        if args.loss_type in ['ce_dice', 'focal_dice']:
            logging.info(
                f"  - CE weight: {args.ce_weight}, Dice weight: {args.dice_weight}")
        if args.loss_type == 'ohem':
            logging.info(
                f"  - OHEM thresh: {args.ohem_thresh}, min_kept: {args.ohem_min_kept}")

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

    # Training loop
    best_miou = 0.0
    history = {'train': [], 'val': []}

    logging.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"{'='*60}")
        logging.info(f"Epoch {epoch}/{args.epochs}")
        logging.info(f"{'='*60}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, num_classes=num_classes, ignore_index=ignore_index,
            aux_weight=args.aux_weight,
            epoch=epoch, total_epochs=args.epochs, is_edl=is_edl,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device,
            num_classes=num_classes, ignore_index=ignore_index,
            epoch=epoch, total_epochs=args.epochs, is_edl=is_edl,
        )

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        logging.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                     f"mIoU: {train_metrics['mean_iou']:.4f}, "
                     f"Acc: {train_metrics['accuracy']:.4f}")
        logging.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                     f"mIoU: {val_metrics['mean_iou']:.4f}, "
                     f"Acc: {val_metrics['accuracy']:.4f}")
        logging.info(f"LR: {current_lr:.6f}")

        # Class-wise IoU
        class_iou_str = ", ".join([f"{name}: {iou:.4f}"
                                   for name, iou in zip(class_names, val_metrics['class_iou'])])
        logging.info(f"Class IoU - {class_iou_str}")

        # Save history
        train_metrics['lr'] = current_lr
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Save best model (to checkpoint_dir - NAS)
        if val_metrics['mean_iou'] > best_miou:
            best_miou = val_metrics['mean_iou']
            logging.info(f"New best mIoU: {best_miou:.4f}")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_miou': best_miou,
                'args': vars(args),
                'config': {
                    'model': args.model,
                    'dataset': args.dataset,
                    'in_channels': in_channels,
                    'num_classes': num_classes,
                    'bands': bands,
                    'class_names': class_names,
                }
            }, checkpoint_dir / 'best_model.pth')

        # Save periodic checkpoint (to checkpoint_dir - NAS)
        if epoch % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_miou': best_miou,
                'args': vars(args),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')

        # 매 에폭마다 history 저장 (중간에 종료되어도 복구 가능)
        save_history(history, output_dir)

    # Final evaluation on test set
    logging.info(f"{'='*60}")
    logging.info("Final Evaluation on Test Set")
    logging.info(f"{'='*60}")

    # Load best model (from checkpoint_dir)
    best_model_path = checkpoint_dir / 'best_model.pth'
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded best model from: {best_model_path}")

    test_metrics = validate(
        model, test_loader, criterion, device,
        num_classes=num_classes, ignore_index=ignore_index,
        epoch=args.epochs, total_epochs=args.epochs, is_edl=is_edl,
    )

    logging.info(f"Test Results:")
    logging.info(f"  Loss: {test_metrics['loss']:.4f}")
    logging.info(f"  mIoU: {test_metrics['mean_iou']:.4f}")
    logging.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logging.info(f"Class-wise IoU:")
    for name, iou in zip(class_names, test_metrics['class_iou']):
        logging.info(f"  {name}: {iou:.4f}")

    # 최종 결과 저장 (log_dir에 저장)
    final_results = {
        'best_val_miou': best_miou,
        'test_loss': test_metrics['loss'],
        'test_miou': test_metrics['mean_iou'],
        'test_accuracy': test_metrics['accuracy'],
        'test_class_iou': {name: float(iou) for name, iou in zip(class_names, test_metrics['class_iou'])},
        'best_epoch': checkpoint['epoch'],
        'total_epochs': args.epochs,
        'checkpoint_path': str(best_model_path),
    }

    results_file = output_dir / 'final_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    logging.info(f"Final results saved: {results_file}")

    logging.info(f"{'='*60}")
    logging.info(f"Training Complete!")
    logging.info(f"Best validation mIoU: {best_miou:.4f}")
    logging.info(f"Test mIoU: {test_metrics['mean_iou']:.4f}")
    logging.info(f"Log directory: {output_dir}")
    logging.info(f"Checkpoint directory: {checkpoint_dir}")
    logging.info(f"{'='*60}")

    return final_results


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
                        choices=['unet', 'deeplabv3plus', 'cdnetv1', 'cdnetv2',
                                 'hrcloudnet', 'vim_tiny', 'vim_small', 'vim_base'],
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

    # VisionMamba specific
    parser.add_argument('--decoder_type', type=str, default='unet',
                        choices=['unet', 'deeplab'],
                        help='Decoder type for VisionMamba (unet or deeplab)')
    parser.add_argument('--head_type', type=str, default='standard',
                        choices=['standard', 'edl'],
                        help='Head type (standard or edl for uncertainty estimation)')
    parser.add_argument('--edl_annealing_epochs', type=int, default=1,
                        help='EDL KL annealing epochs')
    parser.add_argument('--edl_lambda_kl', type=float, default=0.1,
                        help='EDL KL divergence weight')

    # Loss function
    parser.add_argument('--loss_type', type=str, default='focal_dice',
                        choices=['ce', 'weighted_ce', 'focal',
                                 'dice', 'ce_dice', 'focal_dice', 'ohem'],
                        help='Loss function type. Recommended: focal_dice for class imbalance')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for loss function')
    parser.add_argument('--no_class_weights', dest='use_class_weights', action='store_false',
                        help='Do not use class weights')
    parser.add_argument('--class_weight_method', type=str, default='manual',
                        choices=['manual', 'inverse_freq',
                                 'sqrt_inverse_freq', 'effective_num'],
                        help='Method to compute class weights')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (focusing parameter)')
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='CrossEntropy weight in combined loss')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Dice weight in combined loss')
    parser.add_argument('--ohem_thresh', type=float, default=0.7,
                        help='OHEM hard example threshold')
    parser.add_argument('--ohem_min_kept', type=int, default=100000,
                        help='OHEM minimum pixels to keep')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
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
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device id to use (e.g., 0, 1). If None, uses cuda:0 if available')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers (default: 8 for NAS I/O)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for logs, config, and history (local)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory for model weights (NAS). If None, uses output_dir')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Save checkpoint frequency')

    # Data loading optimization
    parser.add_argument('--preload', action='store_true',
                        help='Preload all data to memory for faster training (requires more RAM)')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Number of batches to prefetch per worker (default: 4)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic mode (slower but reproducible)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
