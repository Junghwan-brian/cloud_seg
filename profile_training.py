"""
학습 병목 지점 프로파일링 스크립트

각 데이터셋과 학습 루프의 어느 부분에서 시간이 소요되는지 측정합니다.
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# Datasets
from l8biome_dataset import L8BiomeDataset
from cloudsen12_dataset import CloudSEN12Dataset
from cloud38_95_dataset import Cloud38Dataset, Cloud95Dataset

# Models
from models.modeling import get_model

# Path utilities
from utils.paths import get_nas_path, detect_nas_base


class Timer:
    """간단한 타이머 클래스"""
    def __init__(self):
        self.times = defaultdict(list)
        self._start_times = {}
    
    def start(self, name):
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name):
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.times[name].append(elapsed)
            del self._start_times[name]
            return elapsed
        return 0
    
    def summary(self):
        print("\n" + "=" * 70)
        print("타이밍 요약 (밀리초)")
        print("=" * 70)
        for name, times in sorted(self.times.items()):
            avg = np.mean(times) * 1000
            std = np.std(times) * 1000
            total = np.sum(times) * 1000
            print(f"{name:40s}: avg={avg:8.2f}ms, std={std:8.2f}ms, total={total:10.2f}ms, count={len(times)}")
        print("=" * 70)


def profile_dataloader(dataset_name: str, num_batches: int = 50, batch_size: int = 4, 
                       num_workers: int = 4):
    """DataLoader 성능 프로파일링"""
    timer = Timer()
    
    print(f"\n{'=' * 70}")
    print(f"데이터셋 프로파일링: {dataset_name}")
    print(f"배치 크기: {batch_size}, 워커 수: {num_workers}, 배치 수: {num_batches}")
    print(f"{'=' * 70}")
    
    # 데이터셋 생성
    timer.start("dataset_init")
    nas_base = detect_nas_base()
    
    if dataset_name == 'l8biome':
        dataset = L8BiomeDataset(
            data_dir=get_nas_path('l8biome_extracted/l8biome'),
            split='train',
            patch_size=512,
            bands=list(range(1, 12)),
            normalize=True,
            use_cache=True,
        )
    elif dataset_name == 'cloudsen12_l2a':
        dataset = CloudSEN12Dataset(
            taco_dir=get_nas_path('cloudsen12-l2a'),
            split='train',
            bands=list(range(1, 15)),
            level='l2a',
            normalize=True,
            patch_size=512,
            use_cache=True,
        )
    elif dataset_name == 'cloud38':
        dataset = Cloud38Dataset(
            root=get_nas_path('38-cloud'),
            split='train',
            bands=['red', 'green', 'blue', 'nir'],
            normalize=True,
        )
    elif dataset_name == 'cloud95':
        dataset = Cloud95Dataset(
            root_38cloud=get_nas_path('38-cloud'),
            root_95cloud=get_nas_path('95-cloud'),
            split='train',
            bands=['red', 'green', 'blue', 'nir'],
            normalize=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    timer.stop("dataset_init")
    print(f"데이터셋 크기: {len(dataset)}")
    
    # DataLoader 생성
    timer.start("dataloader_init")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    timer.stop("dataloader_init")
    
    # 단일 샘플 로딩 시간 측정
    print("\n[1] 단일 샘플 로딩 시간 측정 (첫 10개)...")
    for i in range(min(10, len(dataset))):
        timer.start("single_sample_load")
        _ = dataset[i]
        timer.stop("single_sample_load")
    
    # DataLoader 반복 시간 측정
    print(f"\n[2] DataLoader 반복 시간 측정 ({num_batches} 배치)...")
    
    iterator = iter(loader)
    for i in range(num_batches):
        timer.start("batch_load")
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        timer.stop("batch_load")
        
        images, targets = batch
        
        # GPU 전송 시간
        timer.start("to_gpu")
        images = images.cuda()
        targets = targets.cuda()
        torch.cuda.synchronize()
        timer.stop("to_gpu")
    
    timer.summary()
    return timer


def profile_training_loop(dataset_name: str, model_name: str = 'unet', 
                          num_batches: int = 20, batch_size: int = 4,
                          num_workers: int = 4):
    """전체 학습 루프 프로파일링"""
    timer = Timer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'=' * 70}")
    print(f"학습 루프 프로파일링: {dataset_name} + {model_name}")
    print(f"디바이스: {device}")
    print(f"{'=' * 70}")
    
    # 데이터셋 설정
    nas_base = detect_nas_base()
    
    if dataset_name == 'l8biome':
        in_channels = 11
        num_classes = 4
        ignore_index = 255
        dataset = L8BiomeDataset(
            data_dir=get_nas_path('l8biome_extracted/l8biome'),
            split='train',
            patch_size=512,
            bands=list(range(1, 12)),
            normalize=True,
            use_cache=True,
        )
    elif dataset_name == 'cloudsen12_l2a':
        in_channels = 14
        num_classes = 4
        ignore_index = 255
        dataset = CloudSEN12Dataset(
            taco_dir=get_nas_path('cloudsen12-l2a'),
            split='train',
            bands=list(range(1, 15)),
            level='l2a',
            normalize=True,
            patch_size=512,
            use_cache=True,
        )
    elif dataset_name == 'cloud95':
        in_channels = 4
        num_classes = 2
        ignore_index = None
        dataset = Cloud95Dataset(
            root_38cloud=get_nas_path('38-cloud'),
            root_95cloud=get_nas_path('95-cloud'),
            split='train',
            bands=['red', 'green', 'blue', 'nir'],
            normalize=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    # 모델
    timer.start("model_init")
    model = get_model(model_name, in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)
    timer.stop("model_init")
    
    # 손실 함수 & 옵티마이저
    if ignore_index is not None:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    
    print(f"\n학습 루프 프로파일링 ({num_batches} 배치)...")
    
    iterator = iter(loader)
    total_data_time = 0
    total_forward_time = 0
    total_backward_time = 0
    total_metrics_time = 0
    
    for i in range(num_batches):
        # 데이터 로딩
        timer.start("data_loading")
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        
        images, targets = batch
        timer.stop("data_loading")
        
        # GPU 전송
        timer.start("to_device")
        images = images.to(device)
        targets = targets.to(device)
        torch.cuda.synchronize()
        timer.stop("to_device")
        
        # Forward pass
        timer.start("forward")
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            elif isinstance(outputs, dict):
                outputs = outputs['out']
            loss = criterion(outputs, targets)
        torch.cuda.synchronize()
        timer.stop("forward")
        
        # Backward pass
        timer.start("backward")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        timer.stop("backward")
        
        # Metrics 계산 (train.py의 compute_metrics 시뮬레이션)
        timer.start("metrics")
        with torch.no_grad():
            pred = outputs.argmax(1)
            pred_flat = pred.flatten()
            target_flat = targets.flatten()
            
            if ignore_index is not None:
                valid_mask = target_flat != ignore_index
                pred_flat = pred_flat[valid_mask]
                target_flat = target_flat[valid_mask]
            
            # bincount (CPU로 이동 필요)
            indices = target_flat.long() * num_classes + pred_flat.long()
            confusion_matrix = torch.bincount(
                indices.cpu(),
                minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes).float().to(device)
            
            intersection = torch.diag(confusion_matrix)
            union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
            iou = intersection / (union + 1e-10)
            mean_iou = iou.mean()
        torch.cuda.synchronize()
        timer.stop("metrics")
        
        if (i + 1) % 10 == 0:
            print(f"  배치 {i+1}/{num_batches} 완료")
    
    timer.summary()
    
    # 병목 분석
    print("\n" + "=" * 70)
    print("병목 분석")
    print("=" * 70)
    
    total_time = sum(np.sum(times) for times in timer.times.values())
    
    categories = {
        'data_loading': '데이터 로딩',
        'to_device': 'GPU 전송',
        'forward': 'Forward Pass',
        'backward': 'Backward Pass',
        'metrics': '메트릭 계산',
    }
    
    for key, name in categories.items():
        if key in timer.times:
            time_sum = np.sum(timer.times[key])
            pct = time_sum / total_time * 100
            print(f"{name:20s}: {pct:6.2f}% ({time_sum*1000:.2f}ms)")
    
    return timer


def compare_num_workers(dataset_name: str, workers_list: list = [0, 2, 4, 8, 16],
                        num_batches: int = 30, batch_size: int = 4):
    """다양한 num_workers 값에 따른 성능 비교"""
    print(f"\n{'=' * 70}")
    print(f"num_workers 비교: {dataset_name}")
    print(f"{'=' * 70}")
    
    results = {}
    
    for num_workers in workers_list:
        print(f"\n--- num_workers = {num_workers} ---")
        
        nas_base = detect_nas_base()
        
        if dataset_name == 'cloud95':
            dataset = Cloud95Dataset(
                root_38cloud=get_nas_path('38-cloud'),
                root_95cloud=get_nas_path('95-cloud'),
                split='train',
                bands=['red', 'green', 'blue', 'nir'],
                normalize=True,
            )
        elif dataset_name == 'l8biome':
            dataset = L8BiomeDataset(
                data_dir=get_nas_path('l8biome_extracted/l8biome'),
                split='train',
                patch_size=512,
                bands=list(range(1, 12)),
                normalize=True,
                use_cache=True,
            )
        elif dataset_name == 'cloudsen12_l2a':
            dataset = CloudSEN12Dataset(
                taco_dir=get_nas_path('cloudsen12-l2a'),
                split='train',
                bands=list(range(1, 15)),
                level='l2a',
                normalize=True,
                patch_size=512,
                use_cache=True,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        
        # 웜업
        iterator = iter(loader)
        for _ in range(3):
            try:
                _ = next(iterator)
            except StopIteration:
                break
        
        # 측정
        times = []
        iterator = iter(loader)
        for i in range(num_batches):
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        results[num_workers] = (avg_time, std_time)
        print(f"평균 배치 로딩 시간: {avg_time:.2f}ms (±{std_time:.2f}ms)")
    
    print(f"\n{'=' * 70}")
    print("결과 요약")
    print(f"{'=' * 70}")
    for workers, (avg, std) in sorted(results.items()):
        print(f"num_workers={workers:2d}: {avg:8.2f}ms (±{std:.2f}ms)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Training Performance Profiler')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['dataloader', 'training', 'workers', 'all'],
                        help='Profiling mode')
    parser.add_argument('--dataset', type=str, default='cloud95',
                        choices=['l8biome', 'cloudsen12_l2a', 'cloud38', 'cloud95'],
                        help='Dataset to profile')
    parser.add_argument('--model', type=str, default='unet',
                        help='Model to profile')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--num_batches', type=int, default=30,
                        help='Number of batches to profile')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("학습 성능 프로파일러")
    print("=" * 70)
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    if args.mode == 'dataloader' or args.mode == 'all':
        profile_dataloader(
            args.dataset, 
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    if args.mode == 'training' or args.mode == 'all':
        profile_training_loop(
            args.dataset,
            args.model,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    if args.mode == 'workers' or args.mode == 'all':
        compare_num_workers(
            args.dataset,
            workers_list=[0, 2, 4, 8],
            num_batches=args.num_batches,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()
