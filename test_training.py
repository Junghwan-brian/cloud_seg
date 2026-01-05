#!/usr/bin/env python
"""
Training Validation Script

각 모델과 데이터셋 조합에서 훈련이 제대로 돌아가는지 검증하는 스크립트.
전체 학습이 아닌 1 iteration만 실행하여 빠르게 검증합니다.
"""

import os
import sys
import torch
import traceback
from datetime import datetime

# 워닝 억제
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.modeling import get_model, list_models
from train import get_dataset, get_dataset_config

# =============================================================================
# Test Configuration
# =============================================================================

# 테스트할 모델들 (vim 모델은 의존성 이슈로 별도 테스트)
MODELS_TO_TEST = [
    'unet',
    'deeplabv3plus', 
    'cdnetv1',
    'cdnetv2',
    'hrcloudnet',
]

# 테스트할 데이터셋들
DATASETS_TO_TEST = [
    'l8biome',
    'cloudsen12_l1c',
    'cloud38',
    'cloud95',
]

# VIM 모델 (별도 테스트)
VIM_MODELS = ['vim_tiny', 'vim_small', 'vim_base']


def test_model_creation(model_name, in_channels, num_classes, device):
    """모델 생성 테스트"""
    try:
        model = get_model(
            model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained_backbone=True,
        )
        model = model.to(device)
        
        # Forward pass 테스트
        dummy_input = torch.randn(2, in_channels, 256, 256).to(device)
        with torch.no_grad():
            model.eval()
            output = model(dummy_input)
            
        if isinstance(output, tuple):
            output = output[0]
        
        expected_shape = (2, num_classes, 256, 256)
        if output.shape != expected_shape:
            return False, f"Output shape mismatch: {output.shape} vs {expected_shape}"
        
        del model, dummy_input, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True, "OK"
    except Exception as e:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def test_training_iteration(model_name, dataset_name, device, gpu_id=None):
    """단일 훈련 iteration 테스트"""
    try:
        # 데이터셋 설정 가져오기
        config = get_dataset_config()[dataset_name]
        num_classes = config['num_classes']
        default_bands = config['default_bands']
        in_channels = len(default_bands)
        ignore_index = config['ignore_index']
        
        # 모델 생성
        model = get_model(
            model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained_backbone=True,
        )
        model = model.to(device)
        model.train()
        
        # 데이터셋 로드 (단 1개 배치만)
        dataset = get_dataset(dataset_name, 'train', bands=default_bands, patch_size=256)
        
        if len(dataset) == 0:
            return False, "Empty dataset"
        
        # 데이터로더 (배치 크기 2로 테스트)
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        # 1 iteration 훈련
        images, targets = next(iter(loader))
        images = images.to(device)
        targets = targets.to(device)
        
        # 타겟 값 범위 확인 및 클램핑
        if ignore_index is not None:
            valid_mask = targets != ignore_index
            targets_check = targets[valid_mask]
        else:
            targets_check = targets
        
        # 범위 확인
        min_val, max_val = targets_check.min().item(), targets_check.max().item()
        if min_val < 0 or max_val >= num_classes:
            return False, f"Invalid target values: min={min_val}, max={max_val}, num_classes={num_classes}"
        
        # Forward
        outputs = model(images)
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
        else:
            main_out = outputs
            aux_out = None
        
        # Loss 계산
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100
        )
        loss = criterion(main_out, targets)
        
        if aux_out is not None:
            loss += 0.4 * criterion(aux_out, targets)
        
        # Backward
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        
        # 메모리 정리
        del model, images, targets, outputs, main_out, loss, optimizer
        if aux_out is not None:
            del aux_out
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True, f"loss={loss_val:.4f}"
        
    except Exception as e:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        traceback.print_exc()
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def test_vim_model(model_name, device):
    """VIM 모델 테스트 (별도 처리)"""
    try:
        from models.vim_seg import VimSeg, MAMBA_AVAILABLE
        
        if not MAMBA_AVAILABLE:
            return False, "Mamba not available"
        
        # 간단한 설정으로 테스트
        model = VimSeg(
            in_channels=3,
            num_classes=4,
            backbone=model_name.replace('vim_', ''),
            decoder_type='unet',
            head_type='standard',
            pretrained=True,
            img_size=224,
        )
        model = model.to(device)
        model.eval()
        
        # Forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        if isinstance(output, dict):
            output = output['out']
        
        output_shape = output.shape
        
        # 메모리 정리
        del model, dummy_input, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True, f"output_shape={output_shape}"
        
    except Exception as e:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def main():
    print("=" * 70)
    print("Training Validation Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # GPU 설정
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Device: CPU (WARNING: Training will be slow)")
    
    print()
    
    # ==========================================================================
    # Test 1: 모델 생성 테스트
    # ==========================================================================
    print("-" * 70)
    print("Test 1: Model Creation")
    print("-" * 70)
    
    model_results = []
    for model_name in MODELS_TO_TEST:
        success, msg = test_model_creation(model_name, in_channels=4, num_classes=4, device=device)
        status = "✓" if success else "✗"
        print(f"  {status} {model_name:20s} : {msg}")
        model_results.append((model_name, success))
    
    print()
    
    # ==========================================================================
    # Test 2: 데이터셋별 훈련 iteration 테스트
    # ==========================================================================
    print("-" * 70)
    print("Test 2: Training Iteration (Model + Dataset)")
    print("-" * 70)
    
    iteration_results = []
    
    # 각 데이터셋에서 대표 모델들 테스트
    test_combinations = [
        # (model, dataset)
        ('unet', 'l8biome'),
        ('unet', 'cloudsen12_l1c'),
        ('unet', 'cloud38'),
        ('unet', 'cloud95'),
        ('deeplabv3plus', 'l8biome'),
        ('deeplabv3plus', 'cloud95'),
        ('cdnetv1', 'l8biome'),
        ('cdnetv2', 'l8biome'),
        ('hrcloudnet', 'cloud38'),
    ]
    
    for model_name, dataset_name in test_combinations:
        print(f"  Testing: {model_name} + {dataset_name}...", end=" ", flush=True)
        success, msg = test_training_iteration(model_name, dataset_name, device)
        status = "✓" if success else "✗"
        print(f"{status} {msg}")
        iteration_results.append((model_name, dataset_name, success, msg))
    
    print()
    
    # ==========================================================================
    # Test 3: VIM 모델 테스트 (선택적)
    # ==========================================================================
    print("-" * 70)
    print("Test 3: VIM Models (optional)")
    print("-" * 70)
    
    vim_results = []
    for model_name in VIM_MODELS:
        print(f"  Testing: {model_name}...", end=" ", flush=True)
        success, msg = test_vim_model(model_name, device)
        status = "✓" if success else "✗"
        print(f"{status} {msg}")
        vim_results.append((model_name, success))
    
    print()
    
    # ==========================================================================
    # 결과 요약
    # ==========================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    # 모델 생성 결과
    model_pass = sum(1 for _, s in model_results if s)
    print(f"\nModel Creation: {model_pass}/{len(model_results)} passed")
    
    # 훈련 iteration 결과
    iter_pass = sum(1 for _, _, s, _ in iteration_results if s)
    print(f"Training Iteration: {iter_pass}/{len(iteration_results)} passed")
    
    # VIM 모델 결과
    vim_pass = sum(1 for _, s in vim_results if s)
    print(f"VIM Models: {vim_pass}/{len(vim_results)} passed")
    
    # 실패 항목 출력
    failed_items = []
    for name, success in model_results:
        if not success:
            failed_items.append(f"Model creation: {name}")
    for model, dataset, success, msg in iteration_results:
        if not success:
            failed_items.append(f"Training: {model} + {dataset} ({msg})")
    for name, success in vim_results:
        if not success:
            failed_items.append(f"VIM: {name}")
    
    if failed_items:
        print(f"\nFailed items ({len(failed_items)}):")
        for item in failed_items:
            print(f"  - {item}")
    else:
        print("\n✓ All tests passed!")
    
    print()
    print("=" * 70)
    
    return len(failed_items) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
