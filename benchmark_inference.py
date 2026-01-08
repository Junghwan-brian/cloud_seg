"""
Cloud Segmentation Inference Benchmark

추론 속도 테스트를 위한 벤치마크 스크립트
사용법:
    python benchmark_inference.py --model unet --height 512 --width 512 --channels 3
    python benchmark_inference.py --model deeplabv3plus --height 256 --width 256 --channels 4
    python benchmark_inference.py --model vim_tiny --height 256 --width 256 --channels 4
    python benchmark_inference.py --model vit_small --height 512 --width 512 --channels 3
    python benchmark_inference.py --model all --batch_size 4 --num_runs 100
"""

import argparse
import time
import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Models
from models.modeling import get_model, list_models


# =============================================================================
# Benchmark Utilities
# =============================================================================

def get_model_info(model: nn.Module) -> Dict[str, int]:
    """
    모델 정보 반환

    Args:
        model: PyTorch 모델

    Returns:
        dict: total_params, trainable_params
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
    }


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    GPU 메모리 사용량 반환 (GB 단위)

    Returns:
        dict: allocated, reserved, max_allocated
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}


def measure_inference_time(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 100,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    추론 시간 측정

    Args:
        model: PyTorch 모델
        input_tensor: 입력 텐서
        num_warmup: 웜업 횟수
        num_runs: 실제 측정 횟수
        use_amp: Mixed precision 사용 여부

    Returns:
        dict: 평균, 표준편차, 최소, 최대, FPS 등 통계
    """
    model.eval()
    device = next(model.parameters()).device
    is_cuda = device.type == 'cuda'

    # Warmup
    print(f"  Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            if use_amp and is_cuda:
                with torch.cuda.amp.autocast():
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)

    # Sync before measurement
    if is_cuda:
        torch.cuda.synchronize()

    # Reset max memory stats
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Actual measurement
    print(f"  Running benchmark ({num_runs} runs)...")
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            if is_cuda:
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            if use_amp and is_cuda:
                with torch.cuda.amp.autocast():
                    outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)

            if is_cuda:
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append(end_time - start_time)

    times = np.array(times)
    batch_size = input_tensor.size(0)

    return {
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'min_ms': times.min() * 1000,
        'max_ms': times.max() * 1000,
        'median_ms': np.median(times) * 1000,
        'fps': batch_size / times.mean(),
        'total_images': num_runs * batch_size,
        'total_time_s': times.sum(),
    }


def format_number(num: int) -> str:
    """숫자를 읽기 쉽게 포맷팅"""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_model(
    model_name: str,
    height: int,
    width: int,
    in_channels: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 100,
    use_amp: bool = False,
    decoder_type: str = 'unet',
    head_type: str = 'standard',
    output_stride: int = 16,
) -> Optional[Dict]:
    """
    단일 모델 벤치마크

    Args:
        model_name: 모델 이름
        height, width: 입력 이미지 크기
        in_channels: 입력 채널 수
        num_classes: 출력 클래스 수
        batch_size: 배치 크기
        device: 연산 장치
        num_warmup: 웜업 횟수
        num_runs: 측정 횟수
        use_amp: Mixed precision 사용
        decoder_type: VisionMamba 디코더 타입
        head_type: VisionMamba 헤드 타입
        output_stride: DeepLabV3+ output stride

    Returns:
        dict: 벤치마크 결과 또는 None (실패 시)
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    try:
        # 모델 생성 (pretrained=False로 설정)
        print(f"  Creating model...")
        model = get_model(
            model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained_backbone=False,  # No pretrained weights
            decoder_type=decoder_type,
            head_type=head_type,
            output_stride=output_stride,
        )
        model = model.to(device)
        model.eval()

        # 모델 정보
        model_info = get_model_info(model)
        print(
            f"  Total parameters: {format_number(model_info['total_params'])}")
        print(
            f"  Trainable parameters: {format_number(model_info['trainable_params'])}")

        # 입력 텐서 생성
        input_tensor = torch.randn(
            batch_size, in_channels, height, width, device=device)
        print(f"  Input shape: {list(input_tensor.shape)}")

        # 출력 형태 확인
        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)

        # 출력 형태 파싱
        if isinstance(outputs, tuple):
            output_shapes = [
                list(o.shape) if o is not None else None for o in outputs]
            print(f"  Output shapes: {output_shapes}")
        elif isinstance(outputs, dict):
            output_shapes = {k: list(v.shape) if isinstance(v, torch.Tensor) else v
                             for k, v in outputs.items()}
            print(f"  Output shapes: {output_shapes}")
        else:
            print(f"  Output shape: {list(outputs.shape)}")

        # GPU 메모리 (추론 전)
        if device.type == 'cuda':
            mem_before = get_gpu_memory_usage()
            torch.cuda.reset_peak_memory_stats()

        # 추론 시간 측정
        timing_results = measure_inference_time(
            model, input_tensor,
            num_warmup=num_warmup,
            num_runs=num_runs,
            use_amp=use_amp,
        )

        # GPU 메모리 (추론 후)
        if device.type == 'cuda':
            mem_after = get_gpu_memory_usage()
            peak_memory_gb = mem_after['max_allocated_gb']
        else:
            peak_memory_gb = 0

        # 결과 정리
        result = {
            'model_name': model_name,
            'input_shape': [batch_size, in_channels, height, width],
            'num_classes': num_classes,
            **model_info,
            **timing_results,
            'peak_memory_gb': peak_memory_gb,
            'device': str(device),
            'use_amp': use_amp,
        }

        # 결과 출력
        print(f"\n  Results:")
        print(f"    Mean inference time: {timing_results['mean_ms']:.3f} ms")
        print(f"    Std deviation: {timing_results['std_ms']:.3f} ms")
        print(
            f"    Min / Max: {timing_results['min_ms']:.3f} / {timing_results['max_ms']:.3f} ms")
        print(f"    Median: {timing_results['median_ms']:.3f} ms")
        print(f"    FPS: {timing_results['fps']:.2f}")
        if device.type == 'cuda':
            print(f"    Peak GPU memory: {peak_memory_gb:.3f} GB")

        # 메모리 정리
        del model, input_tensor, outputs
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return None


def benchmark_all_models(
    height: int,
    width: int,
    in_channels: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 100,
    use_amp: bool = False,
    decoder_type: str = 'unet',
    head_type: str = 'standard',
    output_stride: int = 16,
) -> List[Dict]:
    """
    모든 모델 벤치마크

    Returns:
        list: 각 모델의 벤치마크 결과 리스트
    """
    all_models = list_models()
    results = []

    print(f"\n{'#'*60}")
    print(f"# Benchmarking ALL models ({len(all_models)} models)")
    print(f"# Input: {batch_size}x{in_channels}x{height}x{width}")
    print(f"# Device: {device}")
    print(f"# AMP: {use_amp}")
    print(f"{'#'*60}")

    for model_name in all_models:
        result = benchmark_model(
            model_name=model_name,
            height=height,
            width=width,
            in_channels=in_channels,
            num_classes=num_classes,
            batch_size=batch_size,
            device=device,
            num_warmup=num_warmup,
            num_runs=num_runs,
            use_amp=use_amp,
            decoder_type=decoder_type,
            head_type=head_type,
            output_stride=output_stride,
        )
        if result is not None:
            results.append(result)

    return results


def print_summary(results: List[Dict]):
    """결과 요약 출력"""
    if not results:
        print("No results to summarize.")
        return

    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    # 헤더
    header = f"{'Model':<20} {'Params':<12} {'Mean (ms)':<12} {'FPS':<10} {'Memory (GB)':<12}"
    print(header)
    print("-" * 80)

    # FPS 기준 정렬
    sorted_results = sorted(results, key=lambda x: x['fps'], reverse=True)

    for r in sorted_results:
        model_name = r['model_name'][:20]
        params = format_number(r['total_params'])
        mean_ms = f"{r['mean_ms']:.2f}"
        fps = f"{r['fps']:.1f}"
        memory = f"{r['peak_memory_gb']:.3f}" if r['peak_memory_gb'] > 0 else "N/A"

        print(f"{model_name:<20} {params:<12} {mean_ms:<12} {fps:<10} {memory:<12}")

    print("-" * 80)

    # 가장 빠른/느린 모델
    fastest = sorted_results[0]
    slowest = sorted_results[-1]
    print(f"\nFastest: {fastest['model_name']} ({fastest['fps']:.1f} FPS)")
    print(f"Slowest: {slowest['model_name']} ({slowest['fps']:.1f} FPS)")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Cloud Segmentation Inference Benchmark')

    # Input specification
    parser.add_argument('--height', type=int, default=512,
                        help='Input image height (default: 512)')
    parser.add_argument('--width', type=int, default=512,
                        help='Input image width (default: 512)')
    parser.add_argument('--channels', type=int, default=3,
                        help='Input channels (default: 3)')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of output classes (default: 4)')

    # Model selection
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'unet', 'deeplabv3plus', 'cdnetv1', 'cdnetv2',
                                 'hrcloudnet', 'vim_tiny', 'vim_small', 'vim_base',
                                 'vit_nano', 'vit_small'],
                        help='Model to benchmark (default: all)')

    # Benchmark settings
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--num_warmup', type=int, default=1,
                        help='Number of warmup runs (default: 1)')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of benchmark runs (default: 1)')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')

    # Device
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device id to use. If None, uses cuda:0 if available')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference')

    # Model-specific options
    parser.add_argument('--decoder_type', type=str, default='deeplab',
                        choices=['unet', 'deeplab'],
                        help='Decoder type for VisionMamba/ViT (default: unet)')
    parser.add_argument('--head_type', type=str, default='standard',
                        choices=['standard', 'edl'],
                        help='Head type for VisionMamba (default: standard)')
    parser.add_argument('--output_stride', type=int, default=16,
                        help='Output stride for DeepLabV3+ (default: 16)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    if args.cpu:
        device = torch.device('cpu')
    elif args.gpu is not None:
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

    # Print configuration
    print(f"\n{'#'*60}")
    print("# Cloud Segmentation Inference Benchmark")
    print(f"{'#'*60}")
    print(
        f"Input shape: {args.batch_size} x {args.channels} x {args.height} x {args.width}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Device: {device}")
    print(f"AMP: {args.amp}")
    print(f"Warmup runs: {args.num_warmup}")
    print(f"Benchmark runs: {args.num_runs}")
    if args.model.startswith('vim_'):
        print(f"Decoder type: {args.decoder_type}")
        print(f"Head type: {args.head_type}")
    if args.model == 'deeplabv3plus':
        print(f"Output stride: {args.output_stride}")
    print(f"{'#'*60}")

    # Run benchmark
    if args.model == 'all':
        results = benchmark_all_models(
            height=args.height,
            width=args.width,
            in_channels=args.channels,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            device=device,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            use_amp=args.amp,
            decoder_type=args.decoder_type,
            head_type=args.head_type,
            output_stride=args.output_stride,
        )
        print_summary(results)
    else:
        result = benchmark_model(
            model_name=args.model,
            height=args.height,
            width=args.width,
            in_channels=args.channels,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            device=device,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            use_amp=args.amp,
            decoder_type=args.decoder_type,
            head_type=args.head_type,
            output_stride=args.output_stride,
        )
        if result:
            print_summary([result])


if __name__ == '__main__':
    main()
