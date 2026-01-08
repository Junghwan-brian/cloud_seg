"""
학습 최적화 테스트 스크립트
compute_metrics 병목 현상을 확인하고 개선 효과를 측정합니다.
타임아웃: 30초
"""

import time
import signal
import sys
import torch
import numpy as np


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("시간 초과!")


def compute_metrics_old(pred, target, num_classes, ignore_index=None):
    """기존 느린 버전 (Python for 루프 사용)"""
    pred = pred.flatten()
    target = target.flatten()

    if ignore_index is not None:
        valid_mask = target != ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]

    confusion_matrix = torch.zeros(
        num_classes, num_classes, device=pred.device)
    for t, p in zip(target, pred):  # 🔴 매우 느림!
        confusion_matrix[t.long(), p.long()] += 1

    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
    iou = intersection / (union + 1e-10)
    valid_classes = union > 0
    mean_iou = iou[valid_classes].mean()
    accuracy = intersection.sum() / (confusion_matrix.sum() + 1e-10)

    return {
        'iou': iou.cpu().numpy(),
        'mean_iou': mean_iou.item(),
        'accuracy': accuracy.item(),
    }


def compute_metrics_new(pred, target, num_classes, ignore_index=None):
    """새로운 벡터화된 버전"""
    pred = pred.flatten()
    target = target.flatten()

    if ignore_index is not None:
        valid_mask = target != ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]

    # 벡터화된 Confusion matrix 계산 (for 루프 제거)
    indices = target.long() * num_classes + pred.long()
    confusion_matrix = torch.bincount(
        indices.cpu(),
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).float().to(pred.device)

    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
    iou = intersection / (union + 1e-10)
    valid_classes = union > 0
    mean_iou = iou[valid_classes].mean()
    accuracy = intersection.sum() / (confusion_matrix.sum() + 1e-10)

    return {
        'iou': iou.cpu().numpy(),
        'mean_iou': mean_iou.item(),
        'accuracy': accuracy.item(),
    }


def benchmark_metrics():
    """compute_metrics 함수 벤치마크 - 짧은 버전"""
    print("=" * 60)
    print("compute_metrics 벤치마크 (빠른 테스트)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 작은 크기로 테스트 (64x64, 배치 1)
    small_h, small_w = 64, 64
    num_classes = 4

    pred_small = torch.randint(
        0, num_classes, (1, small_h, small_w), device=device)
    target_small = torch.randint(
        0, num_classes, (1, small_h, small_w), device=device)

    print(f"\n[1] 작은 크기 테스트 ({small_h}x{small_w} = {small_h*small_w:,} 픽셀)")
    print("-" * 40)

    # 새 버전 먼저 (빠름)
    start = time.perf_counter()
    result_new = compute_metrics_new(pred_small, target_small, num_classes)
    time_new_small = time.perf_counter() - start
    print(f"새 버전 (벡터화): {time_new_small*1000:.3f} ms")

    # 기존 버전 (느림)
    start = time.perf_counter()
    result_old = compute_metrics_old(pred_small, target_small, num_classes)
    time_old_small = time.perf_counter() - start
    print(f"기존 버전 (for 루프): {time_old_small*1000:.3f} ms")

    speedup_small = time_old_small / time_new_small
    print(f"속도 향상: {speedup_small:.1f}x")

    # 실제 크기로 새 버전만 테스트
    print(f"\n[2] 실제 크기 테스트 (512x512 x 4 배치 = {4*512*512:,} 픽셀)")
    print("-" * 40)

    pred_large = torch.randint(0, num_classes, (4, 512, 512), device=device)
    target_large = torch.randint(0, num_classes, (4, 512, 512), device=device)
    target_large[target_large == 0] = 255  # ignore_index 시뮬레이션

    # 새 버전 (빠름)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        result = compute_metrics_new(
            pred_large, target_large, num_classes, ignore_index=255)
        times.append(time.perf_counter() - start)

    time_new_large = np.mean(times)
    print(f"새 버전 (벡터화): {time_new_large*1000:.2f} ms (평균 5회)")
    print(f"  mIoU: {result['mean_iou']:.4f}, Acc: {result['accuracy']:.4f}")

    # 기존 버전은 추정만 (너무 오래 걸림)
    # 작은 크기 비율로 추정
    pixels_ratio = (4 * 512 * 512) / (small_h * small_w)
    estimated_old = time_old_small * pixels_ratio

    print(f"\n기존 버전 예상 시간: {estimated_old:.1f}초 ({estimated_old/60:.1f}분)")
    print(f"(64x64 결과 기반 추정: {pixels_ratio:.0f}배 픽셀)")

    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 최적화 결과 요약")
    print("=" * 60)
    estimated_speedup = estimated_old / time_new_large
    print(f"예상 속도 향상: ~{estimated_speedup:.0f}x 빠름")
    print(f"\n배치당 절약 시간: ~{estimated_old - time_new_large:.1f}초")
    print(
        f"1000 배치(~1 에폭) 절약: ~{(estimated_old - time_new_large) * 1000 / 60:.0f}분")

    # 결과 일치 확인 (작은 크기에서)
    print("\n✅ 결과 일치 확인 (64x64):")
    print(
        f"  mIoU 차이: {abs(result_old['mean_iou'] - result_new['mean_iou']):.6f}")
    print(
        f"  Acc 차이: {abs(result_old['accuracy'] - result_new['accuracy']):.6f}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == '__main__':
    # 30초 타임아웃 설정
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)

    try:
        benchmark_metrics()
    except TimeoutError:
        print("\n⚠️ 30초 타임아웃으로 종료됨")
        sys.exit(1)
    finally:
        signal.alarm(0)  # 타이머 해제
