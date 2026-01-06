#!/usr/bin/env python
"""
Hyperparameter Tuning Script with Multiprocessing

다중 GPU에서 병렬로 실험을 수행하는 스크립트.
각 모델별로 하이퍼파라미터 튜닝을 지원.

사용법:
    # 특정 데이터셋으로 모든 모델 실험
    python run_experiments.py --gpus 0 1 2 3 4 5 6 7 --checkpoint_dir /nas/junghwan/cloud_seg/checkpoints --all_datasets --all_models --include_vim --max_parallel 16

"""

import argparse
import itertools
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import Dict, List, Any, Optional

# =============================================================================
# Model-specific Default Hyperparameter Configurations
# =============================================================================

# 각 모델별 탐색할 하이퍼파라미터 설정
MODEL_HYPERPARAMS = {
    'unet': {
        'lr': [5e-4],
        'batch_size': [8],
        'optimizer': ['adamw'],
        'scheduler': ['cosine'],
        'weight_decay': [1e-4],
    },
    'deeplabv3plus': {
        'lr': [5e-4],
        'batch_size': [8],
        'optimizer': ['adamw'],
        'scheduler': ['cosine'],
        'weight_decay': [1e-4],
        'output_stride': [16],
    },
    'cdnetv1': {
        'lr': [5e-4],
        'batch_size': [8],
        'optimizer': ['adamw'],
        'scheduler': ['cosine'],
        'weight_decay': [1e-4],
    },
    'cdnetv2': {
        'lr': [5e-4],
        'batch_size': [8],
        'optimizer': ['adamw'],
        'scheduler': ['cosine'],
        'weight_decay': [1e-4],
        'aux_weight': [0.2],
    },
    'hrcloudnet': {
        'lr': [5e-4],
        'batch_size': [8],  # HRCloudNet은 메모리 사용량이 큼
        'optimizer': ['adamw'],
        'scheduler': ['cosine'],
        'weight_decay': [1e-4],
    },
    'vim_tiny': {
        'lr': [1e-4, 1e-5],
        'batch_size': [4],
        'optimizer': ['adamw'],
        'scheduler': ['cosine'],
        'weight_decay': [1e-4],
        'decoder_type': ['unet', 'deeplab'],
        'head_type': ['standard', 'edl'],
    },
    'vim_small': {
        'lr': [1e-4, 1e-5],
        'batch_size': [4],
        'optimizer': ['adamw'],
        'scheduler': ['cosine'],
        'weight_decay': [1e-4],
        'decoder_type': ['unet', 'deeplab'],
        'head_type': ['standard', 'edl'],
    },
    # 'vim_base': {
    #     'lr': [1e-4, 1e-5],
    #     'batch_size': [4],  # vim_base는 메모리 사용량이 매우 큼
    #     'optimizer': ['adamw'],
    #     'scheduler': ['cosine'],
    #     'weight_decay': [1e-4],
    #     'decoder_type': ['unet', 'deeplab'],
    #     'head_type': ['standard', 'edl'],
    # },
}

# 데이터셋별 기본 설정
DATASET_DEFAULTS = {
    'l8biome': {'epochs': 10, 'patch_size': 512},
    'cloudsen12_l1c': {'epochs': 10, 'patch_size': 512},
    'cloudsen12_l2a': {'epochs': 10, 'patch_size': 512},
    'cloud38': {'epochs': 10, 'patch_size': 384},
    'cloud95': {'epochs': 10, 'patch_size': 384},
}

# =============================================================================
# Global State for Process Management
# =============================================================================

# Manager for shared state across processes
_manager = None
_should_stop = None
_child_processes = None


def init_globals():
    """Initialize global state for multiprocessing."""
    global _manager, _should_stop, _child_processes
    _manager = Manager()
    _should_stop = _manager.Value('b', False)
    _child_processes = _manager.list()


def cleanup_processes():
    """Terminate all child processes."""
    global _child_processes, _should_stop

    if _should_stop is not None:
        _should_stop.value = True

    if _child_processes is not None:
        for pid in list(_child_processes):
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"[Cleanup] Terminated process {pid}")
            except (ProcessLookupError, OSError):
                pass


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print(
        f"\n[Signal] Received signal {signum}. Terminating all experiments...")
    cleanup_processes()
    sys.exit(1)


# =============================================================================
# Experiment Execution
# =============================================================================

def generate_experiment_configs(
    model: str,
    dataset: str,
    hyperparams: Optional[Dict[str, List[Any]]] = None,
    epochs: Optional[int] = None,
    patch_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    하이퍼파라미터 조합을 생성하여 실험 설정 리스트를 반환.

    Args:
        model: 모델 이름
        dataset: 데이터셋 이름
        hyperparams: 탐색할 하이퍼파라미터 (None이면 MODEL_HYPERPARAMS 사용)
        epochs: 학습 에폭 수
        patch_size: 패치 크기

    Returns:
        실험 설정 딕셔너리 리스트
    """
    # 기본 하이퍼파라미터 설정 가져오기
    if hyperparams is None:
        hyperparams = MODEL_HYPERPARAMS.get(model, MODEL_HYPERPARAMS['unet'])

    # 데이터셋 기본값
    dataset_defaults = DATASET_DEFAULTS.get(
        dataset, {'epochs': 10, 'patch_size': 512})

    if epochs is None:
        epochs = dataset_defaults['epochs']
    if patch_size is None:
        patch_size = dataset_defaults['patch_size']

    # 모든 조합 생성
    keys = list(hyperparams.keys())
    values = [hyperparams[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        config['model'] = model
        config['dataset'] = dataset
        config['epochs'] = epochs
        config['patch_size'] = patch_size
        configs.append(config)

    return configs


def run_single_experiment(args):
    """
    단일 실험 실행 (worker 프로세스에서 호출됨).

    Args:
        args: (config, gpu_id, output_base, checkpoint_base, should_stop, child_processes)

    Returns:
        (config, success, message)
    """
    config, gpu_id, output_base, checkpoint_base, should_stop, child_processes = args

    # 종료 신호 체크
    if should_stop.value:
        return (config, False, "Terminated by user")

    # 실험 이름 생성
    exp_name = f"{config['model']}_{config['dataset']}"
    for k, v in config.items():
        if k not in ['model', 'dataset', 'epochs', 'patch_size']:
            exp_name += f"_{k}{v}"

    # 명령어 생성
    cmd = [
        sys.executable, 'train.py',
        '--model', config['model'],
        '--dataset', config['dataset'],
        '--gpu', str(gpu_id),
        '--epochs', str(config['epochs']),
        '--patch_size', str(config['patch_size']),
        '--output_dir', str(output_base),
    ]

    # checkpoint_dir 추가 (NAS 경로)
    if checkpoint_base is not None:
        cmd.extend(['--checkpoint_dir', str(checkpoint_base)])

    # 하이퍼파라미터 추가
    hp_keys = ['lr', 'batch_size', 'optimizer', 'scheduler', 'weight_decay',
               'output_stride', 'aux_weight', 'decoder_type', 'head_type',
               'edl_annealing_epochs', 'edl_lambda_kl']

    for key in hp_keys:
        if key in config:
            cmd.extend([f'--{key}', str(config[key])])

    # 로그 파일 설정
    log_file = output_base / f"{exp_name}_gpu{gpu_id}.log"

    print(f"[GPU {gpu_id}] Starting: {exp_name}")
    print(f"[GPU {gpu_id}] Log: {log_file}")

    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                preexec_fn=os.setsid  # 새 프로세스 그룹 생성
            )

            # 자식 프로세스 등록
            child_processes.append(process.pid)

            # 프로세스 완료 대기 (주기적으로 종료 신호 체크)
            while process.poll() is None:
                if should_stop.value:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    return (config, False, "Terminated by user")
                time.sleep(1)

            # 자식 프로세스 제거
            try:
                child_processes.remove(process.pid)
            except ValueError:
                pass

            if process.returncode == 0:
                print(f"[GPU {gpu_id}] Completed: {exp_name}")
                return (config, True, "Success")
            else:
                print(
                    f"[GPU {gpu_id}] Failed: {exp_name} (exit code: {process.returncode})")
                return (config, False, f"Exit code: {process.returncode}")

    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {exp_name} - {e}")
        return (config, False, str(e))


def run_experiments(
    configs: List[Dict[str, Any]],
    gpus: List[int],
    max_parallel: int,
    output_base: Path,
    checkpoint_base: Optional[Path] = None,
):
    """
    멀티프로세싱을 사용하여 여러 실험을 병렬 실행.

    Args:
        configs: 실험 설정 리스트
        gpus: 사용할 GPU ID 리스트
        max_parallel: 최대 동시 실행 수
        output_base: 출력 디렉토리 (로그, config 등 - 로컬)
        checkpoint_base: 체크포인트 디렉토리 (모델 - NAS)
    """
    global _should_stop, _child_processes

    # 실험에 GPU 할당 (라운드 로빈)
    experiment_args = []
    for i, config in enumerate(configs):
        gpu_id = gpus[i % len(gpus)]
        experiment_args.append(
            (config, gpu_id, output_base, checkpoint_base, _should_stop, _child_processes))

    print(f"\n{'='*60}")
    print(f"Running {len(configs)} experiments on GPUs {gpus}")
    print(f"Max parallel: {max_parallel}")
    print(f"Log directory: {output_base}")
    if checkpoint_base:
        print(f"Checkpoint directory: {checkpoint_base}")
    print(f"{'='*60}\n")

    # 실험 목록 출력
    for i, (config, gpu_id, _, _, _, _) in enumerate(experiment_args):
        print(f"  [{i+1}] GPU {gpu_id}: {config['model']} on {config['dataset']}")
        params = {k: v for k, v in config.items()
                  if k not in ['model', 'dataset', 'epochs', 'patch_size']}
        print(f"       Params: {params}")
    print()

    # 멀티프로세싱 풀 실행
    results = []

    try:
        with Pool(processes=max_parallel) as pool:
            results = pool.map(run_single_experiment, experiment_args)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received. Cleaning up...")
        cleanup_processes()
        sys.exit(1)

    # 결과 요약
    print(f"\n{'='*60}")
    print("Experiment Results Summary")
    print(f"{'='*60}")

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for config, _, msg in successful:
        print(f"  ✓ {config['model']} on {config['dataset']} - {msg}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for config, _, msg in failed:
            print(f"  ✗ {config['model']} on {config['dataset']} - {msg}")

    print(f"\n{'='*60}")

    return results


# =============================================================================
# Main Function
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Hyperparameter Tuning with Multiprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run UNet hyperparameter tuning on 2 GPUs
  python run_experiments.py --gpus 0 1 --model unet --dataset l8biome

  # Run all models on a single dataset
  python run_experiments.py --gpus 0 1 2 3 --all_models --dataset cloud95

  # Custom hyperparameters
  python run_experiments.py --gpus 0 1 --model deeplabv3plus --lr 1e-3 1e-4 --batch_size 8 16
        """
    )

    # GPU 설정
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='GPU device IDs to use (e.g., 0 1 2)')
    parser.add_argument('--max_parallel', type=int, default=None,
                        help='Maximum number of parallel experiments (default: num GPUs)')

    # 모델/데이터셋 선택
    parser.add_argument('--model', type=str, default=None,
                        choices=['unet', 'deeplabv3plus', 'cdnetv1', 'cdnetv2',
                                 'hrcloudnet', 'vim_tiny', 'vim_small', 'vim_base'],
                        help='Model to tune (if not specified, uses --all_models)')
    parser.add_argument('--all_models', action='store_true',
                        help='Run experiments for all models')
    parser.add_argument('--include_vim', action='store_true',
                        help='Include VisionMamba models when using --all_models (requires more GPU memory)')
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
                        choices=['l8biome', 'cloudsen12_l1c', 'cloudsen12_l2a',
                                 'cloud38', 'cloud95'],
                        help='Dataset(s) to use. If not specified, runs on all datasets')
    parser.add_argument('--all_datasets', action='store_true',
                        help='Run experiments on all datasets')

    # 하이퍼파라미터 (커스텀 오버라이드)
    parser.add_argument('--lr', type=float, nargs='+', default=None,
                        help='Learning rates to try')
    parser.add_argument('--batch_size', type=int, nargs='+', default=None,
                        help='Batch sizes to try')
    parser.add_argument('--optimizer', type=str, nargs='+', default=None,
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizers to try')
    parser.add_argument('--scheduler', type=str, nargs='+', default=None,
                        choices=['none', 'step', 'cosine', 'poly'],
                        help='Schedulers to try')
    parser.add_argument('--weight_decay', type=float, nargs='+', default=None,
                        help='Weight decay values to try')

    # 학습 설정
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Patch size')

    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='./experiment_outputs',
                        help='Base output directory for logs, config (local)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Base checkpoint directory for model weights (NAS). If None, uses output_dir')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: timestamp)')

    # 빠른 테스트
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (reduced hyperparameter space)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 글로벌 상태 초기화
    init_globals()

    # 최대 병렬 수 설정
    if args.max_parallel is None:
        args.max_parallel = len(args.gpus)

    # 출력 디렉토리 설정 (로그 - 로컬)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.exp_name or timestamp
    output_base = Path(args.output_dir) / exp_name
    output_base.mkdir(parents=True, exist_ok=True)

    # 체크포인트 디렉토리 설정 (모델 - NAS)
    if args.checkpoint_dir is not None:
        checkpoint_base = Path(args.checkpoint_dir) / exp_name
        checkpoint_base.mkdir(parents=True, exist_ok=True)
        print(f"[Info] Checkpoints will be saved to: {checkpoint_base}")
    else:
        checkpoint_base = None
        print(
            f"[Info] Checkpoints will be saved to: {output_base} (same as logs)")

    # 모델 목록 결정
    if args.all_models:
        models = ['unet', 'deeplabv3plus', 'cdnetv1', 'cdnetv2', 'hrcloudnet']
        # VisionMamba 모델 포함 (메모리 사용량이 크므로 batch_size 조정 필요)
        if args.include_vim:
            models += ['vim_tiny', 'vim_small']
            print(
                "[Info] VisionMamba models included. Note: These models require more GPU memory.")
    elif args.model:
        models = [args.model]
    else:
        print("Error: Specify --model or --all_models")
        sys.exit(1)

    # 데이터셋 목록 결정
    all_datasets = ['l8biome', 'cloudsen12_l1c',
                    'cloudsen12_l2a', 'cloud38', 'cloud95']
    if args.all_datasets or args.dataset is None:
        datasets = all_datasets
        print(f"[Info] Running on all datasets: {datasets}")
    elif isinstance(args.dataset, list):
        datasets = args.dataset
    else:
        datasets = [args.dataset]

    # 하이퍼파라미터 설정
    all_configs = []

    for model in models:
        for dataset in datasets:
            # 기본 하이퍼파라미터 가져오기
            default_hp = MODEL_HYPERPARAMS.get(
                model, MODEL_HYPERPARAMS['unet']).copy()

            # 커스텀 하이퍼파라미터 오버라이드
            if args.lr is not None:
                default_hp['lr'] = args.lr
            if args.batch_size is not None:
                default_hp['batch_size'] = args.batch_size
            if args.optimizer is not None:
                default_hp['optimizer'] = args.optimizer
            if args.scheduler is not None:
                default_hp['scheduler'] = args.scheduler
            if args.weight_decay is not None:
                default_hp['weight_decay'] = args.weight_decay

            # 빠른 테스트 모드
            if args.quick:
                for key in default_hp:
                    if isinstance(default_hp[key], list) and len(default_hp[key]) > 1:
                        default_hp[key] = [default_hp[key][0]]  # 첫 번째 값만 사용

            # 설정 생성
            configs = generate_experiment_configs(
                model=model,
                dataset=dataset,
                hyperparams=default_hp,
                epochs=args.epochs,
                patch_size=args.patch_size,
            )
            all_configs.extend(configs)

    if not all_configs:
        print("Error: No experiment configurations generated")
        sys.exit(1)

    # 실험 실행
    try:
        results = run_experiments(
            configs=all_configs,
            gpus=args.gpus,
            max_parallel=args.max_parallel,
            output_base=output_base,
            checkpoint_base=checkpoint_base,
        )

        # 결과 저장
        import json
        results_file = output_base / 'experiment_results.json'
        with open(results_file, 'w') as f:
            json.dump([
                {'config': r[0], 'success': r[1], 'message': r[2]}
                for r in results
            ], f, indent=2)
        print(f"\nResults saved to: {results_file}")

    except Exception as e:
        print(f"Error during experiments: {e}")
        cleanup_processes()
        sys.exit(1)

    print("\nAll experiments completed!")


if __name__ == '__main__':
    main()
