"""
CloudSEN12 Dataset for Cloud Segmentation

로컬 TACO 파일을 사용하여 CloudSEN12 데이터셋을 로드하는 PyTorch Dataset 클래스
L1C (Level-1C) 및 L2A (Level-2A) 데이터 모두 지원

사용 예시:
    from cloudsen12_dataset import CloudSEN12Dataset
    
    # L1C 데이터셋 생성
    train_dataset_l1c = CloudSEN12Dataset(
        taco_dir='/home/telepix_nas/junghwan/cloud_seg/cloudsen12-l1c',
        split='train',
        bands=[4, 3, 2],  # RGB (B4, B3, B2)
        level='l1c',
    )
    
    # L2A 데이터셋 생성
    train_dataset_l2a = CloudSEN12Dataset(
        taco_dir='/home/telepix_nas/junghwan/cloud_seg/cloudsen12-l2a',
        split='train',
        bands=[4, 3, 2],  # RGB (B4, B3, B2)
        level='l2a',
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset_l1c, batch_size=8, shuffle=True)
"""

import glob
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Union, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio as rio
import tacoreader
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# =============================================================================
# 파일 핸들 캐싱을 위한 유틸리티
# =============================================================================

class RasterioFileCache:
    """
    Rasterio 파일 핸들을 캐싱하여 반복적인 파일 열기/닫기 오버헤드를 줄입니다.
    """

    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []

    def get(self, path: str):
        """캐시된 파일 핸들을 반환하거나 새로 엽니다."""
        if path in self._cache:
            # LRU 업데이트
            self._access_order.remove(path)
            self._access_order.append(path)
            return self._cache[path]

        # 캐시가 꽉 차면 가장 오래된 항목 제거
        if len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            self._cache[oldest].close()
            del self._cache[oldest]

        # 새 파일 열기
        src = rio.open(path)
        self._cache[path] = src
        self._access_order.append(path)
        return src

    def close_all(self):
        """모든 캐시된 파일 핸들을 닫습니다."""
        for src in self._cache.values():
            src.close()
        self._cache.clear()
        self._access_order.clear()

    def __del__(self):
        self.close_all()


# 전역 파일 캐시 (프로세스별)
_cloudsen12_file_cache = None


def _get_file_cache() -> RasterioFileCache:
    """전역 파일 캐시를 반환합니다."""
    global _cloudsen12_file_cache
    if _cloudsen12_file_cache is None:
        _cloudsen12_file_cache = RasterioFileCache(max_size=256)
    return _cloudsen12_file_cache


# 기본 TACO 디렉토리 경로
DEFAULT_TACO_DIRS = {
    'l1c': '/home/telepix_nas/junghwan/cloud_seg/cloudsen12-l1c',
    'l2a': '/home/telepix_nas/junghwan/cloud_seg/cloudsen12-l2a',
}


class CloudSEN12Dataset(Dataset):
    """
    CloudSEN12 데이터셋을 위한 PyTorch Dataset 클래스

    로컬에 저장된 TACO 파일들을 읽어 학습/검증/테스트 데이터를 제공합니다.
    L1C (Top-of-Atmosphere) 및 L2A (Surface Reflectance) 데이터 모두 지원.

    Attributes:
        taco_dir: TACO 파일이 저장된 디렉토리 경로
        split: 데이터 분할 ('train', 'validation', 'test', 'all')
        bands: 사용할 밴드 인덱스 리스트 (1-indexed, Sentinel-2 밴드 순서)
        level: 데이터 레벨 ('l1c' 또는 'l2a')
        transform: 이미지에 적용할 변환 함수
        target_transform: 레이블에 적용할 변환 함수
        normalize: 이미지 정규화 여부 (True: 0-1 범위로 정규화)

    Sentinel-2 L1C 밴드 순서 (1-indexed, 총 13개):
        1: B1 (Coastal aerosol, 60m)
        2: B2 (Blue, 10m)
        3: B3 (Green, 10m)
        4: B4 (Red, 10m)
        5: B5 (Vegetation Red Edge, 20m)
        6: B6 (Vegetation Red Edge, 20m)
        7: B7 (Vegetation Red Edge, 20m)
        8: B8 (NIR, 10m)
        9: B8A (Vegetation Red Edge, 20m)
        10: B9 (Water vapour, 60m)
        11: B10 (SWIR - Cirrus, 60m)
        12: B11 (SWIR, 20m)
        13: B12 (SWIR, 20m)

    Sentinel-2 L2A 밴드 순서 (1-indexed, 총 14개):
        1: B1 (Coastal aerosol, 60m)
        2: B2 (Blue, 10m)
        3: B3 (Green, 10m)
        4: B4 (Red, 10m)
        5: B5 (Vegetation Red Edge, 20m)
        6: B6 (Vegetation Red Edge, 20m)
        7: B7 (Vegetation Red Edge, 20m)
        8: B8 (NIR, 10m)
        9: B8A (Vegetation Red Edge, 20m)
        10: B9 (Water vapour, 60m)
        11: B11 (SWIR, 20m)
        12: B12 (SWIR, 20m)
        13: AOT (Aerosol Optical Thickness)
        14: SCL (Scene Classification Layer)

    Label 클래스:
        0: Clear
        1: Thick cloud
        2: Thin cloud
        3: Cloud shadow
    """

    # 정규화를 위한 최대값 (12-bit sensor → 4095, but typically use 10000)
    MAX_VALUE = 10000.0

    # 클래스 이름
    CLASS_NAMES = ['clear', 'thick_cloud', 'thin_cloud', 'cloud_shadow']
    NUM_CLASSES = 4

    # 레벨별 밴드 수
    BANDS_PER_LEVEL = {
        'l1c': 13,
        'l2a': 14,
    }

    def __init__(
        self,
        taco_dir: Optional[Union[str, Path]] = None,
        split: str = 'train',
        bands: Optional[List[int]] = None,
        level: Literal['l1c', 'l2a'] = 'l1c',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        normalize: bool = True,
        return_metadata: bool = False,
        patch_size: Optional[int] = 512,
        random_crop: bool = True,
        use_cache: bool = True,
        preload: bool = False,
    ):
        """
        Args:
            taco_dir: TACO 파일이 저장된 디렉토리 경로. None이면 level에 따라 기본 경로 사용
            split: 데이터 분할 ('train', 'validation', 'test', 'all')
            bands: 사용할 밴드 인덱스 리스트 (1-indexed). None이면 모든 밴드 사용
            level: 데이터 레벨 ('l1c' 또는 'l2a')
            transform: 이미지에 적용할 변환 함수
            target_transform: 레이블에 적용할 변환 함수
            normalize: 이미지 정규화 여부 (True: 0-1 범위로 정규화)
            return_metadata: True면 메타데이터도 함께 반환
            patch_size: 출력 패치 크기. None이면 원본 크기 유지 (배치 사용 시 주의)
            random_crop: True면 랜덤 crop, False면 center crop (validation/test용)
            use_cache: True면 파일 핸들 캐싱 사용 (권장)
            preload: True면 모든 데이터를 메모리에 미리 로드 (빠른 학습, 메모리 사용 증가)
        """
        self.level = level.lower()
        if self.level not in ['l1c', 'l2a']:
            raise ValueError(f"level must be 'l1c' or 'l2a', got '{level}'")

        # taco_dir이 None이면 기본 경로 사용
        if taco_dir is None:
            taco_dir = DEFAULT_TACO_DIRS[self.level]

        self.taco_dir = Path(taco_dir)
        self.split = split
        self.bands = bands  # 1-indexed
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        self.return_metadata = return_metadata
        self.patch_size = patch_size
        # validation/test는 center crop, train은 random crop
        self.random_crop = random_crop if split == 'train' else False
        self.use_cache = use_cache
        self.preload = preload
        self._preloaded_data = None

        # TACO 파일 로드
        taco_files = sorted(glob.glob(str(self.taco_dir / '*.taco')))
        if not taco_files:
            raise FileNotFoundError(f"No TACO files found in {self.taco_dir}")

        print(
            f"Loading {len(taco_files)} TACO files from {self.taco_dir} ({self.level.upper()})...")
        self.full_dataset = tacoreader.load(taco_files)

        # 데이터 분할 필터링
        if split == 'all':
            self.dataset = self.full_dataset
        else:
            valid_splits = ['train', 'validation', 'test']
            if split not in valid_splits:
                raise ValueError(
                    f"split must be one of {valid_splits} or 'all', got '{split}'")

            mask = self.full_dataset['tortilla:data_split'] == split
            self.dataset = self.full_dataset[mask].reset_index(drop=True)

        # TortillaDataFrame을 다시 wrap
        if hasattr(self.full_dataset, '_storage_options'):
            self.dataset = tacoreader.v1.TortillaDataFrame.TortillaDataFrame(
                self.dataset,
                storage_options=self.full_dataset._storage_options
            )

        # 파일 경로 캐싱 (tacoreader 호출 오버헤드 제거)
        print(f"Caching file paths for {len(self.dataset)} samples...")
        self._cached_paths = self._cache_all_paths()
        
        print(f"Dataset loaded: {len(self.dataset)} samples ({split})")
        
        # 데이터 프리로드
        if preload:
            self._preload_all_data()
    
    def _cache_all_paths(self) -> List[Tuple[str, str]]:
        """모든 샘플의 이미지/레이블 경로를 캐싱합니다."""
        cached_paths = []
        for idx in range(len(self.dataset)):
            sample = self.dataset.read(idx)
            image_path = sample.read(0)
            label_path = sample.read(1)
            cached_paths.append((image_path, label_path))
        return cached_paths
    
    def _load_single_sample(self, idx: int) -> dict:
        """단일 샘플을 로드합니다 (병렬 로딩용)."""
        image_path, label_path = self._cached_paths[idx]
        sample_row = self.dataset.iloc[idx]
        
        with rio.open(image_path) as src:
            if self.bands is not None:
                image = src.read(self.bands)
            else:
                image = src.read()
        
        with rio.open(label_path) as src:
            label = src.read(1)
        
        image = image.astype(np.float32)
        label = label.astype(np.int64)
        
        # 유효하지 않은 라벨 값을 ignore_index (255)로 매핑
        invalid_mask = (label < 0) | (label > 3)
        label[invalid_mask] = 255
        
        return {
            'image': image,
            'label': label,
            'metadata': {
                'roi_id': sample_row.get('roi_id', None),
                's2_id': sample_row.get('s2_id', None),
                'data_split': sample_row.get('tortilla:data_split', None),
            }
        }
    
    def _preload_all_data(self):
        """모든 데이터를 메모리에 병렬로 미리 로드합니다."""
        print(f"Preloading {len(self.dataset)} samples to memory (parallel)...")
        
        self._preloaded_data = [None] * len(self.dataset)
        num_threads = min(16, len(self.dataset))
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(self._load_single_sample, idx): idx
                for idx in range(len(self.dataset))
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preloading"):
                idx = futures[future]
                self._preloaded_data[idx] = future.result()
        
        # 메모리 사용량 계산
        sample = self._preloaded_data[0]
        bytes_per_sample = sample['image'].nbytes + sample['label'].nbytes
        total_gb = len(self._preloaded_data) * bytes_per_sample / 1e9
        print(f"Preloading complete. Memory usage: ~{total_gb:.2f} GB")

    def __len__(self) -> int:
        return len(self.dataset)

    def _crop_to_patch_size(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지와 레이블을 패치 크기에 맞게 crop합니다.

        Args:
            image: (C, H, W) 형태의 이미지
            label: (H, W) 형태의 레이블

        Returns:
            cropped image, cropped label
        """
        _, h, w = image.shape
        patch_size = self.patch_size

        # 패치 크기보다 작으면 패딩
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            # 이미지 패딩 (C, H, W) -> 0으로 패딩
            image = np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
            # 레이블 패딩 -> 255 (ignore_index)로 패딩
            label = np.pad(
                label,
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=255
            )
            h, w = h + pad_h, w + pad_w

        # Crop 위치 결정
        if self.random_crop:
            top = np.random.randint(0, h - patch_size + 1)
            left = np.random.randint(0, w - patch_size + 1)
        else:
            # Center crop
            top = (h - patch_size) // 2
            left = (w - patch_size) // 2

        # Crop 적용 (contiguous array를 위해 copy 사용)
        image = image[:, top:top + patch_size, left:left + patch_size].copy()
        label = label[top:top + patch_size, left:left + patch_size].copy()

        return image, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        데이터셋에서 샘플을 가져옵니다.

        Args:
            idx: 샘플 인덱스

        Returns:
            image: (C, H, W) 형태의 이미지 텐서
            label: (H, W) 형태의 레이블 텐서
            metadata: (optional) 메타데이터 딕셔너리
        """
        # 프리로드된 데이터 사용
        if self._preloaded_data is not None:
            data = self._preloaded_data[idx]
            image = data['image'].copy()
            label = data['label'].copy()
            metadata_base = data['metadata']
        else:
            # 캐싱된 경로 사용 (tacoreader 호출 오버헤드 제거)
            image_path, label_path = self._cached_paths[idx]
            sample_row = self.dataset.iloc[idx]
            
            # 파일 핸들 캐싱 사용
            if self.use_cache:
                cache = _get_file_cache()
                img_src = cache.get(image_path)
                label_src = cache.get(label_path)
                
                if self.bands is not None:
                    image = img_src.read(self.bands)
                else:
                    image = img_src.read()
                label = label_src.read(1)
            else:
                # 이미지 읽기
                with rio.open(image_path) as src:
                    if self.bands is not None:
                        image = src.read(self.bands)  # bands는 1-indexed
                    else:
                        image = src.read()  # 모든 밴드 읽기

                # 레이블 읽기
                with rio.open(label_path) as src:
                    label = src.read(1)  # 단일 채널

            # 데이터 타입 변환
            image = image.astype(np.float32)
            label = label.astype(np.int64)
            
            # 유효하지 않은 라벨 값을 ignore_index (255)로 매핑
            # 유효한 값: 0 (clear), 1 (thick cloud), 2 (thin cloud), 3 (cloud shadow)
            # 유효하지 않은 값: 4, 5, 6, 99 등 -> 255
            invalid_mask = (label < 0) | (label > 3)
            label[invalid_mask] = 255
            
            metadata_base = {
                'roi_id': sample_row.get('roi_id', None),
                's2_id': sample_row.get('s2_id', None),
                'data_split': sample_row.get('tortilla:data_split', None),
            }

        # 패치 크기에 맞게 crop
        if self.patch_size is not None:
            image, label = self._crop_to_patch_size(image, label)

        # 정규화
        if self.normalize:
            image = np.clip(image / self.MAX_VALUE, 0, 1)

        # Transform 적용
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        # 텐서 변환
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)

        if self.return_metadata:
            sample_row = self.dataset.iloc[idx]
            metadata = {
                **metadata_base,
                'thick_percentage': sample_row.get('thick_percentage', None),
                'thin_percentage': sample_row.get('thin_percentage', None),
                'cloud_shadow_percentage': sample_row.get('cloud_shadow_percentage', None),
                'clear_percentage': sample_row.get('clear_percentage', None),
                'level': self.level,
            }
            return image, label, metadata

        return image, label

    def get_split_sizes(self) -> dict:
        """각 분할의 샘플 수를 반환합니다."""
        splits = self.full_dataset['tortilla:data_split'].value_counts(
        ).to_dict()
        return splits

    def get_num_bands(self) -> int:
        """현재 레벨의 총 밴드 수를 반환합니다."""
        return self.BANDS_PER_LEVEL[self.level]

    @staticmethod
    def get_rgb_bands() -> List[int]:
        """RGB 밴드 인덱스 반환 (1-indexed: B4, B3, B2) - L1C/L2A 공통"""
        return [4, 3, 2]

    @staticmethod
    def get_all_10m_bands() -> List[int]:
        """10m 해상도 밴드 인덱스 반환 (1-indexed: B2, B3, B4, B8) - L1C/L2A 공통"""
        return [2, 3, 4, 8]

    @staticmethod
    def get_swir_bands_l1c() -> List[int]:
        """L1C SWIR 밴드 인덱스 반환 (1-indexed: B11, B12)"""
        return [12, 13]

    @staticmethod
    def get_swir_bands_l2a() -> List[int]:
        """L2A SWIR 밴드 인덱스 반환 (1-indexed: B11, B12)"""
        return [11, 12]

    @staticmethod
    def get_scl_band_l2a() -> int:
        """L2A Scene Classification Layer 밴드 인덱스 반환 (1-indexed)"""
        return 14


def create_dataloaders(
    taco_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    bands: Optional[List[int]] = None,
    level: Literal['l1c', 'l2a'] = 'l1c',
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    학습, 검증, 테스트 데이터로더를 생성합니다.

    Args:
        taco_dir: TACO 파일이 저장된 디렉토리 경로. None이면 level에 따라 기본 경로 사용
        batch_size: 배치 크기
        bands: 사용할 밴드 인덱스 리스트 (1-indexed)
        level: 데이터 레벨 ('l1c' 또는 'l2a')
        num_workers: 데이터 로딩에 사용할 워커 수
        pin_memory: GPU 학습 시 True 권장
        normalize: 이미지 정규화 여부

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    # 데이터셋 생성
    train_dataset = CloudSEN12Dataset(
        taco_dir=taco_dir,
        split='train',
        bands=bands,
        level=level,
        normalize=normalize,
    )

    val_dataset = CloudSEN12Dataset(
        taco_dir=taco_dir,
        split='validation',
        bands=bands,
        level=level,
        normalize=normalize,
    )

    test_dataset = CloudSEN12Dataset(
        taco_dir=taco_dir,
        split='test',
        bands=bands,
        level=level,
        normalize=normalize,
    )

    # 데이터로더 생성
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("CloudSEN12 Dataset Test")
    print("=" * 60)

    # L1C 데이터셋 테스트
    print("\n[1] Testing L1C Dataset")
    print("-" * 40)

    dataset_l1c = CloudSEN12Dataset(
        level='l1c',
        split='train',
        bands=CloudSEN12Dataset.get_rgb_bands(),
        normalize=True,
        return_metadata=True,
    )

    print(f"L1C Dataset size: {len(dataset_l1c)}")
    print(f"L1C Total bands: {dataset_l1c.get_num_bands()}")
    print(f"Split sizes: {dataset_l1c.get_split_sizes()}")

    image_l1c, label_l1c, metadata_l1c = dataset_l1c[0]
    print(f"\nL1C Sample shape:")
    print(f"  Image: {image_l1c.shape}, dtype: {image_l1c.dtype}")
    print(f"  Label: {label_l1c.shape}, dtype: {label_l1c.dtype}")
    print(f"  Label classes: {torch.unique(label_l1c).tolist()}")

    # L2A 데이터셋 테스트
    print("\n[2] Testing L2A Dataset")
    print("-" * 40)

    dataset_l2a = CloudSEN12Dataset(
        level='l2a',
        split='train',
        bands=CloudSEN12Dataset.get_rgb_bands(),
        normalize=True,
        return_metadata=True,
    )

    print(f"L2A Dataset size: {len(dataset_l2a)}")
    print(f"L2A Total bands: {dataset_l2a.get_num_bands()}")
    print(f"Split sizes: {dataset_l2a.get_split_sizes()}")

    image_l2a, label_l2a, metadata_l2a = dataset_l2a[0]
    print(f"\nL2A Sample shape:")
    print(f"  Image: {image_l2a.shape}, dtype: {image_l2a.dtype}")
    print(f"  Label: {label_l2a.shape}, dtype: {label_l2a.dtype}")
    print(f"  Label classes: {torch.unique(label_l2a).tolist()}")

    # 시각화
    print("\n[3] Generating Visualization")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 레이블 색상 맵
    label_colors = {
        0: [0.2, 0.8, 0.2],    # Clear - 녹색
        1: [1.0, 1.0, 1.0],    # Thick cloud - 흰색
        2: [0.7, 0.7, 0.7],    # Thin cloud - 회색
        3: [0.3, 0.3, 0.3],    # Cloud shadow - 어두운 회색
    }

    # L1C RGB
    rgb_l1c = image_l1c.numpy().transpose(1, 2, 0)
    rgb_l1c = np.clip(rgb_l1c * 3, 0, 1)
    axes[0, 0].imshow(rgb_l1c)
    axes[0, 0].set_title('L1C - Sentinel-2 RGB')
    axes[0, 0].axis('off')

    # L1C Label
    label_l1c_np = label_l1c.numpy()
    label_l1c_rgb = np.zeros((*label_l1c_np.shape, 3))
    for class_id, color in label_colors.items():
        label_l1c_rgb[label_l1c_np == class_id] = color
    axes[0, 1].imshow(label_l1c_rgb)
    axes[0, 1].set_title('L1C - Cloud Mask')
    axes[0, 1].axis('off')

    # L2A RGB
    rgb_l2a = image_l2a.numpy().transpose(1, 2, 0)
    rgb_l2a = np.clip(rgb_l2a * 3, 0, 1)
    axes[1, 0].imshow(rgb_l2a)
    axes[1, 0].set_title('L2A - Sentinel-2 RGB')
    axes[1, 0].axis('off')

    # L2A Label
    label_l2a_np = label_l2a.numpy()
    label_l2a_rgb = np.zeros((*label_l2a_np.shape, 3))
    for class_id, color in label_colors.items():
        label_l2a_rgb[label_l2a_np == class_id] = color
    axes[1, 1].imshow(label_l2a_rgb)
    axes[1, 1].set_title('L2A - Cloud Mask')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(
        '/home/junghwan/cloud_seg/sample_visualization_l1c_l2a.png', dpi=150)
    plt.show()
    print("\nVisualization saved to 'sample_visualization_l1c_l2a.png'")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
