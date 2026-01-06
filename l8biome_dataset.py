"""
L8 Biome Dataset for Cloud Segmentation

Landsat 8 Cloud Cover Assessment Validation Data를 기반으로 한 PyTorch Dataset 클래스
대용량 이미지(~9000x9000)를 효율적으로 처리하기 위한 패치 기반 로딩 지원

사용 예시:
    from l8biome_dataset import L8BiomeDataset, create_dataloaders
    
    # 전체 biome 데이터셋 생성 (패치 기반)
    train_dataset = L8BiomeDataset(
        data_dir='/home/telepix_nas/junghwan/cloud_seg/l8biome_extracted/l8biome',
        split='train',
        patch_size=512,
        bands=[4, 3, 2],  # RGB (B4, B3, B2)
    )
    
    # 특정 biome만 사용
    snow_dataset = L8BiomeDataset(
        data_dir='/home/telepix_nas/junghwan/cloud_seg/l8biome_extracted/l8biome',
        biomes=['snow_ice'],
        split='train',
    )
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir='/home/telepix_nas/junghwan/cloud_seg/l8biome_extracted/l8biome',
        batch_size=8,
        patch_size=512,
    )
"""

import random
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Union, Literal

import numpy as np
import rasterio as rio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
# 파일 핸들 캐싱을 위한 유틸리티
# =============================================================================

class RasterioFileCache:
    """
    Rasterio 파일 핸들을 캐싱하여 반복적인 파일 열기/닫기 오버헤드를 줄입니다.
    """

    def __init__(self, max_size: int = 64):
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
_file_cache = None


def _get_file_cache() -> RasterioFileCache:
    """전역 파일 캐시를 반환합니다."""
    global _file_cache
    if _file_cache is None:
        _file_cache = RasterioFileCache(max_size=128)
    return _file_cache


# 기본 경로
DEFAULT_DATA_DIR = '/home/telepix_nas/junghwan/cloud_seg/l8biome_extracted/l8biome'

# Biome 종류
BIOMES = ['barren', 'forest', 'grass_crops', 'shrubland',
          'snow_ice', 'urban', 'water', 'wetlands']


class L8BiomeDataset(Dataset):
    """
    L8 Biome 데이터셋을 위한 PyTorch Dataset 클래스

    대용량 Landsat 8 이미지를 패치 단위로 효율적으로 로딩합니다.

    Attributes:
        data_dir: 데이터가 저장된 디렉토리 경로
        biomes: 사용할 biome 리스트 (None이면 전체 사용)
        split: 데이터 분할 ('train', 'val', 'test', 'all')
        patch_size: 패치 크기 (None이면 전체 이미지 사용)
        stride: 패치 추출 시 stride (None이면 patch_size와 동일)
        bands: 사용할 밴드 인덱스 리스트 (1-indexed)
        normalize: 이미지 정규화 여부
        transform: 이미지와 마스크에 적용할 변환 함수

    Landsat 8 OLI/TIRS 밴드 (1-indexed, 총 11개):
        1: B1 - Coastal Aerosol (0.43–0.45 μm, 30m)
        2: B2 - Blue (0.45–0.51 μm, 30m)
        3: B3 - Green (0.53–0.59 μm, 30m)
        4: B4 - Red (0.64–0.67 μm, 30m)
        5: B5 - NIR (0.85–0.88 μm, 30m)
        6: B6 - SWIR 1 (1.57–1.65 μm, 30m)
        7: B7 - SWIR 2 (2.11–2.29 μm, 30m)
        8: B8 - Pan (0.50–0.68 μm, 15m) - 리샘플된 30m
        9: B9 - Cirrus (1.36–1.38 μm, 30m)
        10: B10 - TIRS 1 (10.60–11.19 μm, 100m)
        11: B11 - TIRS 2 (11.50–12.51 μm, 100m)

    Label 클래스:
        0: Clear (맑음)
        1: Thin Cloud (얇은 구름)  - 원본 64
        2: Cloud (구름)             - 원본 128
        3: Cloud Shadow (구름 그림자) - 원본 192
        255: Fill/No Data (채우기/무효) - 무시됨
    """

    # 원본 마스크 값 → 클래스 인덱스 매핑
    MASK_MAPPING = {
        0: 0,     # Clear
        64: 1,    # Thin Cloud
        128: 2,   # Cloud
        192: 3,   # Cloud Shadow
        255: 255,  # Fill/No Data (ignore)
    }

    # 클래스 이름
    CLASS_NAMES = ['clear', 'thin_cloud', 'cloud', 'cloud_shadow']
    NUM_CLASSES = 4  # Fill 제외
    IGNORE_INDEX = 255

    # 밴드 수
    NUM_BANDS = 11

    # 정규화를 위한 최대값 (uint8 → 255)
    MAX_VALUE = 255.0

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        biomes: Optional[List[str]] = None,
        split: Literal['train', 'val', 'test', 'all'] = 'train',
        patch_size: Optional[int] = 512,
        stride: Optional[int] = None,
        bands: Optional[List[int]] = None,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        min_valid_ratio: float = 0.5,
        random_seed: int = 42,
        return_metadata: bool = False,
        use_cache: bool = True,
        preload: bool = False,
    ):
        """
        Args:
            data_dir: 데이터 디렉토리 경로. None이면 기본 경로 사용
            biomes: 사용할 biome 리스트. None이면 전체 사용
            split: 데이터 분할 ('train', 'val', 'test', 'all')
            patch_size: 패치 크기. None이면 전체 이미지 사용 (메모리 주의)
            stride: 패치 추출 stride. None이면 patch_size와 동일 (겹침 없음)
            bands: 사용할 밴드 인덱스 리스트 (1-indexed). None이면 모든 밴드 사용
            normalize: 이미지 정규화 여부 (True: 0-1 범위로 정규화)
            transform: 이미지와 마스크에 적용할 변환 함수 (albumentations 호환)
            min_valid_ratio: 최소 유효 픽셀 비율 (Fill 제외). 이보다 낮으면 패치 제외
            random_seed: 데이터 분할을 위한 시드
            return_metadata: True면 메타데이터도 함께 반환
            use_cache: True면 파일 핸들 캐싱 사용 (권장)
            preload: True면 모든 패치를 메모리에 미리 로드 (빠른 학습, 메모리 사용 증가)
        """
        self.data_dir = Path(data_dir) if data_dir else Path(DEFAULT_DATA_DIR)
        self.biomes = biomes if biomes else BIOMES
        self.split = split
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size
        self.bands = bands  # 1-indexed
        self.normalize = normalize
        self.transform = transform
        self.min_valid_ratio = min_valid_ratio
        self.random_seed = random_seed
        self.return_metadata = return_metadata
        self.use_cache = use_cache
        self.preload = preload
        self._preloaded_data = None

        # biome 유효성 검사
        for biome in self.biomes:
            if biome not in BIOMES:
                raise ValueError(
                    f"Invalid biome: {biome}. Valid biomes: {BIOMES}")

        # 장면 목록 수집
        self.scenes = self._collect_scenes()

        # 데이터 분할
        self.split_scenes = self._split_data()

        # 패치 인덱스 생성
        if self.patch_size:
            self.patches = self._generate_patches()
            print(f"L8 Biome Dataset loaded: {len(self.patches)} patches from "
                  f"{len(self.split_scenes)} scenes ({split})")
        else:
            self.patches = None
            print(
                f"L8 Biome Dataset loaded: {len(self.split_scenes)} scenes ({split})")

        # 데이터 프리로드
        if preload and self.patches:
            self._preload_all_data()

    def _collect_scenes(self) -> List[dict]:
        """각 biome에서 장면 정보를 수집합니다."""
        scenes = []

        for biome in self.biomes:
            biome_dir = self.data_dir / biome
            if not biome_dir.exists():
                print(f"Warning: Biome directory not found: {biome_dir}")
                continue

            for scene_dir in sorted(biome_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue

                scene_id = scene_dir.name
                image_path = scene_dir / f"{scene_id}.TIF"
                mask_path = scene_dir / f"{scene_id}_fixedmask.TIF"

                if image_path.exists() and mask_path.exists():
                    # 이미지 크기 정보 가져오기
                    with rio.open(image_path) as src:
                        height, width = src.height, src.width

                    scenes.append({
                        'scene_id': scene_id,
                        'biome': biome,
                        'image_path': str(image_path),
                        'mask_path': str(mask_path),
                        'height': height,
                        'width': width,
                    })

        return scenes

    def _split_data(self) -> List[dict]:
        """데이터를 train/val/test로 분할합니다."""
        if self.split == 'all':
            return self.scenes

        # 각 biome별로 분할 (stratified)
        random.seed(self.random_seed)

        train_scenes, val_scenes, test_scenes = [], [], []

        for biome in self.biomes:
            biome_scenes = [s for s in self.scenes if s['biome'] == biome]
            random.shuffle(biome_scenes)

            n = len(biome_scenes)
            # 8:1:1 비율 (12개 장면: 10 train, 1 val, 1 test)
            n_train = max(1, int(n * 0.8))
            n_val = max(1, int(n * 0.1))

            train_scenes.extend(biome_scenes[:n_train])
            val_scenes.extend(biome_scenes[n_train:n_train + n_val])
            test_scenes.extend(biome_scenes[n_train + n_val:])

        if self.split == 'train':
            return train_scenes
        elif self.split == 'val':
            return val_scenes
        elif self.split == 'test':
            return test_scenes
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def _generate_patches(self) -> List[dict]:
        """각 장면에서 패치 정보를 생성합니다."""
        patches = []

        for scene_idx, scene in enumerate(self.split_scenes):
            height, width = scene['height'], scene['width']

            # 패치 위치 계산
            for row in range(0, height - self.patch_size + 1, self.stride):
                for col in range(0, width - self.patch_size + 1, self.stride):
                    patches.append({
                        'scene_idx': scene_idx,
                        'row': row,
                        'col': col,
                    })

        return patches

    def _preload_all_data(self):
        """모든 패치를 메모리에 미리 로드합니다."""
        print(f"Preloading {len(self.patches)} patches to memory...")
        self._preloaded_data = []

        # 장면별로 이미지를 한 번만 열고 모든 패치 추출
        scene_patches = {}
        for idx, patch in enumerate(self.patches):
            scene_idx = patch['scene_idx']
            if scene_idx not in scene_patches:
                scene_patches[scene_idx] = []
            scene_patches[scene_idx].append((idx, patch))

        # 장면별로 처리
        for scene_idx in tqdm(sorted(scene_patches.keys()), desc="Preloading scenes"):
            scene = self.split_scenes[scene_idx]

            with rio.open(scene['image_path']) as img_src, \
                    rio.open(scene['mask_path']) as mask_src:

                for idx, patch in scene_patches[scene_idx]:
                    row, col = patch['row'], patch['col']
                    window = Window(col, row, self.patch_size, self.patch_size)

                    if self.bands:
                        image = img_src.read(self.bands, window=window)
                    else:
                        image = img_src.read(window=window)

                    mask = mask_src.read(1, window=window)
                    mask = self._convert_mask(mask)

                    # 유효 픽셀 비율 체크
                    valid_ratio = np.mean(mask != self.IGNORE_INDEX)

                    self._preloaded_data.append({
                        'image': image.astype(np.float32),
                        'mask': mask,
                        'valid_ratio': valid_ratio,
                        'scene_idx': scene_idx,
                        'row': row,
                        'col': col,
                    })

        # 메모리 사용량 계산
        sample = self._preloaded_data[0]
        bytes_per_sample = sample['image'].nbytes + sample['mask'].nbytes
        total_gb = len(self._preloaded_data) * bytes_per_sample / 1e9
        print(f"Preloading complete. Memory usage: ~{total_gb:.2f} GB")

    def __len__(self) -> int:
        if self.patches:
            return len(self.patches)
        return len(self.split_scenes)

    def _convert_mask(self, mask: np.ndarray) -> np.ndarray:
        """원본 마스크 값을 클래스 인덱스로 변환합니다."""
        converted = np.full_like(mask, self.IGNORE_INDEX, dtype=np.int64)

        for orig_val, class_idx in self.MASK_MAPPING.items():
            converted[mask == orig_val] = class_idx

        return converted

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
            mask = data['mask'].copy()
            valid_ratio = data['valid_ratio']
            scene = self.split_scenes[data['scene_idx']]
            row, col = data['row'], data['col']

            # 유효 픽셀 비율 체크
            if valid_ratio < self.min_valid_ratio:
                return self.__getitem__(random.randint(0, len(self) - 1))
        elif self.patches:
            patch_info = self.patches[idx]
            scene = self.split_scenes[patch_info['scene_idx']]
            row, col = patch_info['row'], patch_info['col']

            # Windowed reading
            window = Window(col, row, self.patch_size, self.patch_size)

            # 캐시 사용
            if self.use_cache:
                cache = _get_file_cache()
                img_src = cache.get(scene['image_path'])
                mask_src = cache.get(scene['mask_path'])

                if self.bands:
                    image = img_src.read(self.bands, window=window)
                else:
                    image = img_src.read(window=window)
                mask = mask_src.read(1, window=window)
            else:
                with rio.open(scene['image_path']) as src:
                    if self.bands:
                        image = src.read(self.bands, window=window)
                    else:
                        image = src.read(window=window)

                with rio.open(scene['mask_path']) as src:
                    mask = src.read(1, window=window)

            # 마스크 변환
            mask = self._convert_mask(mask)

            # 유효 픽셀 비율 체크
            valid_ratio = np.mean(mask != self.IGNORE_INDEX)
            if valid_ratio < self.min_valid_ratio:
                # 유효하지 않은 패치는 랜덤 샘플링으로 대체
                return self.__getitem__(random.randint(0, len(self) - 1))

            # 데이터 타입 변환
            image = image.astype(np.float32)
        else:
            # 전체 이미지 로딩 (메모리 주의)
            scene = self.split_scenes[idx]

            with rio.open(scene['image_path']) as src:
                if self.bands:
                    image = src.read(self.bands)
                else:
                    image = src.read()

            with rio.open(scene['mask_path']) as src:
                mask = src.read(1)

            row, col = 0, 0

            # 마스크 변환
            mask = self._convert_mask(mask)

            # 데이터 타입 변환
            image = image.astype(np.float32)

        # 정규화
        if self.normalize:
            image = image / self.MAX_VALUE

        # Transform 적용 (albumentations 스타일)
        if self.transform is not None:
            # (C, H, W) -> (H, W, C) for albumentations
            image_hwc = image.transpose(1, 2, 0)
            transformed = self.transform(image=image_hwc, mask=mask)
            image = transformed['image'].transpose(
                2, 0, 1)  # Back to (C, H, W)
            mask = transformed['mask']

        # 텐서 변환
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long()

        if self.return_metadata:
            metadata = {
                'scene_id': scene['scene_id'],
                'biome': scene['biome'],
                'row': row,
                'col': col,
            }
            return image, mask, metadata

        return image, mask

    def get_scene_info(self) -> List[dict]:
        """현재 분할의 장면 정보를 반환합니다."""
        return self.split_scenes

    def get_split_sizes(self) -> dict:
        """각 분할의 장면 수를 반환합니다."""
        sizes = {}
        for split in ['train', 'val', 'test']:
            temp = L8BiomeDataset.__new__(L8BiomeDataset)
            temp.data_dir = self.data_dir
            temp.biomes = self.biomes
            temp.random_seed = self.random_seed
            temp.scenes = self.scenes
            temp.split = split
            temp.split_scenes = temp._split_data()
            sizes[split] = len(temp.split_scenes)
        return sizes

    @staticmethod
    def get_rgb_bands() -> List[int]:
        """RGB 밴드 인덱스 반환 (1-indexed: B4, B3, B2 = Red, Green, Blue)"""
        return [4, 3, 2]

    @staticmethod
    def get_true_color_bands() -> List[int]:
        """True Color 밴드 인덱스 반환 (B4, B3, B2)"""
        return [4, 3, 2]

    @staticmethod
    def get_false_color_bands() -> List[int]:
        """False Color (NIR) 밴드 인덱스 반환 (B5, B4, B3)"""
        return [5, 4, 3]

    @staticmethod
    def get_swir_bands() -> List[int]:
        """SWIR 밴드 인덱스 반환 (B6, B7)"""
        return [6, 7]

    @staticmethod
    def get_thermal_bands() -> List[int]:
        """Thermal 밴드 인덱스 반환 (B10, B11)"""
        return [10, 11]

    @staticmethod
    def get_cloud_detection_bands() -> List[int]:
        """Cloud detection에 유용한 밴드 조합 반환 (Blue, NIR, SWIR1, Cirrus)"""
        return [2, 5, 6, 9]


def create_dataloaders(
    data_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    patch_size: int = 512,
    stride: Optional[int] = None,
    biomes: Optional[List[str]] = None,
    bands: Optional[List[int]] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    학습, 검증, 테스트 데이터로더를 생성합니다.

    Args:
        data_dir: 데이터 디렉토리 경로
        batch_size: 배치 크기
        patch_size: 패치 크기
        stride: 패치 stride (None이면 patch_size와 동일)
        biomes: 사용할 biome 리스트
        bands: 사용할 밴드 인덱스 리스트 (1-indexed)
        num_workers: 데이터 로딩 워커 수
        pin_memory: GPU 학습 시 True 권장
        normalize: 이미지 정규화 여부
        train_transform: 학습 데이터 변환
        val_transform: 검증/테스트 데이터 변환

    Returns:
        train_loader, val_loader, test_loader
    """
    # 데이터셋 생성
    train_dataset = L8BiomeDataset(
        data_dir=data_dir,
        biomes=biomes,
        split='train',
        patch_size=patch_size,
        stride=stride,
        bands=bands,
        normalize=normalize,
        transform=train_transform,
    )

    val_dataset = L8BiomeDataset(
        data_dir=data_dir,
        biomes=biomes,
        split='val',
        patch_size=patch_size,
        stride=patch_size,  # 검증 시 겹침 없음
        bands=bands,
        normalize=normalize,
        transform=val_transform,
    )

    test_dataset = L8BiomeDataset(
        data_dir=data_dir,
        biomes=biomes,
        split='test',
        patch_size=patch_size,
        stride=patch_size,  # 테스트 시 겹침 없음
        bands=bands,
        normalize=normalize,
        transform=val_transform,
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


def get_class_weights(data_dir: Optional[Union[str, Path]] = None,
                      biomes: Optional[List[str]] = None) -> torch.Tensor:
    """
    클래스별 가중치를 계산합니다 (클래스 불균형 해결용).

    Returns:
        class_weights: 클래스별 가중치 텐서
    """
    data_dir = Path(data_dir) if data_dir else Path(DEFAULT_DATA_DIR)
    biomes = biomes if biomes else BIOMES

    class_counts = np.zeros(L8BiomeDataset.NUM_CLASSES, dtype=np.float64)

    for biome in biomes:
        biome_dir = data_dir / biome
        if not biome_dir.exists():
            continue

        for scene_dir in biome_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            mask_path = scene_dir / f"{scene_dir.name}_fixedmask.TIF"
            if not mask_path.exists():
                continue

            with rio.open(mask_path) as src:
                mask = src.read(1)

            # 클래스별 카운트
            for orig_val, class_idx in L8BiomeDataset.MASK_MAPPING.items():
                if class_idx != L8BiomeDataset.IGNORE_INDEX:
                    class_counts[class_idx] += np.sum(mask == orig_val)

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (L8BiomeDataset.NUM_CLASSES * class_counts + 1e-10)
    weights = weights / weights.sum() * L8BiomeDataset.NUM_CLASSES  # 정규화

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("L8 Biome Dataset Test")
    print("=" * 60)

    # 데이터셋 테스트
    print("\n[1] Testing Dataset with Patches")
    print("-" * 40)

    dataset = L8BiomeDataset(
        data_dir=DEFAULT_DATA_DIR,
        split='train',
        patch_size=512,
        bands=L8BiomeDataset.get_rgb_bands(),
        normalize=True,
        return_metadata=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Split sizes: {dataset.get_split_sizes()}")
    print(f"Scenes in current split: {len(dataset.split_scenes)}")

    # 샘플 가져오기
    image, label, metadata = dataset[0]
    print(f"\nSample shape:")
    print(f"  Image: {image.shape}, dtype: {image.dtype}")
    print(f"  Label: {label.shape}, dtype: {label.dtype}")
    print(f"  Metadata: {metadata}")

    # 클래스 분포
    unique, counts = torch.unique(label, return_counts=True)
    print(f"\nClass distribution in sample:")
    for u, c in zip(unique.tolist(), counts.tolist()):
        if u != 255:
            pct = c / label.numel() * 100
            class_name = dataset.CLASS_NAMES[u] if u < len(
                dataset.CLASS_NAMES) else 'unknown'
            print(f"  {u} ({class_name}): {c} pixels ({pct:.2f}%)")

    # 클래스 가중치 계산
    print("\n[2] Computing Class Weights")
    print("-" * 40)
    weights = get_class_weights()
    print(f"Class weights: {weights.tolist()}")
    for i, (name, w) in enumerate(zip(dataset.CLASS_NAMES, weights.tolist())):
        print(f"  {name}: {w:.4f}")

    # 시각화
    print("\n[3] Generating Visualization")
    print("-" * 40)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 레이블 색상 맵
    label_colors = {
        0: [0.2, 0.8, 0.2],    # Clear - 녹색
        1: [0.7, 0.7, 0.7],    # Thin cloud - 회색
        2: [1.0, 1.0, 1.0],    # Cloud - 흰색
        3: [0.3, 0.3, 0.3],    # Cloud shadow - 어두운 회색
    }

    # 여러 샘플 시각화
    sample_indices = [i * (len(dataset) // 4) for i in range(4)]

    for i, idx in enumerate(sample_indices):
        image, label, metadata = dataset[idx]

        # RGB 이미지
        rgb = image.numpy().transpose(1, 2, 0)
        rgb = np.clip(rgb * 2.5, 0, 1)  # 밝기 조정

        axes[0, i].imshow(rgb)
        axes[0, i].set_title(
            f"{metadata['biome']}\n{metadata['scene_id'][:15]}...")
        axes[0, i].axis('off')

        # 마스크
        label_np = label.numpy()
        label_rgb = np.zeros((*label_np.shape, 3))
        for class_id, color in label_colors.items():
            label_rgb[label_np == class_id] = color
        # Fill 영역은 빨간색으로 표시
        label_rgb[label_np == 255] = [1.0, 0.0, 0.0]

        axes[1, i].imshow(label_rgb)
        axes[1, i].set_title(f'Cloud Mask')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('/home/junghwan/cloud_seg/l8biome_visualization.png', dpi=150)
    plt.show()
    print("\nVisualization saved to 'l8biome_visualization.png'")

    # DataLoader 테스트
    print("\n[4] Testing DataLoader")
    print("-" * 40)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=DEFAULT_DATA_DIR,
        batch_size=4,
        patch_size=256,
        bands=L8BiomeDataset.get_rgb_bands(),
        num_workers=0,  # 테스트용
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 배치 테스트
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
