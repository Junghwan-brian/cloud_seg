"""
38-Cloud / 95-Cloud Dataset for Cloud Segmentation

Landsat 8 Cloud Segmentation Dataset (38-Cloud 및 95-Cloud)를 위한 PyTorch Dataset 클래스
4채널(RGB+NIR) 384x384 패치 기반 클라우드 세그멘테이션 데이터셋

데이터셋 구조:
- 38-Cloud: 18개 Landsat 8 장면에서 추출한 8,400개 학습 패치 + 20개 장면 테스트
- 95-Cloud: 38-Cloud에 추가로 57개 장면 (총 95개 장면)

사용 예시:
    from cloud38_95_dataset import Cloud38Dataset, Cloud95Dataset, create_dataloaders
    
    # 38-Cloud 데이터셋
    train_dataset = Cloud38Dataset(
        root_38cloud='/home/telepix_nas/junghwan/cloud_seg/38-cloud',
        split='train',
        use_nonempty=True,  # 비어있지 않은 패치만 사용
    )
    
    # 95-Cloud 데이터셋 (38-Cloud + 추가 데이터)
    train_dataset = Cloud95Dataset(
        root_38cloud='/home/telepix_nas/junghwan/cloud_seg/38-cloud',
        root_95cloud='/home/telepix_nas/junghwan/cloud_seg/95-cloud',
        split='train',
        use_nonempty=True,
    )
    
    # 데이터 로더 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        root_38cloud='/home/telepix_nas/junghwan/cloud_seg/38-cloud',
        root_95cloud='/home/telepix_nas/junghwan/cloud_seg/95-cloud',
        batch_size=16,
        use_95cloud=True,
    )
"""

import random
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Union, Literal

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm


# =============================================================================
# 이미지 캐싱을 위한 유틸리티 함수들
# =============================================================================

@lru_cache(maxsize=2048)
def _cached_load_image(path: str) -> np.ndarray:
    """
    LRU 캐시를 사용하여 이미지를 로드합니다.
    캐시 크기: 2048개 (약 4GB 메모리 사용, 384x384 float32 기준)
    """
    return np.array(Image.open(path), dtype=np.float32)


@lru_cache(maxsize=2048)
def _cached_load_mask(path: str) -> np.ndarray:
    """
    LRU 캐시를 사용하여 마스크를 로드합니다.
    """
    return np.array(Image.open(path), dtype=np.int64)


# 기본 경로
DEFAULT_38CLOUD_DIR = '/home/telepix_nas/junghwan/cloud_seg/38-cloud'
DEFAULT_95CLOUD_DIR = '/home/telepix_nas/junghwan/cloud_seg/95-cloud'


class Cloud38Dataset(Dataset):
    """
    38-Cloud 데이터셋을 위한 PyTorch Dataset 클래스

    Landsat 8에서 추출한 384x384 크기의 패치 이미지와 클라우드 마스크를 제공합니다.

    Attributes:
        root: 38-Cloud 데이터 루트 디렉토리
        split: 데이터 분할 ('train', 'val', 'test')
        use_nonempty: True면 비어있지 않은 패치만 사용
        bands: 사용할 밴드 리스트 ('red', 'green', 'blue', 'nir')
        normalize: 이미지 정규화 여부
        transform: 이미지와 마스크에 적용할 변환 함수

    밴드 정보:
        - Red: Landsat 8 Band 4 (0.64–0.67 μm)
        - Green: Landsat 8 Band 3 (0.53–0.59 μm)
        - Blue: Landsat 8 Band 2 (0.45–0.51 μm)
        - NIR: Landsat 8 Band 5 (0.85–0.88 μm)

    Label 클래스:
        0: Clear (맑음)
        1: Cloud (구름) - 원본에서 255
    """

    # 사용 가능한 밴드
    AVAILABLE_BANDS = ['red', 'green', 'blue', 'nir']

    # 이미지 크기
    PATCH_SIZE = 384

    # 클래스 정보
    CLASS_NAMES = ['clear', 'cloud']
    NUM_CLASSES = 2

    # 정규화를 위한 최대값 (uint16 → 65535)
    MAX_VALUE = 65535.0

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: Literal['train', 'val', 'test', 'all'] = 'train',
        use_nonempty: bool = True,
        bands: Optional[List[str]] = None,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        val_ratio: float = 0.1,
        random_seed: int = 42,
        return_metadata: bool = False,
        preload: bool = False,
    ):
        """
        Args:
            root: 38-Cloud 데이터 루트 디렉토리
            split: 데이터 분할 ('train', 'val', 'test', 'all')
            use_nonempty: 비어있지 않은 패치만 사용할지 여부
            bands: 사용할 밴드 리스트. None이면 모든 밴드 사용
            normalize: 이미지 정규화 여부 (True: 0-1 범위로 정규화)
            transform: 이미지와 마스크에 적용할 변환 함수 (albumentations 호환)
            val_ratio: 학습 데이터에서 검증 데이터로 분리할 비율
            random_seed: 데이터 분할을 위한 시드
            return_metadata: True면 메타데이터도 함께 반환
            preload: True면 모든 데이터를 메모리에 미리 로드 (빠른 학습, 메모리 사용 증가)
        """
        self.root = Path(root) if root else Path(DEFAULT_38CLOUD_DIR)
        self.split = split
        self.use_nonempty = use_nonempty
        self.bands = bands if bands else self.AVAILABLE_BANDS
        self.normalize = normalize
        self.transform = transform
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.return_metadata = return_metadata
        self.preload = preload
        self._preloaded_data = None

        # 밴드 유효성 검사
        for band in self.bands:
            if band not in self.AVAILABLE_BANDS:
                raise ValueError(
                    f"Invalid band: {band}. Valid bands: {self.AVAILABLE_BANDS}")

        # 경로 설정
        if split == 'test':
            self.data_dir = self.root / '38-Cloud_test'
            self.has_gt = False
        else:
            self.data_dir = self.root / '38-Cloud_training'
            self.has_gt = True

        # 패치 목록 로드
        self.patches = self._load_patches()

        # Train/Val 분할
        if split in ['train', 'val']:
            self.patches = self._split_train_val()

        print(f"Cloud38 Dataset loaded: {len(self.patches)} patches ({split})")

        # 데이터 프리로드
        if preload:
            self._preload_all_data()

    def _load_patches(self) -> List[str]:
        """패치 목록을 로드합니다."""
        if self.split == 'test':
            csv_path = self.data_dir / 'test_patches_38-Cloud.csv'
        else:
            if self.use_nonempty:
                csv_path = self.root / 'training_patches_38-cloud_nonempty.csv'
            else:
                csv_path = self.data_dir / 'training_patches_38-Cloud.csv'

        df = pd.read_csv(csv_path)
        return df['name'].tolist()

    def _split_train_val(self) -> List[str]:
        """학습/검증 데이터를 분할합니다."""
        random.seed(self.random_seed)
        patches = self.patches.copy()
        random.shuffle(patches)

        n_val = int(len(patches) * self.val_ratio)

        if self.split == 'val':
            return patches[:n_val]
        else:  # train
            return patches[n_val:]

    def _load_single_patch(self, patch_name: str) -> dict:
        """단일 패치를 로드합니다 (병렬 로딩용)."""
        band_images = []
        for band in self.bands:
            band_path = self._get_band_path(patch_name, band)
            img = np.array(Image.open(band_path), dtype=np.float32)
            band_images.append(img)

        image = np.stack(band_images, axis=0)

        if self.has_gt:
            gt_path = self._get_gt_path(patch_name)
            mask = np.array(Image.open(gt_path), dtype=np.int64)
            mask = (mask == 255).astype(np.int64)
        else:
            mask = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=np.int64)

        return {'image': image, 'mask': mask}

    def _preload_all_data(self):
        """모든 데이터를 메모리에 병렬로 미리 로드합니다."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(
            f"Preloading {len(self.patches)} patches to memory (parallel)...")

        # ThreadPoolExecutor로 병렬 로드 (I/O 바운드 작업에 적합)
        num_threads = min(32, len(self.patches))
        self._preloaded_data = [None] * len(self.patches)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(self._load_single_patch, patch_name): idx
                for idx, patch_name in enumerate(self.patches)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Preloading"):
                idx = futures[future]
                self._preloaded_data[idx] = future.result()

        print(
            f"Preloading complete. Memory usage: ~{len(self._preloaded_data) * 4 * 384 * 384 * 4 / 1e9:.2f} GB")

    def _get_band_path(self, patch_name: str, band: str) -> Path:
        """밴드 이미지 경로를 생성합니다."""
        prefix = 'train' if self.has_gt else 'test'
        folder = f'{prefix}_{band}'
        filename = f'{band}_{patch_name}.TIF'
        return self.data_dir / folder / filename

    def _get_gt_path(self, patch_name: str) -> Path:
        """GT 마스크 경로를 생성합니다."""
        filename = f'gt_{patch_name}.TIF'
        return self.data_dir / 'train_gt' / filename

    def __len__(self) -> int:
        return len(self.patches)

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
        patch_name = self.patches[idx]

        # 프리로드된 데이터가 있으면 사용
        if self._preloaded_data is not None:
            image = self._preloaded_data[idx]['image'].copy()
            mask = self._preloaded_data[idx]['mask'].copy()
        else:
            # 캐시된 로딩 함수 사용
            band_images = []
            for band in self.bands:
                band_path = str(self._get_band_path(patch_name, band))
                img = _cached_load_image(band_path)
                band_images.append(img)

            # 채널 축으로 스택 (C, H, W)
            image = np.stack(band_images, axis=0)

            # GT 마스크 로드
            if self.has_gt:
                gt_path = str(self._get_gt_path(patch_name))
                mask = _cached_load_mask(gt_path)
                # 원본에서 255=cloud, 0=clear -> 0=clear, 1=cloud로 변환
                mask = (mask == 255).astype(np.int64)
            else:
                # 테스트셋은 마스크 없음 -> 더미 마스크 생성
                mask = np.zeros(
                    (self.PATCH_SIZE, self.PATCH_SIZE), dtype=np.int64)

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
                'patch_name': patch_name,
                'dataset': '38-cloud',
            }
            return image, mask, metadata

        return image, mask

    @staticmethod
    def get_rgb_bands() -> List[str]:
        """RGB 밴드 리스트 반환"""
        return ['red', 'green', 'blue']

    @staticmethod
    def get_all_bands() -> List[str]:
        """모든 밴드 리스트 반환"""
        return ['red', 'green', 'blue', 'nir']


class Cloud38AdditionalDataset(Dataset):
    """
    95-Cloud에서 38-Cloud에 추가된 데이터만 로드하는 Dataset 클래스

    95-Cloud는 38-Cloud에 57개의 추가 장면을 더한 것입니다.
    """

    AVAILABLE_BANDS = ['red', 'green', 'blue', 'nir']
    PATCH_SIZE = 384
    CLASS_NAMES = ['clear', 'cloud']
    NUM_CLASSES = 2
    MAX_VALUE = 65535.0

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        use_nonempty: bool = True,
        bands: Optional[List[str]] = None,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        """
        Args:
            root: 95-Cloud 데이터 루트 디렉토리
            use_nonempty: 비어있지 않은 패치만 사용할지 여부
            bands: 사용할 밴드 리스트
            normalize: 이미지 정규화 여부
            transform: 변환 함수
            return_metadata: 메타데이터 반환 여부
        """
        self.root = Path(root) if root else Path(DEFAULT_95CLOUD_DIR)
        self.use_nonempty = use_nonempty
        self.bands = bands if bands else self.AVAILABLE_BANDS
        self.normalize = normalize
        self.transform = transform
        self.return_metadata = return_metadata

        # 데이터 경로
        self.data_dir = self.root / '95-cloud_training_only_additional_to38-cloud'

        # 패치 목록 로드
        self.patches = self._load_patches()

        print(
            f"Cloud38 Additional Dataset loaded: {len(self.patches)} patches")

    def _load_patches(self) -> List[str]:
        """패치 목록을 로드합니다."""
        if self.use_nonempty:
            csv_path = self.data_dir / 'training_patches_95-cloud_nonempty.csv'
        else:
            csv_path = self.data_dir / 'training_patches_95-cloud.csv'

        df = pd.read_csv(csv_path)

        # 38-cloud에 이미 포함된 패치 제외 (95-cloud 추가분만)
        nonempty_38_path = self.root / 'training_patches_38-cloud_nonempty.csv'
        if nonempty_38_path.exists():
            df_38 = pd.read_csv(nonempty_38_path)
            patches_38 = set(df_38['name'].tolist())
            patches = [p for p in df['name'].tolist() if p not in patches_38]
            return patches

        return df['name'].tolist()

    def _get_band_path(self, patch_name: str, band: str) -> Path:
        """밴드 이미지 경로를 생성합니다."""
        folder = f'train_{band}_additional_to38cloud'
        filename = f'{band}_{patch_name}.TIF'
        return self.data_dir / folder / filename

    def _get_gt_path(self, patch_name: str) -> Path:
        """GT 마스크 경로를 생성합니다."""
        filename = f'gt_{patch_name}.TIF'
        return self.data_dir / 'train_gt_additional_to38cloud' / filename

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """데이터셋에서 샘플을 가져옵니다."""
        patch_name = self.patches[idx]

        # 각 밴드 이미지 로드
        band_images = []
        for band in self.bands:
            band_path = self._get_band_path(patch_name, band)
            img = np.array(Image.open(band_path), dtype=np.float32)
            band_images.append(img)

        image = np.stack(band_images, axis=0)

        # GT 마스크 로드
        gt_path = self._get_gt_path(patch_name)
        mask = np.array(Image.open(gt_path), dtype=np.int64)
        mask = (mask == 255).astype(np.int64)

        # 정규화
        if self.normalize:
            image = image / self.MAX_VALUE

        # Transform 적용
        if self.transform is not None:
            image_hwc = image.transpose(1, 2, 0)
            transformed = self.transform(image=image_hwc, mask=mask)
            image = transformed['image'].transpose(2, 0, 1)
            mask = transformed['mask']

        # 텐서 변환
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long()

        if self.return_metadata:
            metadata = {
                'patch_name': patch_name,
                'dataset': '95-cloud-additional',
            }
            return image, mask, metadata

        return image, mask


class Cloud95Dataset(Dataset):
    """
    95-Cloud 전체 데이터셋 (38-Cloud + 추가 57개 장면)

    38-Cloud와 추가 데이터를 결합하여 전체 95-Cloud 데이터셋을 제공합니다.
    """

    AVAILABLE_BANDS = ['red', 'green', 'blue', 'nir']
    PATCH_SIZE = 384
    CLASS_NAMES = ['clear', 'cloud']
    NUM_CLASSES = 2
    MAX_VALUE = 65535.0

    def __init__(
        self,
        root_38cloud: Optional[Union[str, Path]] = None,
        root_95cloud: Optional[Union[str, Path]] = None,
        split: Literal['train', 'val', 'test'] = 'train',
        use_nonempty: bool = True,
        bands: Optional[List[str]] = None,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        val_ratio: float = 0.1,
        random_seed: int = 42,
        return_metadata: bool = False,
        preload: bool = False,
    ):
        """
        Args:
            root_38cloud: 38-Cloud 데이터 루트 디렉토리
            root_95cloud: 95-Cloud 데이터 루트 디렉토리
            split: 데이터 분할 ('train', 'val', 'test')
            use_nonempty: 비어있지 않은 패치만 사용할지 여부
            bands: 사용할 밴드 리스트
            normalize: 이미지 정규화 여부
            transform: 변환 함수
            val_ratio: 검증 데이터 비율
            random_seed: 랜덤 시드
            return_metadata: 메타데이터 반환 여부
            preload: True면 모든 데이터를 메모리에 미리 로드 (빠른 학습, 메모리 사용 증가)
        """
        self.root_38cloud = Path(
            root_38cloud) if root_38cloud else Path(DEFAULT_38CLOUD_DIR)
        self.root_95cloud = Path(
            root_95cloud) if root_95cloud else Path(DEFAULT_95CLOUD_DIR)
        self.split = split
        self.use_nonempty = use_nonempty
        self.bands = bands if bands else self.AVAILABLE_BANDS
        self.normalize = normalize
        self.transform = transform
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.return_metadata = return_metadata
        self.preload = preload
        self._preloaded_data = None

        # 테스트셋은 38-Cloud 테스트셋 사용
        if split == 'test':
            self._use_38cloud_test = True
            self.patches = self._load_test_patches()
        else:
            self._use_38cloud_test = False
            # 학습/검증용 전체 패치 로드
            self.patches = self._load_train_patches()
            self.patches = self._split_train_val()

        print(f"Cloud95 Dataset loaded: {len(self.patches)} patches ({split})")

        # 데이터 프리로드
        if preload:
            self._preload_all_data()

    def _load_test_patches(self) -> List[dict]:
        """테스트 패치 목록을 로드합니다."""
        test_dir = self.root_38cloud / '38-Cloud_test'
        csv_path = test_dir / 'test_patches_38-Cloud.csv'
        df = pd.read_csv(csv_path)

        return [{
            'name': name,
            'source': '38-cloud-test',
            'data_dir': test_dir,
        } for name in df['name'].tolist()]

    def _load_train_patches(self) -> List[dict]:
        """학습 패치 목록을 로드합니다 (38-Cloud + 95-Cloud 추가분)."""
        patches = []

        # 38-Cloud 학습 데이터
        train_dir_38 = self.root_38cloud / '38-Cloud_training'
        if self.use_nonempty:
            csv_38 = self.root_38cloud / 'training_patches_38-cloud_nonempty.csv'
        else:
            csv_38 = train_dir_38 / 'training_patches_38-Cloud.csv'

        df_38 = pd.read_csv(csv_38)
        patches_38_set = set(df_38['name'].tolist())

        for name in df_38['name'].tolist():
            patches.append({
                'name': name,
                'source': '38-cloud',
                'data_dir': train_dir_38,
            })

        # 95-Cloud 추가 데이터
        add_dir = self.root_95cloud / '95-cloud_training_only_additional_to38-cloud'
        if self.use_nonempty:
            csv_95 = add_dir / 'training_patches_95-cloud_nonempty.csv'
        else:
            csv_95 = add_dir / 'training_patches_95-cloud.csv'

        if csv_95.exists():
            df_95 = pd.read_csv(csv_95)
            for name in df_95['name'].tolist():
                # 38-cloud에 없는 패치만 추가
                if name not in patches_38_set:
                    patches.append({
                        'name': name,
                        'source': '95-cloud-additional',
                        'data_dir': add_dir,
                    })

        return patches

    def _split_train_val(self) -> List[dict]:
        """학습/검증 데이터를 분할합니다."""
        random.seed(self.random_seed)
        patches = self.patches.copy()
        random.shuffle(patches)

        n_val = int(len(patches) * self.val_ratio)

        if self.split == 'val':
            return patches[:n_val]
        else:  # train
            return patches[n_val:]

    def _load_single_patch(self, patch_info: dict) -> dict:
        """단일 패치를 로드합니다 (병렬 로딩용)."""
        band_images = []
        for band in self.bands:
            band_path = self._get_band_path(patch_info, band)
            img = np.array(Image.open(band_path), dtype=np.float32)
            band_images.append(img)

        image = np.stack(band_images, axis=0)

        if self._use_38cloud_test:
            mask = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=np.int64)
        else:
            gt_path = self._get_gt_path(patch_info)
            mask = np.array(Image.open(gt_path), dtype=np.int64)
            mask = (mask == 255).astype(np.int64)

        return {'image': image, 'mask': mask}

    def _preload_all_data(self):
        """모든 데이터를 메모리에 병렬로 미리 로드합니다."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(
            f"Preloading {len(self.patches)} patches to memory (parallel)...")

        # ThreadPoolExecutor로 병렬 로드 (I/O 바운드 작업에 적합)
        num_threads = min(32, len(self.patches))
        self._preloaded_data = [None] * len(self.patches)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(self._load_single_patch, patch_info): idx
                for idx, patch_info in enumerate(self.patches)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Preloading"):
                idx = futures[future]
                self._preloaded_data[idx] = future.result()

        print(
            f"Preloading complete. Memory usage: ~{len(self._preloaded_data) * 4 * 384 * 384 * 4 / 1e9:.2f} GB")

    def _get_band_path(self, patch_info: dict, band: str) -> Path:
        """밴드 이미지 경로를 생성합니다."""
        name = patch_info['name']
        data_dir = patch_info['data_dir']
        source = patch_info['source']

        if source == '38-cloud':
            folder = f'train_{band}'
        elif source == '38-cloud-test':
            folder = f'test_{band}'
        else:  # 95-cloud-additional
            folder = f'train_{band}_additional_to38cloud'

        filename = f'{band}_{name}.TIF'
        return data_dir / folder / filename

    def _get_gt_path(self, patch_info: dict) -> Path:
        """GT 마스크 경로를 생성합니다."""
        name = patch_info['name']
        data_dir = patch_info['data_dir']
        source = patch_info['source']

        if source == '38-cloud':
            folder = 'train_gt'
        else:  # 95-cloud-additional
            folder = 'train_gt_additional_to38cloud'

        filename = f'gt_{name}.TIF'
        return data_dir / folder / filename

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """데이터셋에서 샘플을 가져옵니다."""
        patch_info = self.patches[idx]

        # 프리로드된 데이터가 있으면 사용
        if self._preloaded_data is not None:
            image = self._preloaded_data[idx]['image'].copy()
            mask = self._preloaded_data[idx]['mask'].copy()
        else:
            # 캐시된 로딩 함수 사용
            band_images = []
            for band in self.bands:
                band_path = str(self._get_band_path(patch_info, band))
                img = _cached_load_image(band_path)
                band_images.append(img)

            image = np.stack(band_images, axis=0)

            # GT 마스크 로드
            if self._use_38cloud_test:
                # 테스트셋은 GT 없음
                mask = np.zeros(
                    (self.PATCH_SIZE, self.PATCH_SIZE), dtype=np.int64)
            else:
                gt_path = str(self._get_gt_path(patch_info))
                mask = _cached_load_mask(gt_path)
                mask = (mask == 255).astype(np.int64)

        # 정규화
        if self.normalize:
            image = image / self.MAX_VALUE

        # Transform 적용
        if self.transform is not None:
            image_hwc = image.transpose(1, 2, 0)
            transformed = self.transform(image=image_hwc, mask=mask)
            image = transformed['image'].transpose(2, 0, 1)
            mask = transformed['mask']

        # 텐서 변환
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long()

        if self.return_metadata:
            metadata = {
                'patch_name': patch_info['name'],
                'source': patch_info['source'],
                'dataset': '95-cloud',
            }
            return image, mask, metadata

        return image, mask

    @staticmethod
    def get_rgb_bands() -> List[str]:
        """RGB 밴드 리스트 반환"""
        return ['red', 'green', 'blue']

    @staticmethod
    def get_all_bands() -> List[str]:
        """모든 밴드 리스트 반환"""
        return ['red', 'green', 'blue', 'nir']


def create_dataloaders(
    root_38cloud: Optional[Union[str, Path]] = None,
    root_95cloud: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    use_95cloud: bool = True,
    use_nonempty: bool = True,
    bands: Optional[List[str]] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    학습, 검증, 테스트 데이터로더를 생성합니다.

    Args:
        root_38cloud: 38-Cloud 데이터 디렉토리 경로
        root_95cloud: 95-Cloud 데이터 디렉토리 경로
        batch_size: 배치 크기
        use_95cloud: True면 95-Cloud 전체 사용, False면 38-Cloud만 사용
        use_nonempty: 비어있지 않은 패치만 사용
        bands: 사용할 밴드 리스트
        num_workers: 데이터 로딩 워커 수
        pin_memory: GPU 학습 시 True 권장
        normalize: 이미지 정규화 여부
        train_transform: 학습 데이터 변환
        val_transform: 검증/테스트 데이터 변환
        val_ratio: 검증 데이터 비율
        random_seed: 랜덤 시드

    Returns:
        train_loader, val_loader, test_loader
    """
    if use_95cloud:
        train_dataset = Cloud95Dataset(
            root_38cloud=root_38cloud,
            root_95cloud=root_95cloud,
            split='train',
            use_nonempty=use_nonempty,
            bands=bands,
            normalize=normalize,
            transform=train_transform,
            val_ratio=val_ratio,
            random_seed=random_seed,
        )

        val_dataset = Cloud95Dataset(
            root_38cloud=root_38cloud,
            root_95cloud=root_95cloud,
            split='val',
            use_nonempty=use_nonempty,
            bands=bands,
            normalize=normalize,
            transform=val_transform,
            val_ratio=val_ratio,
            random_seed=random_seed,
        )

        test_dataset = Cloud95Dataset(
            root_38cloud=root_38cloud,
            root_95cloud=root_95cloud,
            split='test',
            use_nonempty=use_nonempty,
            bands=bands,
            normalize=normalize,
            transform=val_transform,
            val_ratio=val_ratio,
            random_seed=random_seed,
        )
    else:
        train_dataset = Cloud38Dataset(
            root=root_38cloud,
            split='train',
            use_nonempty=use_nonempty,
            bands=bands,
            normalize=normalize,
            transform=train_transform,
            val_ratio=val_ratio,
            random_seed=random_seed,
        )

        val_dataset = Cloud38Dataset(
            root=root_38cloud,
            split='val',
            use_nonempty=use_nonempty,
            bands=bands,
            normalize=normalize,
            transform=val_transform,
            val_ratio=val_ratio,
            random_seed=random_seed,
        )

        test_dataset = Cloud38Dataset(
            root=root_38cloud,
            split='test',
            use_nonempty=use_nonempty,
            bands=bands,
            normalize=normalize,
            transform=val_transform,
            val_ratio=val_ratio,
            random_seed=random_seed,
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


def get_class_weights(
    root_38cloud: Optional[Union[str, Path]] = None,
    root_95cloud: Optional[Union[str, Path]] = None,
    use_95cloud: bool = True,
    use_nonempty: bool = True,
    num_samples: int = 1000,
) -> torch.Tensor:
    """
    클래스별 가중치를 계산합니다 (클래스 불균형 해결용).

    Args:
        root_38cloud: 38-Cloud 데이터 디렉토리
        root_95cloud: 95-Cloud 데이터 디렉토리
        use_95cloud: 95-Cloud 사용 여부
        use_nonempty: 비어있지 않은 패치만 사용
        num_samples: 샘플링할 패치 수

    Returns:
        class_weights: 클래스별 가중치 텐서
    """
    if use_95cloud:
        dataset = Cloud95Dataset(
            root_38cloud=root_38cloud,
            root_95cloud=root_95cloud,
            split='train',
            use_nonempty=use_nonempty,
            normalize=False,
        )
    else:
        dataset = Cloud38Dataset(
            root=root_38cloud,
            split='train',
            use_nonempty=use_nonempty,
            normalize=False,
        )

    # 랜덤 샘플링
    sample_indices = random.sample(
        range(len(dataset)), min(num_samples, len(dataset)))

    class_counts = np.zeros(2, dtype=np.float64)

    for idx in sample_indices:
        _, mask = dataset[idx]
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        class_counts[0] += np.sum(mask_np == 0)  # Clear
        class_counts[1] += np.sum(mask_np == 1)  # Cloud

    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (2 * class_counts + 1e-10)
    weights = weights / weights.sum() * 2  # 정규화

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("38-Cloud / 95-Cloud Dataset Test")
    print("=" * 60)

    # 38-Cloud 데이터셋 테스트
    print("\n[1] Testing 38-Cloud Dataset")
    print("-" * 40)

    dataset_38 = Cloud38Dataset(
        root=DEFAULT_38CLOUD_DIR,
        split='train',
        use_nonempty=True,
        bands=Cloud38Dataset.get_all_bands(),
        normalize=True,
        return_metadata=True,
    )

    print(f"38-Cloud Train size: {len(dataset_38)}")

    # 샘플 가져오기
    image, label, metadata = dataset_38[0]
    print(f"\nSample shape:")
    print(f"  Image: {image.shape}, dtype: {image.dtype}")
    print(f"  Label: {label.shape}, dtype: {label.dtype}")
    print(f"  Metadata: {metadata}")

    # 클래스 분포
    unique, counts = torch.unique(label, return_counts=True)
    print(f"\nClass distribution in sample:")
    for u, c in zip(unique.tolist(), counts.tolist()):
        pct = c / label.numel() * 100
        class_name = dataset_38.CLASS_NAMES[u]
        print(f"  {u} ({class_name}): {c} pixels ({pct:.2f}%)")

    # 95-Cloud 데이터셋 테스트
    print("\n[2] Testing 95-Cloud Dataset")
    print("-" * 40)

    dataset_95 = Cloud95Dataset(
        root_38cloud=DEFAULT_38CLOUD_DIR,
        root_95cloud=DEFAULT_95CLOUD_DIR,
        split='train',
        use_nonempty=True,
        bands=Cloud95Dataset.get_all_bands(),
        normalize=True,
        return_metadata=True,
    )

    print(f"95-Cloud Train size: {len(dataset_95)}")

    # 샘플 가져오기
    image, label, metadata = dataset_95[0]
    print(f"\nSample shape:")
    print(f"  Image: {image.shape}, dtype: {image.dtype}")
    print(f"  Label: {label.shape}, dtype: {label.dtype}")
    print(f"  Metadata: {metadata}")

    # 시각화
    print("\n[3] Generating Visualization")
    print("-" * 40)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 레이블 색상 맵
    label_colors = {
        0: [0.2, 0.8, 0.2],    # Clear - 녹색
        1: [1.0, 1.0, 1.0],    # Cloud - 흰색
    }

    # 여러 샘플 시각화
    sample_indices = [i * (len(dataset_95) // 4) for i in range(4)]

    for i, idx in enumerate(sample_indices):
        image, label, metadata = dataset_95[idx]

        # RGB 이미지 (Red, Green, Blue 밴드)
        rgb = image[:3].numpy().transpose(1, 2, 0)
        rgb = np.clip(rgb * 3.0, 0, 1)  # 밝기 조정

        axes[0, i].imshow(rgb)
        title = f"{metadata['source']}\n{metadata['patch_name'][:30]}..."
        axes[0, i].set_title(title, fontsize=8)
        axes[0, i].axis('off')

        # 마스크
        label_np = label.numpy()
        label_rgb = np.zeros((*label_np.shape, 3))
        for class_id, color in label_colors.items():
            label_rgb[label_np == class_id] = color

        axes[1, i].imshow(label_rgb)
        axes[1, i].set_title('Cloud Mask')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('/home/junghwan/cloud_seg/cloud38_95_visualization.png', dpi=150)
    plt.show()
    print("\nVisualization saved to 'cloud38_95_visualization.png'")

    # DataLoader 테스트
    print("\n[4] Testing DataLoader")
    print("-" * 40)

    train_loader, val_loader, test_loader = create_dataloaders(
        root_38cloud=DEFAULT_38CLOUD_DIR,
        root_95cloud=DEFAULT_95CLOUD_DIR,
        batch_size=8,
        use_95cloud=True,
        bands=Cloud95Dataset.get_all_bands(),
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

    # 클래스 가중치 계산
    print("\n[5] Computing Class Weights")
    print("-" * 40)
    weights = get_class_weights(
        root_38cloud=DEFAULT_38CLOUD_DIR,
        root_95cloud=DEFAULT_95CLOUD_DIR,
        use_95cloud=True,
        num_samples=100,
    )
    print(f"Class weights: {weights.tolist()}")
    for i, (name, w) in enumerate(zip(Cloud95Dataset.CLASS_NAMES, weights.tolist())):
        print(f"  {name}: {w:.4f}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
