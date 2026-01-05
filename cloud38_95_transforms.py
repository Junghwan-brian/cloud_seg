"""
38-Cloud / 95-Cloud Dataset용 데이터 증강 및 변환 유틸리티

albumentations 라이브러리를 사용한 데이터 증강 파이프라인 제공

사용 예시:
    from cloud38_95_transforms import get_train_transforms, get_val_transforms
    from cloud38_95_dataset import Cloud95Dataset
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = Cloud95Dataset(
        split='train',
        transform=train_transform,
    )
"""

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Run: pip install albumentations")

import numpy as np


def get_train_transforms(
    use_geometric: bool = True,
    use_color: bool = True,
    use_noise: bool = False,
    resize: int = None,
):
    """
    학습용 데이터 증강 변환을 반환합니다.

    Args:
        use_geometric: 기하학적 변환 사용 여부
        use_color: 색상/밝기 변환 사용 여부
        use_noise: 노이즈 추가 여부
        resize: 리사이즈할 크기 (None이면 원본 크기 유지, 384)

    Returns:
        albumentations 변환 객체
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for transforms")

    transforms_list = []

    # 리사이즈
    if resize is not None:
        transforms_list.append(
            A.Resize(height=resize, width=resize, p=1.0)
        )

    # 기하학적 변환
    if use_geometric:
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=0,  # cv2.BORDER_CONSTANT
                value=0,
                mask_value=0,  # Clear로 채움
                p=0.3,
            ),
        ])

    # 색상/밝기 변환 (위성 이미지에 적합한 파라미터)
    if use_color:
        transforms_list.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
        ])

    # 노이즈 추가
    if use_noise:
        transforms_list.extend([
            A.GaussNoise(var_limit=(0.0001, 0.001), p=0.2),
        ])

    return A.Compose(transforms_list)


def get_val_transforms(resize: int = None):
    """
    검증/테스트용 변환을 반환합니다 (증강 없음).

    Args:
        resize: 리사이즈할 크기 (None이면 원본 크기 유지)

    Returns:
        albumentations 변환 객체
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for transforms")

    transforms_list = []

    if resize is not None:
        transforms_list.append(
            A.Resize(height=resize, width=resize, p=1.0)
        )

    # 변환이 없어도 Compose 객체 반환
    return A.Compose(transforms_list) if transforms_list else None


def get_strong_augmentation(resize: int = None):
    """
    강한 데이터 증강 변환을 반환합니다.
    
    과적합 방지를 위한 더 강한 증강 파이프라인

    Returns:
        albumentations 변환 객체
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for transforms")

    transforms_list = []

    if resize is not None:
        transforms_list.append(
            A.Resize(height=resize, width=resize, p=1.0)
        )

    transforms_list.extend([
        # 기하학적 변환
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=0,
            value=0,
            mask_value=0,
            p=0.5,
        ),
        # 색상 변환
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # 노이즈
        A.GaussNoise(var_limit=(0.0001, 0.002), p=0.3),
        # 블러
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
    ])

    return A.Compose(transforms_list)


def get_tta_transforms():
    """
    Test-Time Augmentation (TTA)용 변환 리스트를 반환합니다.
    
    Returns:
        변환 리스트 (각각 적용 후 예측 결과를 평균)
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for transforms")

    return [
        None,  # 원본
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.VerticalFlip(p=1.0)]),
        A.Compose([A.RandomRotate90(p=1.0)]),  # 90도 회전
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
        ]),  # 180도 회전
    ]


def get_inverse_tta_transforms():
    """
    TTA 결과를 원본 방향으로 되돌리는 역변환 리스트를 반환합니다.
    
    Returns:
        역변환 함수 리스트
    """
    def identity(x):
        return x
    
    def horizontal_flip(x):
        return np.flip(x, axis=-1).copy()
    
    def vertical_flip(x):
        return np.flip(x, axis=-2).copy()
    
    def rotate90_back(x):
        return np.rot90(x, k=-1, axes=(-2, -1)).copy()
    
    def rotate180(x):
        return np.rot90(x, k=2, axes=(-2, -1)).copy()
    
    return [
        identity,
        horizontal_flip,
        vertical_flip,
        rotate90_back,
        rotate180,
    ]


class RandomCrop:
    """
    랜덤 크롭 변환 (384x384 → 임의 크기)
    """
    def __init__(self, crop_size: int = 256):
        self.crop_size = crop_size
        if ALBUMENTATIONS_AVAILABLE:
            self.transform = A.RandomCrop(
                height=crop_size, 
                width=crop_size, 
                p=1.0
            )
        
    def __call__(self, image, mask):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required")
        
        result = self.transform(image=image, mask=mask)
        return result['image'], result['mask']


class CenterCrop:
    """
    중앙 크롭 변환
    """
    def __init__(self, crop_size: int = 256):
        self.crop_size = crop_size
        if ALBUMENTATIONS_AVAILABLE:
            self.transform = A.CenterCrop(
                height=crop_size, 
                width=crop_size, 
                p=1.0
            )
        
    def __call__(self, image, mask):
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required")
        
        result = self.transform(image=image, mask=mask)
        return result['image'], result['mask']


def normalize_image(image: np.ndarray, max_value: float = 65535.0) -> np.ndarray:
    """
    이미지를 0-1 범위로 정규화합니다.
    
    Args:
        image: 입력 이미지 (uint16)
        max_value: 최대값 (uint16의 경우 65535)
    
    Returns:
        정규화된 이미지 (float32)
    """
    return image.astype(np.float32) / max_value


def denormalize_image(image: np.ndarray, max_value: float = 65535.0) -> np.ndarray:
    """
    정규화된 이미지를 원래 범위로 되돌립니다.
    
    Args:
        image: 정규화된 이미지 (0-1)
        max_value: 최대값
    
    Returns:
        역정규화된 이미지
    """
    return (image * max_value).astype(np.uint16)


if __name__ == '__main__':
    print("=" * 60)
    print("38-Cloud / 95-Cloud Transforms Test")
    print("=" * 60)
    
    if not ALBUMENTATIONS_AVAILABLE:
        print("albumentations not available!")
        exit(1)
    
    import torch
    from cloud38_95_dataset import Cloud95Dataset
    
    # 변환 생성
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    strong_transform = get_strong_augmentation()
    
    print(f"\nTrain transform: {train_transform}")
    print(f"Val transform: {val_transform}")
    print(f"Strong transform: {strong_transform}")
    
    # 데이터셋과 함께 테스트
    print("\n[1] Testing with Dataset")
    print("-" * 40)
    
    dataset = Cloud95Dataset(
        root_38cloud='/home/telepix_nas/junghwan/cloud_seg/38-cloud',
        root_95cloud='/home/telepix_nas/junghwan/cloud_seg/95-cloud',
        split='train',
        use_nonempty=True,
        bands=['red', 'green', 'blue', 'nir'],
        normalize=True,
        transform=train_transform,
        return_metadata=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 샘플 가져오기
    image, label, metadata = dataset[0]
    print(f"Sample shape: image={image.shape}, label={label.shape}")
    
    # 동일 샘플에 증강 여러 번 적용
    print("\n[2] Testing Augmentation Variations")
    print("-" * 40)
    
    for i in range(3):
        img, lbl, _ = dataset[0]
        cloud_ratio = (lbl == 1).float().mean() * 100
        print(f"  Attempt {i+1}: cloud ratio = {cloud_ratio:.2f}%")
    
    # TTA 테스트
    print("\n[3] Testing TTA Transforms")
    print("-" * 40)
    
    tta_transforms = get_tta_transforms()
    inverse_tta = get_inverse_tta_transforms()
    
    print(f"Number of TTA transforms: {len(tta_transforms)}")
    
    # 원본 이미지
    original_image = np.random.rand(4, 384, 384).astype(np.float32)
    
    for i, (tta, inv) in enumerate(zip(tta_transforms, inverse_tta)):
        if tta is not None:
            # HWC 형태로 변환
            img_hwc = original_image.transpose(1, 2, 0)
            augmented = tta(image=img_hwc)['image']
            augmented_chw = augmented.transpose(2, 0, 1)
        else:
            augmented_chw = original_image
        
        # 역변환
        restored = inv(augmented_chw)
        
        # 원본과 비교
        diff = np.abs(original_image - restored).mean()
        print(f"  TTA {i}: diff from original = {diff:.6f}")
    
    print("\n" + "=" * 60)
    print("Transforms Test Complete!")
    print("=" * 60)

