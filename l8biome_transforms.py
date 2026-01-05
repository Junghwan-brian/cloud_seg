"""
L8 Biome Dataset용 데이터 증강 및 변환 유틸리티

albumentations 라이브러리를 사용한 데이터 증강 파이프라인 제공

사용 예시:
    from l8biome_transforms import get_train_transforms, get_val_transforms
    from l8biome_dataset import L8BiomeDataset
    
    train_transform = get_train_transforms(patch_size=512)
    val_transform = get_val_transforms(patch_size=512)
    
    train_dataset = L8BiomeDataset(
        split='train',
        patch_size=512,
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
    patch_size: int = 512,
    use_geometric: bool = True,
    use_color: bool = True,
    use_noise: bool = False,
):
    """
    학습용 데이터 증강 변환을 반환합니다.

    Args:
        patch_size: 출력 패치 크기
        use_geometric: 기하학적 변환 사용 여부
        use_color: 색상/밝기 변환 사용 여부
        use_noise: 노이즈 추가 여부

    Returns:
        albumentations 변환 객체
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for transforms")

    transforms_list = []

    # 기하학적 변환
    if use_geometric:
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                border_mode=0,  # cv2.BORDER_CONSTANT
                value=0,
                mask_value=255,  # IGNORE_INDEX
                p=0.5,
            ),
        ])

    # 색상/밝기 변환
    if use_color:
        transforms_list.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
        ])

    # 노이즈 추가
    if use_noise:
        transforms_list.extend([
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
        ])

    return A.Compose(transforms_list)


def get_val_transforms(patch_size: int = 512):
    """
    검증/테스트용 변환을 반환합니다 (증강 없음).

    Args:
        patch_size: 출력 패치 크기

    Returns:
        albumentations 변환 객체
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for transforms")

    return A.Compose([
        # 검증/테스트 시에는 변환 없음
    ])


def get_strong_train_transforms(patch_size: int = 512):
    """
    강한 데이터 증강을 포함한 학습용 변환을 반환합니다.

    Args:
        patch_size: 출력 패치 크기

    Returns:
        albumentations 변환 객체
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required for transforms")

    return A.Compose([
        # 기하학적 변환
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                border_mode=0,
                value=0,
                mask_value=255,
                p=1.0,
            ),
            A.Affine(
                scale=(0.8, 1.2),
                rotate=(-45, 45),
                shear=(-10, 10),
                mode=0,
                cval=0,
                cval_mask=255,
                p=1.0,
            ),
        ], p=0.5),

        # 색상/밝기 변환
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),

        # 노이즈/블러
        A.OneOf([
            A.GaussNoise(var_limit=(0.001, 0.02), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),

        # 날씨 효과 (클라우드 검출에 유용)
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=1.0),
        ], p=0.2),

        # Cutout/Dropout
        A.CoarseDropout(
            max_holes=8,
            max_height=patch_size // 16,
            max_width=patch_size // 16,
            min_holes=1,
            min_height=patch_size // 32,
            min_width=patch_size // 32,
            fill_value=0,
            mask_fill_value=255,
            p=0.2,
        ),
    ])


class MixUp:
    """
    MixUp 데이터 증강

    두 이미지와 마스크를 섞어 새로운 샘플을 생성합니다.
    """

    def __init__(self, alpha: float = 0.4, p: float = 0.5):
        """
        Args:
            alpha: Beta 분포의 파라미터
            p: MixUp 적용 확률
        """
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        image1: np.ndarray,
        mask1: np.ndarray,
        image2: np.ndarray,
        mask2: np.ndarray,
    ):
        """
        Args:
            image1, mask1: 첫 번째 이미지와 마스크
            image2, mask2: 두 번째 이미지와 마스크

        Returns:
            mixed_image, mixed_mask, lambda_value
        """
        if np.random.random() > self.p:
            return image1, mask1, 1.0

        lam = np.random.beta(self.alpha, self.alpha)

        mixed_image = lam * image1 + (1 - lam) * image2

        # 마스크는 lambda가 0.5보다 크면 첫 번째 마스크 사용
        if lam >= 0.5:
            mixed_mask = mask1
        else:
            mixed_mask = mask2

        return mixed_image, mixed_mask, lam


class CutMix:
    """
    CutMix 데이터 증강

    한 이미지의 일부 영역을 다른 이미지로 대체합니다.
    """

    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Args:
            alpha: Beta 분포의 파라미터
            p: CutMix 적용 확률
        """
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        image1: np.ndarray,
        mask1: np.ndarray,
        image2: np.ndarray,
        mask2: np.ndarray,
    ):
        """
        Args:
            image1, mask1: 첫 번째 이미지와 마스크
            image2, mask2: 두 번째 이미지와 마스크

        Returns:
            mixed_image, mixed_mask, lambda_value
        """
        if np.random.random() > self.p:
            return image1, mask1, 1.0

        lam = np.random.beta(self.alpha, self.alpha)

        if len(image1.shape) == 3:
            _, h, w = image1.shape
        else:
            h, w = image1.shape

        # 자를 영역 계산
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        # 랜덤 위치
        cy = np.random.randint(h)
        cx = np.random.randint(w)

        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(w, cx + cut_w // 2)
        bby2 = min(h, cy + cut_h // 2)

        # 이미지 섞기
        mixed_image = image1.copy()
        mixed_mask = mask1.copy()

        if len(image1.shape) == 3:
            mixed_image[:, bby1:bby2,
                        bbx1:bbx2] = image2[:, bby1:bby2, bbx1:bbx2]
        else:
            mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]

        mixed_mask[bby1:bby2, bbx1:bbx2] = mask2[bby1:bby2, bbx1:bbx2]

        # 실제 lambda 값 계산
        lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (h * w)

        return mixed_image, mixed_mask, lam


# Numpy 기반 간단한 변환 (albumentations 없이 사용 가능)
class SimpleTransforms:
    """
    albumentations 없이 사용할 수 있는 간단한 변환들
    """

    @staticmethod
    def random_flip(image: np.ndarray, mask: np.ndarray, p: float = 0.5):
        """랜덤 수평/수직 플립"""
        if np.random.random() < p:
            image = np.flip(image, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()

        if np.random.random() < p:
            image = np.flip(image, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()

        return image, mask

    @staticmethod
    def random_rotate90(image: np.ndarray, mask: np.ndarray, p: float = 0.5):
        """랜덤 90도 회전"""
        if np.random.random() < p:
            k = np.random.randint(1, 4)  # 90, 180, 270도

            if len(image.shape) == 3:
                image = np.rot90(image, k, axes=(1, 2)).copy()
            else:
                image = np.rot90(image, k).copy()

            mask = np.rot90(mask, k).copy()

        return image, mask

    @staticmethod
    def random_brightness(image: np.ndarray, limit: float = 0.2, p: float = 0.5):
        """랜덤 밝기 조정"""
        if np.random.random() < p:
            factor = 1.0 + np.random.uniform(-limit, limit)
            image = np.clip(image * factor, 0, 1)

        return image


if __name__ == '__main__':
    print("Testing L8 Biome Transforms")
    print("-" * 40)

    if ALBUMENTATIONS_AVAILABLE:
        print("albumentations available!")

        # 변환 테스트
        train_tf = get_train_transforms(patch_size=512)
        val_tf = get_val_transforms(patch_size=512)
        strong_tf = get_strong_train_transforms(patch_size=512)

        # 더미 데이터로 테스트
        dummy_image = np.random.rand(512, 512, 3).astype(np.float32)
        dummy_mask = np.random.randint(0, 4, (512, 512)).astype(np.int64)

        result = train_tf(image=dummy_image, mask=dummy_mask)
        print(
            f"Train transform output shape: {result['image'].shape}, {result['mask'].shape}")

        result = strong_tf(image=dummy_image, mask=dummy_mask)
        print(
            f"Strong transform output shape: {result['image'].shape}, {result['mask'].shape}")
    else:
        print("albumentations not available, using simple transforms")

        # 간단한 변환 테스트
        dummy_image = np.random.rand(3, 512, 512).astype(np.float32)
        dummy_mask = np.random.randint(0, 4, (512, 512)).astype(np.int64)

        img, msk = SimpleTransforms.random_flip(dummy_image, dummy_mask)
        print(f"Simple flip output shape: {img.shape}, {msk.shape}")

    # MixUp/CutMix 테스트
    print("\nTesting MixUp/CutMix")
    mixup = MixUp(alpha=0.4, p=1.0)
    cutmix = CutMix(alpha=1.0, p=1.0)

    img1 = np.random.rand(3, 256, 256).astype(np.float32)
    mask1 = np.zeros((256, 256), dtype=np.int64)
    img2 = np.random.rand(3, 256, 256).astype(np.float32)
    mask2 = np.ones((256, 256), dtype=np.int64)

    mixed_img, mixed_mask, lam = mixup(img1, mask1, img2, mask2)
    print(f"MixUp lambda: {lam:.3f}, output shape: {mixed_img.shape}")

    mixed_img, mixed_mask, lam = cutmix(img1, mask1, img2, mask2)
    print(f"CutMix lambda: {lam:.3f}, output shape: {mixed_img.shape}")

    print("\nTest Complete!")
