from Scripts.config import Config
import albumentations as A

training_augmentations = A.Compose(
    [
        A.CoarseDropout(p=0.6),
        A.RandomRotate90(p=0.6),
        A.Flip(p=0.4),
        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=60, val_shift_limit=50
                ),
            ],
            p=0.7,
        ),
        A.OneOf([A.GaussianBlur(), A.GaussNoise()], p=0.65),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.35, rotate_limit=45, p=0.5
        ),
        A.OneOf(
            [
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ],
            p=0.7,
        ),
        A.Normalize(
            mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0, always_apply=True
        ),
    ]
)

validation_augmentations = A.Compose(
    [
        A.Normalize(
            mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0, always_apply=True
        )
    ]
)
testing_augmentations = A.Compose(
    [
        A.Normalize(
            mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0, always_apply=True
        )
    ]
)
