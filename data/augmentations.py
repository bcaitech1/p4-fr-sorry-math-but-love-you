from albumentations.pytorch import ToTensorV2
import albumentations as A


def get_train_transforms(height, width):
    aug_prob = 0.3
    return A.Compose(
        [
            A.Resize(height, width),
            A.ShiftScaleRotate(
                shift_limit=0.0, scale_limit=0.1, rotate_limit=0, p=aug_prob
            ),
            A.GridDistortion(
                p=aug_prob,
                num_steps=8,
                distort_limit=(-0.5, 0.5),
                interpolation=0,
                border_mode=0,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_valid_transforms(height, width):
    return A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ]
    )


def get_test_transforms(height, width):
    return A.Compose(
        [
            A.Resize(height, width, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
