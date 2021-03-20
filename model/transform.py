from albumentations import (
    PadIfNeeded, HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize,ToGray
)

from albumentations.pytorch import ToTensorV2

def get_train_transforms(input_shape, way="pad", crop_rate=0.9):
    if way == "pad":
        return Compose([
                PadIfNeeded(input_shape[0], input_shape[1]),
                Resize(input_shape[0], input_shape[1]),
                HorizontalFlip(p=0.5),
                ToGray(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.0, p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)
    elif way == "resize":
        return Compose([
                #PadIfNeeded(input_shape[0], input_shape[1]),
                RandomResizedCrop(input_shape[0], input_shape[1]),
                HorizontalFlip(p=0.5),
                ToGray(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.0, p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)
    elif way == "center":
        return Compose([
                Resize(input_shape[0], input_shape[1]),
                HorizontalFlip(p=0.5),
                ToGray(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.0, p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #CoarseDropout(p=0.5),
                Cutout(num_holes=20,p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)
    elif way == "crop":
        return Compose([
                Resize(input_shape[0], input_shape[1]),
                CenterCrop(int(input_shape[0]*crop_rate), int(input_shape[1]*crop_rate)),
                HorizontalFlip(p=0.5),
                ToGray(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.0, p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)



def get_valid_transforms(input_shape, way="pad", crop_rate=0.9):
    if way == "pad":
        return Compose([
                PadIfNeeded(input_shape[0], input_shape[1]),
                Resize(input_shape[0], input_shape[1]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)
    elif way == "resize":
        return Compose([
                RandomResizedCrop(input_shape[0], input_shape[1]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)
    elif way == "center":
        return Compose([
                Resize(input_shape[0], input_shape[1]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)
    elif way == "crop":
        return Compose([
                Resize(input_shape[0], input_shape[1]),
                CenterCrop(int(input_shape[0]*crop_rate), int(input_shape[1]*crop_rate)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)

def get_inference_transforms(input_shape, way="pad", crop_rate=1.0):
    if way == "pad":
        return Compose([
                PadIfNeeded(input_shape[0], input_shape[1]),
                Resize(input_shape[0], input_shape[1]),
                HorizontalFlip(p=0.5),
                ToGray(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.0, p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)
    elif way == "resize":
        return Compose([
                RandomResizedCrop(input_shape[0], input_shape[1]),
                HorizontalFlip(p=0.5),
                ToGray(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.0, p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)
    elif way == "center":
        return Compose([
                Resize(input_shape[0], input_shape[1]),
                HorizontalFlip(p=0.5),
                ToGray(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.0, p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)
    elif way == "crop":
        return Compose([
                Resize(input_shape[0], input_shape[1]),
                CenterCrop(int(input_shape[0]*crop_rate), int(input_shape[1]*crop_rate)),
                HorizontalFlip(p=0.5),
                ToGray(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(scale_limit=0.0, p=0.5),
                HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                #CoarseDropout(p=0.5),
                Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)