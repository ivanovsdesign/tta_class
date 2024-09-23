import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: int):
    transforms = {'train_old' :  A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            )]
            ),
        
        'valid' : A.Compose([
            A.Resize(height=img_size, width=img_size, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, p=1)
        ]),

        'tta' : A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.Resize(height=img_size, width=img_size, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, p=1)
        ]),

        'train' : A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.Resize(height=img_size, width=img_size, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, p=1)
        ])

        }
    
    return transforms