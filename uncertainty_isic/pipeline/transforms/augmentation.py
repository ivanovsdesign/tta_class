import albumentations as A


def get_transforms(img_size: int):
    transforms = {'train' :  A.Compose([
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-25, 25), p=0.5), 
            
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0))
            ], p=0.7),
            
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.0),
                A.ElasticTransform(alpha=3),
            ], p=0.7),
            
            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
            
            A.Resize(height=img_size, width=img_size, p=1.0),
            A.CoarseDropout(max_holes=5, max_height=int(img_size * 0.15), max_width=int(img_size * 0.15), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, p=1)
        ]),
        
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
        ])

        }
    
    return transforms