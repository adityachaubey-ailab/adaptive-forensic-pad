import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Returns the training augmentations pipeline.
    
    Args:
        image_size (int): The target image size.
        
    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    return A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(p=0.2),
        A.ImageCompression(quality_lower=50, p=0.3),
        A.RandomResizedCrop(image_size, image_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Returns the validation augmentations pipeline.
    
    Args:
        image_size (int): The target image size.
        
    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    return A.Compose([
        A.Resize(int(image_size * 1.14), int(image_size * 1.14)),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_attack_specific_augments() -> A.Compose:
    """
    Returns augmentations that simulate attack artifacts like moire, halftones, compression, etc.
    
    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    return A.Compose([
        A.ImageCompression(quality_lower=20, quality_upper=60, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.5),
        A.ISONoise(p=0.5),
        # Assuming moire and halftone simulations can be approximated with certain noises/blurs
        A.AdvancedBlur(p=0.3),
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3)
    ])
