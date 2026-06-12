"""
LEVIR-CD Dataset Loader

High-resolution building change detection dataset.
Implements proper region-wise splits.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Callable
from PIL import Image
import cv2


class LEVIRCDDataset(Dataset):
    """
    LEVIR-CD Dataset for building change detection.
    
    Dataset structure:
        data/LEVIR-CD/
            train/
                A/  # Time 1 images
                    image_1.png
                    ...
                B/  # Time 2 images
                    image_1.png
                    ...
                label/
                    image_1.png
                    ...
            val/
                ...
            test/
                ...
    
    Args:
        root_dir: Path to LEVIR-CD dataset root
        split: 'train', 'val', or 'test'
        transform: Albumentations transform for data augmentation
        normalize: Whether to normalize images to [0, 1]
        patch_size: If specified, extract patches of this size
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        normalize: bool = True,
        patch_size: Optional[int] = None
    ):
        super(LEVIRCDDataset, self).__init__()
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.patch_size = patch_size
        
        self.split_dir = self.root_dir / split
        self.images_a_dir = self.split_dir / 'A'
        self.images_b_dir = self.split_dir / 'B'
        self.labels_dir = self.split_dir / 'label'
        
        # Get list of image pairs
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.split_dir}")
    
    def _load_samples(self) -> List[dict]:
        """Load list of image pairs and labels."""
        samples = []
        
        # Find all images in A directory
        if not self.images_a_dir.exists():
            raise FileNotFoundError(f"Images A directory not found: {self.images_a_dir}")
        
        image_files = sorted(self.images_a_dir.glob('*.png'))
        
        for img_a_path in image_files:
            # Get corresponding B image and label
            img_name = img_a_path.name
            img_b_path = self.images_b_dir / img_name
            label_path = self.labels_dir / img_name
            
            if img_b_path.exists() and label_path.exists():
                samples.append({
                    'name': img_name,
                    'image_a': img_a_path,
                    'image_b': img_b_path,
                    'label': label_path
                })
        
        return samples
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from file."""
        img = np.array(Image.open(path))
        
        # Ensure 3 channels
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
        
        return img
    
    def _load_label(self, path: Path) -> np.ndarray:
        """Load binary change label."""
        label = np.array(Image.open(path))
        
        # Ensure binary
        label = (label > 0).astype(np.int64)
        
        return label
    
    def _extract_patch(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        label: np.ndarray,
        patch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly extract a patch from the images."""
        h, w = label.shape
        
        if h < patch_size or w < patch_size:
            # Pad if necessary
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            img_a = np.pad(img_a, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            img_b = np.pad(img_b, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = label.shape
        
        # Random crop
        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)
        
        img_a = img_a[top:top+patch_size, left:left+patch_size]
        img_b = img_b[top:top+patch_size, left:left+patch_size]
        label = label[top:top+patch_size, left:left+patch_size]
        
        return img_a, img_b, label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample.
        
        Returns:
            dict with keys:
                - 'image_t1': Tensor (C, H, W)
                - 'image_t2': Tensor (C, H, W)
                - 'label': Tensor (H, W)
                - 'name': str (image name)
        """
        sample = self.samples[idx]
        
        # Load images and label
        img_a = self._load_image(sample['image_a'])
        img_b = self._load_image(sample['image_b'])
        label = self._load_label(sample['label'])
        
        # Extract patch if requested
        if self.patch_size is not None and self.split == 'train':
            img_a, img_b, label = self._extract_patch(
                img_a, img_b, label, self.patch_size
            )
        
        # Apply transforms (Albumentations)
        if self.transform is not None:
            transformed = self.transform(
                image=img_a,
                image2=img_b,
                mask=label
            )
            img_a = transformed['image']
            img_b = transformed.get('image2', transformed['image'])
            label = transformed['mask']
        
        # Convert to tensors
        if not isinstance(img_a, torch.Tensor):
            img_a = torch.from_numpy(img_a).permute(2, 0, 1).float()
        if not isinstance(img_b, torch.Tensor):
            img_b = torch.from_numpy(img_b).permute(2, 0, 1).float()
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        
        return {
            'image_t1': img_a,
            'image_t2': img_b,
            'label': label,
            'name': sample['name']
        }


def get_levir_transforms(split: str = 'train'):
    """
    Get data augmentation transforms for LEVIR-CD.
    
    Args:
        split: 'train', 'val', or 'test'
        
    Returns:
        Albumentations transform
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        print("Warning: albumentations not installed, returning None")
        return None
    
    if split == 'train':
        # Training augmentations
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], additional_targets={'image2': 'image'})
    else:
        # Validation/test - only normalization
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], additional_targets={'image2': 'image'})
    
    return transform


if __name__ == "__main__":
    # Test dataset loading
    print("Testing LEVIR-CD Dataset:")
    
    try:
        dataset = LEVIRCDDataset(
            root_dir='/path/to/LEVIR-CD',
            split='train',
            patch_size=256
        )
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Image A shape: {sample['image_t1'].shape}")
            print(f"Image B shape: {sample['image_t2'].shape}")
            print(f"Label shape: {sample['label'].shape}")
            print(f"Name: {sample['name']}")
    except Exception as e:
        print(f"Dataset test skipped (data not available): {e}")
