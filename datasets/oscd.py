"""
OSCD (Onera Satellite Change Detection) Dataset Loader

Loads Sentinel-2 bi-temporal imagery with binary change labels.
Implements region-wise splits to prevent pixel leakage.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Callable
import rasterio
from PIL import Image
import cv2


class OSCDDataset(Dataset):
    """
    OSCD Dataset for bi-temporal change detection.
    
    Dataset structure:
        data/OSCD/
            train/
                images/
                    city1_t1.tif
                    city1_t2.tif
                    ...
                labels/
                    city1_label.tif
                    ...
            val/
                ...
            test/
                ...
    
    Args:
        root_dir: Path to OSCD dataset root
        split: 'train', 'val', or 'test'
        transform: Albumentations transform for data augmentation
        normalize: Whether to normalize images to [0, 1]
        use_rgb_only: If True, use only RGB bands (3 channels)
        patch_size: If specified, extract patches of this size
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        normalize: bool = True,
        use_rgb_only: bool = True,
        patch_size: Optional[int] = None
    ):
        super(OSCDDataset, self).__init__()
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.use_rgb_only = use_rgb_only
        self.patch_size = patch_size
        
        self.split_dir = self.root_dir / split
        self.images_dir = self.split_dir / 'images'
        self.labels_dir = self.split_dir / 'labels'
        
        # Get list of image pairs
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.split_dir}")
    
    def _load_samples(self) -> List[dict]:
        """Load list of image pairs and labels."""
        samples = []
        
        # Find all label files
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        label_files = sorted(self.labels_dir.glob('*.tif'))
        
        for label_file in label_files:
            # Extract city/region name from label filename
            # Assuming format: cityname_label.tif
            stem = label_file.stem
            region_name = stem.replace('_label', '')
            
            # Find corresponding t1 and t2 images
            t1_path = self.images_dir / f"{region_name}_t1.tif"
            t2_path = self.images_dir / f"{region_name}_t2.tif"
            
            if t1_path.exists() and t2_path.exists():
                samples.append({
                    'region': region_name,
                    't1': t1_path,
                    't2': t2_path,
                    'label': label_file
                })
        
        return samples
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from file."""
        try:
            # Try loading with rasterio (for multi-band TIF)
            with rasterio.open(path) as src:
                img = src.read()  # (C, H, W)
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
                
                # Select RGB bands if requested
                if self.use_rgb_only and img.shape[2] >= 3:
                    # Assuming RGB are first 3 bands
                    img = img[:, :, :3]
        except Exception:
            # Fallback to PIL/cv2 for standard images
            img = np.array(Image.open(path))
            if len(img.shape) == 2:
                img = np.expand_dims(img, -1)
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
        
        return img
    
    def _load_label(self, path: Path) -> np.ndarray:
        """Load binary change label."""
        try:
            with rasterio.open(path) as src:
                label = src.read(1)  # (H, W)
        except Exception:
            label = np.array(Image.open(path))
        
        # Ensure binary
        label = (label > 0).astype(np.int64)
        
        return label
    
    def _extract_patch(
        self,
        img_t1: np.ndarray,
        img_t2: np.ndarray,
        label: np.ndarray,
        patch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly extract a patch from the images."""
        h, w = label.shape
        
        if h < patch_size or w < patch_size:
            # Pad if necessary
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            img_t1 = np.pad(img_t1, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            img_t2 = np.pad(img_t2, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = label.shape
        
        # Random crop
        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)
        
        img_t1 = img_t1[top:top+patch_size, left:left+patch_size]
        img_t2 = img_t2[top:top+patch_size, left:left+patch_size]
        label = label[top:top+patch_size, left:left+patch_size]
        
        return img_t1, img_t2, label
    
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
                - 'region': str (region name)
        """
        sample = self.samples[idx]
        
        # Load images and label
        img_t1 = self._load_image(sample['t1'])
        img_t2 = self._load_image(sample['t2'])
        label = self._load_label(sample['label'])
        
        # Extract patch if requested
        if self.patch_size is not None and self.split == 'train':
            img_t1, img_t2, label = self._extract_patch(
                img_t1, img_t2, label, self.patch_size
            )
        
        # Apply transforms (Albumentations)
        if self.transform is not None:
            # Albumentations expects specific format
            transformed = self.transform(
                image=img_t1,
                image2=img_t2,
                mask=label
            )
            img_t1 = transformed['image']
            img_t2 = transformed.get('image2', transformed['image'])
            label = transformed['mask']
        
        # Convert to tensors
        if not isinstance(img_t1, torch.Tensor):
            img_t1 = torch.from_numpy(img_t1).permute(2, 0, 1).float()
        if not isinstance(img_t2, torch.Tensor):
            img_t2 = torch.from_numpy(img_t2).permute(2, 0, 1).float()
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        
        return {
            'image_t1': img_t1,
            'image_t2': img_t2,
            'label': label,
            'region': sample['region']
        }


def create_oscd_splits(
    root_dir: str,
    train_regions: List[str],
    val_regions: List[str],
    test_regions: List[str]
):
    """
    Create train/val/test splits based on region names.
    
    This ensures no pixel leakage between splits by separating
    entire regions/cities.
    
    Args:
        root_dir: Path to OSCD dataset
        train_regions: List of region names for training
        val_regions: List of region names for validation
        test_regions: List of region names for testing
    """
    import shutil
    
    root = Path(root_dir)
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (root / split / 'images').mkdir(parents=True, exist_ok=True)
        (root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Define region assignments
    splits_map = {
        'train': train_regions,
        'val': val_regions,
        'test': test_regions
    }
    
    # Move files to appropriate splits
    for split, regions in splits_map.items():
        for region in regions:
            # Find files for this region
            for suffix in ['_t1.tif', '_t2.tif']:
                src = root / 'images' / f"{region}{suffix}"
                if src.exists():
                    dst = root / split / 'images' / f"{region}{suffix}"
                    shutil.copy2(src, dst)
            
            label_src = root / 'labels' / f"{region}_label.tif"
            if label_src.exists():
                label_dst = root / split / 'labels' / f"{region}_label.tif"
                shutil.copy2(label_src, label_dst)


if __name__ == "__main__":
    # Test dataset loading
    print("Testing OSCD Dataset:")
    
    # This is just for testing structure - actual data would need to exist
    try:
        dataset = OSCDDataset(
            root_dir='/path/to/OSCD',
            split='train',
            use_rgb_only=True,
            patch_size=256
        )
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Image t1 shape: {sample['image_t1'].shape}")
            print(f"Image t2 shape: {sample['image_t2'].shape}")
            print(f"Label shape: {sample['label'].shape}")
            print(f"Region: {sample['region']}")
    except Exception as e:
        print(f"Dataset test skipped (data not available): {e}")
