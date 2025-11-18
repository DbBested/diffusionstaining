"""
H&E to IHC Dataset with MONAI Transforms
Supports paired histopathology images for diffusion model training
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandRotated,
    RandFlipd,
    RandZoomd,
    RandAdjustContrastd,
    Resized,
    ToTensord,
)

# Disable PIL decompression bomb protection for large medical images
Image.MAX_IMAGE_PIXELS = None


class HEtoIHCDataset:
    """
    Dataset wrapper for H&E to IHC paired images

    Args:
        root_dir: Root directory containing train/test folders
        split: 'train' or 'test'
        he_subdir: Subdirectory name for H&E images
        ihc_subdir: Subdirectory name for IHC images
        image_size: Target image size (assumes square images)
        cache_rate: Fraction of data to cache in RAM (0.0-1.0)
        augmentation: Whether to apply data augmentation
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        he_subdir: str = "HE",
        ihc_subdir: str = "IHC",
        image_size: int = 256,
        cache_rate: float = 1.0,
        augmentation: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.cache_rate = cache_rate
        self.augmentation = augmentation and (split == "train")

        # Construct paths
        self.he_dir = self.root_dir / split / he_subdir
        self.ihc_dir = self.root_dir / split / ihc_subdir

        # Validate directories
        if not self.he_dir.exists():
            raise ValueError(f"H&E directory not found: {self.he_dir}")
        if not self.ihc_dir.exists():
            raise ValueError(f"IHC directory not found: {self.ihc_dir}")

        # Build data list
        self.data_dicts = self._build_data_list()

        # Create transforms
        self.transforms = self._get_transforms()

        # Create MONAI dataset
        if cache_rate > 0:
            self.dataset = CacheDataset(
                data=self.data_dicts,
                transform=self.transforms,
                cache_rate=cache_rate,
                num_workers=4,
            )
        else:
            self.dataset = Dataset(
                data=self.data_dicts,
                transform=self.transforms,
            )

    def _build_data_list(self) -> List[Dict[str, str]]:
        """
        Build list of paired images
        Assumes corresponding filenames between H&E and IHC directories
        """
        he_files = sorted(self.he_dir.glob("*.png")) + sorted(self.he_dir.glob("*.jpg"))
        data_dicts = []

        for he_file in he_files:
            # Find corresponding IHC file
            ihc_file = self.ihc_dir / he_file.name

            # Also try with different extensions
            if not ihc_file.exists():
                stem = he_file.stem
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    ihc_file = self.ihc_dir / f"{stem}{ext}"
                    if ihc_file.exists():
                        break

            if ihc_file.exists():
                data_dicts.append({
                    "he": str(he_file),
                    "ihc": str(ihc_file),
                })
            else:
                print(f"Warning: No matching IHC file for {he_file.name}")

        if len(data_dicts) == 0:
            raise ValueError(f"No paired images found in {self.he_dir} and {self.ihc_dir}")

        print(f"Found {len(data_dicts)} paired images for {self.split} split")
        return data_dicts

    def _get_transforms(self) -> Compose:
        """
        Create MONAI transform pipeline
        """
        keys = ["he", "ihc"]

        # Base transforms (always applied)
        base_transforms = [
            LoadImaged(keys=keys, image_only=True),
            EnsureChannelFirstd(keys=keys),
            ScaleIntensityd(keys=keys, minv=-1.0, maxv=1.0),  # Scale to [-1, 1]
            Resized(keys=keys, spatial_size=(self.image_size, self.image_size)),
        ]

        # Augmentation transforms (only for training)
        aug_transforms = []
        if self.augmentation:
            aug_transforms = [
                RandRotated(
                    keys=keys,
                    range_x=np.pi / 12,  # ±15 degrees
                    prob=0.5,
                    mode="bilinear",
                    padding_mode="border",
                ),
                RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
                RandZoomd(
                    keys=keys,
                    min_zoom=0.9,
                    max_zoom=1.1,
                    prob=0.3,
                    mode="bilinear",
                    padding_mode="edge",
                ),
                RandAdjustContrastd(
                    keys=keys,
                    gamma=(0.8, 1.2),
                    prob=0.3,
                ),
            ]

        # Final transforms
        final_transforms = [
            ToTensord(keys=keys),
        ]

        return Compose(base_transforms + aug_transforms + final_transforms)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]


def get_dataloader(
    config: Dict,
    split: str = "train",
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for H&E to IHC dataset

    Args:
        config: Configuration dictionary
        split: 'train' or 'test'
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    dataset = HEtoIHCDataset(
        root_dir=config["data"]["root_dir"],
        split=split,
        he_subdir=config["data"]["he_subdir"],
        ihc_subdir=config["data"]["ihc_subdir"],
        image_size=config["data"]["image_size"],
        cache_rate=config["data"]["cache_rate"] if split == "train" else 0.0,
        augmentation=(split == "train"),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    # Test dataset loading
    import yaml

    with open("../configs/baseline.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Testing dataset loading...")
    dataloader = get_dataloader(config, split="train")

    batch = next(iter(dataloader))
    print(f"H&E shape: {batch['he'].shape}")
    print(f"IHC shape: {batch['ihc'].shape}")
    print(f"H&E range: [{batch['he'].min():.2f}, {batch['he'].max():.2f}]")
    print(f"IHC range: [{batch['ihc'].min():.2f}, {batch['ihc'].max():.2f}]")
    print("Dataset loading successful!")
