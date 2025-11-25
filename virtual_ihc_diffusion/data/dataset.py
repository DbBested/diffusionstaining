"""
H&E to IHC Dataset with MONAI Transforms
Supports paired histopathology images for diffusion model training

Now supports the MIST HER2 layout:

  /path/to/TrainValAB/
    trainA  (H&E)
    trainB  (IHC, e.g. HER2)
    valA
    valB

Usage for MIST:
  root_dir = "/orcd/home/002/tomli/orcd/scratch/data/mist_her2/HER2_raw/HER2/TrainValAB"
  split = "train"  -> uses trainA / trainB
  split = "val"    -> uses valA / valB
  split = "test"   -> also uses valA / valB
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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
    Dataset wrapper for H&E to IHC paired images.

    Supports two directory layouts:

    1) MIST TrainValAB layout
       root_dir/
         trainA  (H&E)
         trainB  (IHC)
         valA
         valB

    2) Classic layout
       root_dir/
         train/
           HE/
           IHC/
         test/
           HE/
           IHC/
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

        # Detect MIST TrainValAB layout
        self.use_mist_layout = (
            (self.root_dir / "trainA").exists()
            and (self.root_dir / "trainB").exists()
        )

        if self.use_mist_layout:
            # MIST HER2 layout
            if split == "train":
                self.he_dir = self.root_dir / "trainA"
                self.ihc_dir = self.root_dir / "trainB"
            elif split in ("val", "test"):
                self.he_dir = self.root_dir / "valA"
                self.ihc_dir = self.root_dir / "valB"
            else:
                raise ValueError(
                    f"Unsupported split '{split}' for MIST layout. Use 'train', 'val', or 'test'."
                )
            print(f"Using MIST TrainValAB layout. split={split}")
        else:
            # Classic layout: root_dir/split/HE and root_dir/split/IHC
            self.he_dir = self.root_dir / split / he_subdir
            self.ihc_dir = self.root_dir / split / ihc_subdir
            print(f"Using classic layout. split={split}, he_dir={self.he_dir}, ihc_dir={self.ihc_dir}")

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
        Build list of paired images.
        Assumes corresponding filenames between H&E and IHC directories.
        """
        he_files = sorted(self.he_dir.glob("*.png")) + sorted(self.he_dir.glob("*.jpg")) + sorted(
            self.he_dir.glob("*.jpeg")
        ) + sorted(self.he_dir.glob("*.tif")) + sorted(self.he_dir.glob("*.tiff"))

        data_dicts: List[Dict[str, str]] = []

        for he_file in he_files:
            stem = he_file.stem

            # Try matching by same filename with common extensions
            candidate_exts = [he_file.suffix] + [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
            ihc_file = None
            for ext in candidate_exts:
                candidate = self.ihc_dir / f"{stem}{ext}"
                if candidate.exists():
                    ihc_file = candidate
                    break

            if ihc_file is not None:
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
        Create MONAI transform pipeline.
        """
        keys = ["he", "ihc"]

        # Base transforms (always applied)
        base_transforms = [
            LoadImaged(keys=keys, image_only=True),
            EnsureChannelFirstd(keys=keys),
            ScaleIntensityd(keys=keys, minv=-1.0, maxv=1.0),  # scale to [-1, 1]
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
    Create DataLoader for H&E to IHC dataset.

    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    data_cfg = config["data"]

    dataset = HEtoIHCDataset(
        root_dir=data_cfg["root_dir"],
        split=split,
        he_subdir=data_cfg.get("he_subdir", "HE"),     # ignored for MIST layout
        ihc_subdir=data_cfg.get("ihc_subdir", "IHC"),  # ignored for MIST layout
        image_size=data_cfg["image_size"],
        cache_rate=data_cfg["cache_rate"] if split == "train" else 0.0,
        augmentation=(split == "train"),
    )

    # NEW: optionally limit dataset size from config
    if split == "train":
        max_samples = data_cfg.get("max_train_samples")
    else:
        max_samples = data_cfg.get("max_val_samples")

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        if max_samples < len(dataset):
            # random subset, seeded by Trainer.set_seed for reproducibility
            indices = torch.randperm(len(dataset))[:max_samples]
            dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    # Simple test
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
