"""
Prepare the E-Staining dataset for training
Organizes images into train/test splits with proper structure
"""

import os
import shutil
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

# Source paths
source_dir = Path("/orcd/home/002/tomli/orcd/scratch/data/unstained_skin/E-Staining DermaRepo/H&E-Staining dataset")
unstained_dir = source_dir / "Un-Stained"
stained_dir = source_dir / "C_Stained"

# Destination paths
dest_dir = Path("/orcd/home/002/tomli/orcd/scratch/data/virtual_staining_prepared")
dest_dir.mkdir(exist_ok=True, parents=True)

# Create directory structure
for split in ["train", "test"]:
    (dest_dir / split / "unstained").mkdir(exist_ok=True, parents=True)
    (dest_dir / split / "stained").mkdir(exist_ok=True, parents=True)

# Get all unstained files
unstained_files = sorted(list(unstained_dir.glob("*.jpg")))
print(f"Found {len(unstained_files)} unstained images")

# Create matching pairs
pairs = []
for unstained_file in unstained_files:
    # Try to find matching stained file
    # Remove "unstained" and "UNSTAINED" from filename
    base_name = unstained_file.stem
    base_name = base_name.replace(" unstained", "").replace(" UNSTAINED", "")

    # Try different variations
    stained_candidates = [
        stained_dir / f"{base_name}.jpg",
        stained_dir / f"{base_name}.20X.jpg",
        stained_dir / f"{base_name}.10X.jpg",
        stained_dir / unstained_file.name,
    ]

    stained_file = None
    for candidate in stained_candidates:
        if candidate.exists():
            stained_file = candidate
            break

    if stained_file:
        pairs.append((unstained_file, stained_file))
    else:
        print(f"Warning: No matching stained file for {unstained_file.name}")

print(f"Found {len(pairs)} matching pairs")

# Shuffle and split
random.shuffle(pairs)
train_size = int(0.8 * len(pairs))
train_pairs = pairs[:train_size]
test_pairs = pairs[train_size:]

print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

# Copy files
def copy_pairs(pairs, split):
    for idx, (unstained, stained) in enumerate(pairs):
        # Use simple numbered names for consistency
        dest_unstained = dest_dir / split / "unstained" / f"{idx:04d}.jpg"
        dest_stained = dest_dir / split / "stained" / f"{idx:04d}.jpg"

        shutil.copy2(unstained, dest_unstained)
        shutil.copy2(stained, dest_stained)

    print(f"Copied {len(pairs)} pairs to {split}")

copy_pairs(train_pairs, "train")
copy_pairs(test_pairs, "test")

print(f"\nDataset prepared at: {dest_dir}")
print(f"Train: {len(train_pairs)} pairs")
print(f"Test: {len(test_pairs)} pairs")
