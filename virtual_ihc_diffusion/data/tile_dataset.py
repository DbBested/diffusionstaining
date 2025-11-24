from pathlib import Path
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

SRC_ROOT = Path("/orcd/home/002/tomli/orcd/scratch/data/virtual_staining_prepared")
DST_ROOT = Path("/orcd/home/002/tomli/orcd/scratch/data/virtual_staining_tiles")

TILE_SIZE = 512
STRIDE = 512

# thresholding to detect non blank tiles
INTENSITY_THRESH = 0.9      # in range 0 to 1
MIN_FG_FRACTION = 0.05      # at least five percent non white pixels

def is_non_blank(pil_img):
    """Return True if the tile has enough non white pixels."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        gray = arr.mean(axis=2)
    else:
        gray = arr
    fg_fraction = (gray < INTENSITY_THRESH).mean()
    return fg_fraction >= MIN_FG_FRACTION

def tile_split(split: str):
    src_unstained = SRC_ROOT / split / "unstained"
    src_stained   = SRC_ROOT / split / "stained"

    dst_unstained = DST_ROOT / split / "unstained"
    dst_stained   = DST_ROOT / split / "stained"

    dst_unstained.mkdir(parents=True, exist_ok=True)
    dst_stained.mkdir(parents=True, exist_ok=True)

    he_files = sorted(src_unstained.glob("*.jpg"))
    print(f"Tiling split {split}, found {len(he_files)} unstained jpg files")

    for he_path in he_files:
        name = he_path.name
        ihc_path = src_stained / name
        if not ihc_path.exists():
            print(f"Skipping {name}, no matching stained file")
            continue

        he_img = Image.open(he_path)
        ihc_img = Image.open(ihc_path)

        if he_img.size != ihc_img.size:
            print(f"Warning size mismatch for {name}: {he_img.size} vs {ihc_img.size}")

        w, h = he_img.size
        kept = 0
        total = 0

        for top in range(0, h - TILE_SIZE + 1, STRIDE):
            for left in range(0, w - TILE_SIZE + 1, STRIDE):
                box = (left, top, left + TILE_SIZE, top + TILE_SIZE)
                he_crop = he_img.crop(box)
                ihc_crop = ihc_img.crop(box)
                total += 1

                if not is_non_blank(he_crop):
                    continue

                out_name = f"{name[:-4]}_{top}_{left}.png"
                he_crop.save(dst_unstained / out_name)
                ihc_crop.save(dst_stained / out_name)
                kept += 1

        print(f"{name}: kept {kept} of {total} tiles")

if __name__ == "__main__":
    for split in ["train", "test"]:
        tile_split(split)
    print("Tiling complete")
