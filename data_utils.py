"""DUTS image/mask pairing, resizing, and captured-photo helpers."""
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resize_image_256(image):
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)


def resize_mask_256(mask):
    return cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)


def _index_images(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return {p.stem.lower(): p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def discover_pairs(root):
    """
    Supports common DUTS layouts:
      DUTS-TR-Image/ + DUTS-TR-Mask/
      DUTS-TR/DUTS-TR-Image/ + DUTS-TR/DUTS-TR-Mask/
      root/images/ + root/masks/
    Pairing is by filename stem.
    """
    root = Path(root)
    image_dirs = ["DUTS-TR-Image", "images", "image", "train/images"]
    mask_dirs = ["DUTS-TR-Mask", "masks", "mask", "train/masks"]
    images = masks = None
    for rel in image_dirs:
        candidate = root / rel
        if candidate.exists():
            images = _index_images(candidate)
            break
    for rel in mask_dirs:
        candidate = root / rel
        if candidate.exists():
            masks = _index_images(candidate)
            break
    if images is None or masks is None:
        # Search recursively for folders whose names indicate image/mask.
        dirs = [p for p in root.rglob("*") if p.is_dir()]
        for d in dirs:
            low = d.name.lower()
            if images is None and ("image" in low) and ("mask" not in low):
                idx = _index_images(d)
                if len(idx) > (len(images or {}) if images else 0):
                    images = idx
            if masks is None and "mask" in low:
                idx = _index_images(d)
                if len(idx) > (len(masks or {}) if masks else 0):
                    masks = idx
    if not images or not masks:
        raise RuntimeError(
            f"Could not find image and mask folders under {root}. "
            "Extract DUTS-TR and pass the directory containing DUTS-TR-Image and DUTS-TR-Mask."
        )
    pairs = [(images[k], masks[k]) for k in sorted(images.keys() & masks.keys())]
    if not pairs:
        raise RuntimeError("No image/mask filename matches found. Check your extracted DUTS folder.")
    return pairs


class SegmentationDataset(Dataset):
    def __init__(self, pairs, train_size=128, augment=False):
        self.pairs = pairs
        self.train_size = int(train_size)
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        image = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Unable to read pair: {ip} / {mp}")
        image = resize_image_256(image)
        mask = resize_mask_256(mask)
        # Training at 128x128 reduces CPU inference cost; final metrics use 256x256.
        image = cv2.resize(image, (self.train_size, self.train_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.train_size, self.train_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask >= 128).astype(np.int64)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Mild, deterministic-safe horizontal flip augmentation.
        if self.augment and np.random.random() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask = torch.from_numpy(mask.copy())
        return image, mask
