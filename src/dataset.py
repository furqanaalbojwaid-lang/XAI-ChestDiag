"""PyTorch dataset and pre-processing / augmentation pipeline."""
from __future__ import annotations

import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return img


def minmax(img: np.ndarray) -> np.ndarray:
    """I_norm = (I - I_min) / (I_max - I_min)  -> [0, 1]."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo + 1e-8)


def to_tensor(img01: np.ndarray, imagenet_normalize: bool) -> torch.Tensor:
    x = np.repeat(img01[..., None], 3, axis=2)  # gray -> 3 channels
    if imagenet_normalize:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float()


def preprocess(path: str, size: int, imagenet_normalize: bool = True) -> tuple[torch.Tensor, np.ndarray]:
    """Deterministic evaluation pre-processing. Returns (tensor, resized uint8 image)."""
    img = read_gray(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return to_tensor(minmax(img), imagenet_normalize), img


class Augmenter:
    """Training-only augmentation: rotation, horizontal flip, zoom, CLAHE."""

    def __init__(self, cfg: dict):
        self.rot = cfg["rotation_deg"]
        self.hflip_p = cfg["hflip_p"]
        self.zoom = cfg["zoom"]
        self.clahe_p = cfg["clahe_p"]
        self.clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip_limit"],
                                     tileGridSize=(cfg["clahe_tile_grid"], cfg["clahe_tile_grid"]))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.clahe_p:
            img = self.clahe.apply(img)
        h, w = img.shape
        angle = random.uniform(-self.rot, self.rot)
        scale = random.uniform(1 - self.zoom, 1 + self.zoom)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if random.random() < self.hflip_p:
            img = cv2.flip(img, 1)
        return img


class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_index: dict[str, str], size: int,
                 imagenet_normalize: bool = True, augment: Augmenter | None = None, multilabel: bool = False,
                 hflip: bool = False):
        missing = [im for im in df["image"] if im not in image_index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images listed in the CSV were not found, e.g. {missing[:3]}")
        self.paths = [image_index[im] for im in df["image"]]
        self.labels = df["label"].astype(np.float32).values
        self.size = size
        self.norm = imagenet_normalize
        self.augment = augment
        self.hflip = hflip  # deterministic flip, used for test-time augmentation
        self.y14 = None
        if multilabel:
            from .proposed import multilabel_targets
            self.y14 = multilabel_targets(df["findings"].values)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = read_gray(self.paths[i])
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.augment is not None:
            img = self.augment(img)
        if self.hflip:
            img = cv2.flip(img, 1)
        x = to_tensor(minmax(img), self.norm)
        if self.y14 is not None:
            return x, torch.tensor(self.labels[i]), torch.from_numpy(self.y14[i])
        return x, torch.tensor(self.labels[i])


# ----------------------------------------------------------------------------- fast path
# Pre-load all images once into a uint8 array (threaded decode) and run the
# augmentation on the GPU. Avoids per-epoch PNG decoding when CPUs are slow.
def preload_images(paths, size: int, workers: int = 8, clahe_p: float = 0.0, clahe_cfg: dict | None = None,
                   seed: int = 0) -> np.ndarray:
    """Return N x size x size uint8. If clahe_p > 0, CLAHE is applied to a random
    (seeded) fraction clahe_p of the images at load time."""
    from concurrent.futures import ThreadPoolExecutor
    rng = np.random.default_rng(seed)
    use_clahe = rng.random(len(paths)) < clahe_p
    cfg = clahe_cfg or {"clahe_clip_limit": 2.0, "clahe_tile_grid": 8}

    def load(i):
        img = cv2.resize(read_gray(paths[i]), (size, size), interpolation=cv2.INTER_AREA)
        if use_clahe[i]:
            img = cv2.createCLAHE(clipLimit=cfg["clahe_clip_limit"],
                                  tileGridSize=(cfg["clahe_tile_grid"], cfg["clahe_tile_grid"])).apply(img)
        return img

    out = np.empty((len(paths), size, size), dtype=np.uint8)
    with ThreadPoolExecutor(workers) as ex:
        for i, img in enumerate(ex.map(load, range(len(paths)), chunksize=64)):
            out[i] = img
            if i % 20000 == 0:
                print(f"  preloaded {i}/{len(paths)}", flush=True)
    return out


def gpu_preprocess(xb_uint8, imagenet_normalize: bool = True, augment: dict | None = None, hflip_all: bool = False):
    """xb_uint8: (B,H,W) uint8 tensor on the GPU -> (B,3,H,W) float tensor.
    Same steps as the CPU pipeline: [affine rotation/zoom + h-flip] -> min-max -> ImageNet norm."""
    import math
    import torch.nn.functional as F
    x = xb_uint8.float().unsqueeze(1) / 255.0
    b = x.shape[0]
    if augment is not None:
        ang = (torch.rand(b, device=x.device) * 2 - 1) * math.radians(augment["rotation_deg"])
        sc = 1 + (torch.rand(b, device=x.device) * 2 - 1) * augment["zoom"]
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                             torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        flip = torch.rand(b, device=x.device) < augment["hflip_p"]
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    if hflip_all:
        x = x.flip(-1)
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    x = (x - lo) / (hi - lo + 1e-8)
    x = x.repeat(1, 3, 1, 1)
    if imagenet_normalize:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x
