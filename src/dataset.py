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
                 imagenet_normalize: bool = True, augment: Augmenter | None = None):
        missing = [im for im in df["image"] if im not in image_index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images listed in the CSV were not found, e.g. {missing[:3]}")
        self.paths = [image_index[im] for im in df["image"]]
        self.labels = df["label"].astype(np.float32).values
        self.size = size
        self.norm = imagenet_normalize
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = read_gray(self.paths[i])
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.augment is not None:
            img = self.augment(img)
        x = to_tensor(minmax(img), self.norm)
        return x, torch.tensor(self.labels[i])
