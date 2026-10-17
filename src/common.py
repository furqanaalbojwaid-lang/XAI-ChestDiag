"""Shared helpers: configuration, seeding, device and image indexing."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def index_images(image_root: str | os.PathLike) -> dict[str, str]:
    """Map image file name -> full path (searches sub-folders recursively)."""
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Image folder not found: {root}")
    index = {}
    for p in root.rglob("*.png"):
        index[p.name] = str(p)
    if not index:
        raise FileNotFoundError(f"No .png images found under {root}")
    return index


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
