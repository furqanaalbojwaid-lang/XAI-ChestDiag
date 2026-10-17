"""Optional speed-up: pre-resize all NIH images once (1024x1024 -> 256x256).

Decoding 112k full-resolution PNGs every epoch is usually the bottleneck.
After running this, set `data.image_root` in the config to the output folder.
Bounding-box coordinates stay valid because the pointing game rescales using
the original width/height stored in Data_Entry_2017.csv.

Usage:
    python -m src.resize_cache --src data/nih/images --dst data/nih/images_256 --size 256
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


def _one(args):
    src, dst, size = args
    if Path(dst).exists():
        return
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(dst, cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(dst / p.name), a.size) for p in Path(a.src).rglob("*.png")]
    print(f"{len(jobs)} images -> {dst}")
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(_one, jobs, chunksize=256)):
            if i % 10000 == 0:
                print(i)


if __name__ == "__main__":
    main()
