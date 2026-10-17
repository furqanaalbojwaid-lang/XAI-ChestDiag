"""Step 4 - Measure inference throughput (images / second) of each architecture.

Model-only timing on random 224x224 tensors (no disk I/O), with warm-up and
CUDA synchronisation. Report the batch size together with the number.

Usage:
    python -m src.throughput --batch-sizes 1 64
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .common import ensure_dir, get_device, load_config
from .models import build_model, count_parameters


@torch.no_grad()
def measure(model, device, batch, size=224, warmup=20, iters=100, amp=False):
    model.eval()
    x = torch.randn(batch, 3, size, size, device=device)
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, enabled=amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast(device_type=device.type, enabled=amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch * iters / dt, dt / iters * 1000 / batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--amp", action="store_true", help="use mixed precision (FP16)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    res = []
    for name in cfg["models"]:
        model = build_model(name, pretrained=False).to(device)
        for b in args.batch_sizes:
            ips, ms = measure(model, device, b, cfg["preprocessing"]["image_size"], iters=args.iters,
                              amp=args.amp and device.type == "cuda")
            row = dict(model=name, parameters_M=round(count_parameters(model) / 1e6, 2), batch_size=b,
                       images_per_sec=round(ips, 1), ms_per_image=round(ms, 3), device=str(device),
                       gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None, amp=args.amp)
            print(row)
            res.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = ensure_dir(f"{cfg['output_dir']}/summary")
    with open(out / "throughput.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
