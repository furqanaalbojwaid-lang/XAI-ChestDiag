"""Step 2 - Train one model on the patient-level training split.

Usage:
    python -m src.train --model densenet121 --config configs/default.yaml --seed 42
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import ensure_dir, get_device, index_images, load_config, set_seed
from .dataset import Augmenter, CXRDataset
from .models import build_model, count_parameters


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    total, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * x.size(0)
            n += x.size(0)
            correct += ((logits.detach() > 0).float() == y).sum().item()
    return total / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="densenet121")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=None, help="override (for quick tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, pc = cfg["training"], cfg["preprocessing"]
    set_seed(args.seed)
    device = get_device()
    amp = bool(tc["amp"]) and device.type == "cuda"
    out_dir = ensure_dir(f"{cfg['output_dir']}/{args.model}/seed{args.seed}")

    split_dir = cfg["data"]["split_dir"]
    train_df = pd.read_csv(f"{split_dir}/train.csv")
    val_df = pd.read_csv(f"{split_dir}/val.csv")
    index = index_images(cfg["data"]["image_root"])

    train_ds = CXRDataset(train_df, index, pc["image_size"], pc["imagenet_normalize"], Augmenter(cfg["augmentation"]))
    val_ds = CXRDataset(val_df, index, pc["image_size"], pc["imagenet_normalize"])
    g = torch.Generator().manual_seed(args.seed)
    kw = dict(batch_size=tc["batch_size"], num_workers=tc["num_workers"], pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=False, generator=g, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)

    model = build_model(args.model, pretrained=tc["pretrained"], dropout=tc["dropout"]).to(device)
    n_params = count_parameters(model)
    print(f"{args.model}: {n_params / 1e6:.2f} M parameters | device={device} | amp={amp}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"], betas=tuple(tc["betas"]),
                                 weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=tc["lr_factor"], patience=tc["lr_patience"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    epochs = args.max_epochs or tc["epochs"]
    best, best_epoch, wait, history = np.inf, -1, 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer, scaler, amp)
        vl, va = run_epoch(model, val_dl, criterion, device, amp=amp)
        scheduler.step(vl)
        lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va, lr=lr))
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"epoch {epoch:03d} | train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f} | lr {lr:.2e}")
        if vl < best - 1e-6:
            best, best_epoch, wait = vl, epoch, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            wait += 1
            if wait >= tc["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    info = dict(model=args.model, seed=args.seed, parameters=n_params, best_epoch=best_epoch,
                best_val_loss=best, epochs_run=len(history), train_time_sec=time.time() - t0,
                device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                torch_version=torch.__version__)
    with open(out_dir / "train_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
