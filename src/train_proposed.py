"""Train the proposed XAI-ChestDiag model (DenseNet121 + optional CBAM + optional multi-task head)
under either the baseline recipe (identical to the comparison models) or a modern recipe.

Every choice (best epoch, early stopping) is made on the VALIDATION split only.

Usage:
    python -m src.train_proposed --config configs/kaggle.yaml --variant configs/ablation/a2_mt_cbam.yaml --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .common import ensure_dir, get_device, index_images, load_config, set_seed
from .dataset import Augmenter, CXRDataset
from .models import count_parameters
from .proposed import XAIChestNet

DEFAULT_VARIANT = dict(
    name="proposed", attention="none", multitask=False, aux_weight=1.0, image_size=224,
    recipe="baseline",          # "baseline" = Adam + ReduceLROnPlateau + early stop on val loss
    batch_size=64, lr=1e-4, weight_decay=0.0, epochs=50, patience=7,
    warmup_epochs=1, ema_decay=0.999, select="val_loss", dropout=0.3,
    fast_data=True,             # preload images to RAM + GPU augmentation (CLAHE fixed per image)
    init="imagenet",            # or "xrv_chex": chest-X-ray pre-trained DenseNet121 (CheXpert only)
)


class ArrayLoader:
    """Iterates over a pre-loaded uint8 image array; batches are augmented on the GPU."""

    def __init__(self, arr, labels, y14, bs, device, augment=None, norm=True, shuffle=False, seed=0, drop_last=False):
        from .dataset import gpu_preprocess
        self.arr, self.labels, self.y14 = torch.from_numpy(arr), torch.from_numpy(labels), \
            (torch.from_numpy(y14) if y14 is not None else None)
        self.bs, self.device, self.aug, self.norm = bs, device, augment, norm
        self.shuffle, self.drop_last, self.pre = shuffle, drop_last, gpu_preprocess
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        n = len(self.arr)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def __iter__(self):
        n = len(self.arr)
        idx = torch.randperm(n, generator=self.g) if self.shuffle else torch.arange(n)
        for k in range(len(self)):
            j = idx[k * self.bs:(k + 1) * self.bs]
            x = self.pre(self.arr[j].to(self.device, non_blocking=True), self.norm, self.aug)
            out = [x, self.labels[j].to(self.device)]
            if self.y14 is not None:
                out.append(self.y14[j].to(self.device))
            yield out


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    losses, probs, ys = [], [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logit = model(x)
        losses.append(F.binary_cross_entropy_with_logits(logit.float(), y, reduction="sum").item())
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        ys.append(y.cpu().numpy())
    p, y = np.concatenate(probs), np.concatenate(ys)
    return sum(losses) / len(y), float(roc_auc_score(y, p)), float(((p >= 0.5) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--variant", required=True, help="yaml with the variant settings")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    v = {**DEFAULT_VARIANT, **load_config(args.variant)}
    set_seed(args.seed)
    device = get_device()
    amp = device.type == "cuda"
    out_dir = ensure_dir(f"{cfg['output_dir']}/{v['name']}/seed{args.seed}")
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({**v, "seed": args.seed}, f, indent=2)

    split_dir = cfg["data"]["split_dir"]
    tr_df, va_df = pd.read_csv(f"{split_dir}/train.csv"), pd.read_csv(f"{split_dir}/val.csv")
    index = index_images(cfg["data"]["image_root"])
    norm = cfg["preprocessing"]["imagenet_normalize"]
    nw = cfg["training"]["num_workers"]
    if v["fast_data"] and device.type == "cuda":
        from .dataset import preload_images
        from .proposed import multilabel_targets
        aug = cfg["augmentation"]
        t1 = time.time()
        tr_arr = preload_images([index[i] for i in tr_df.image], v["image_size"], max(8, 2 * nw),
                                clahe_p=aug["clahe_p"], clahe_cfg=aug, seed=args.seed)
        va_arr = preload_images([index[i] for i in va_df.image], v["image_size"], max(8, 2 * nw))
        print(f"preloaded {len(tr_arr)} + {len(va_arr)} images in {time.time() - t1:.0f} s", flush=True)
        tr_dl = ArrayLoader(tr_arr, tr_df.label.values.astype(np.float32),
                            multilabel_targets(tr_df.findings.values) if v["multitask"] else None,
                            v["batch_size"], device, augment=aug, norm=norm, shuffle=True, seed=args.seed, drop_last=True)
        va_dl = ArrayLoader(va_arr, va_df.label.values.astype(np.float32), None, v["batch_size"], device, norm=norm)
    else:
        tr_ds = CXRDataset(tr_df, index, v["image_size"], norm, Augmenter(cfg["augmentation"]), multilabel=v["multitask"])
        va_ds = CXRDataset(va_df, index, v["image_size"], norm)
        g = torch.Generator().manual_seed(args.seed)
        tr_dl = DataLoader(tr_ds, batch_size=v["batch_size"], shuffle=True, num_workers=nw, pin_memory=True,
                           drop_last=True, generator=g, persistent_workers=nw > 0)
        va_dl = DataLoader(va_ds, batch_size=v["batch_size"], shuffle=False, num_workers=nw, pin_memory=True)

    model = XAIChestNet(pretrained=True, attention=v["attention"], multitask=v["multitask"],
                        dropout=v["dropout"], init=v["init"]).to(device)
    print(f"{v['name']}: {count_parameters(model) / 1e6:.2f} M params | init={v['init']} | attention={v['attention']} "
          f"multitask={v['multitask']} recipe={v['recipe']} size={v['image_size']} | device={device}")

    epochs = args.max_epochs or v["epochs"]
    steps_per_epoch = len(tr_dl)
    if v["recipe"] == "modern":
        opt = torch.optim.AdamW(model.parameters(), lr=v["lr"], weight_decay=v["weight_decay"])
        total, warm = epochs * steps_per_epoch, v["warmup_epochs"] * steps_per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        ema = EMA(model, v["ema_decay"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=v["lr"], betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
        ema = None
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best, best_epoch, wait, hist = -np.inf, -1, 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tl, n, te = 0.0, 0, time.time()
        for step, batch in enumerate(tr_dl):
            if step == 50:
                print(f"  epoch {epoch}: ~{(time.time() - te) / 50 * len(tr_dl) / 60:.1f} min/epoch (estimated)", flush=True)
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logit, aux = model.forward_all(x)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y)
                if aux is not None:
                    loss = loss + v["aux_weight"] * F.binary_cross_entropy_with_logits(aux.float(), batch[2].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if v["recipe"] == "modern":
                sched.step()
                ema.update(model)
            tl += loss.item() * x.size(0)
            n += x.size(0)
        eval_model = ema.shadow if ema is not None else model
        vl, vauc, vacc = validate(eval_model, va_dl, device, amp)
        if v["recipe"] != "modern":
            sched.step(vl)
        lr = opt.param_groups[0]["lr"]
        hist.append(dict(epoch=epoch, train_loss=tl / n, val_loss=vl, val_auc=vauc, val_acc=vacc, lr=lr))
        pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
        print(f"epoch {epoch:03d} | train {tl / n:.4f} | val loss {vl:.4f} auc {vauc:.4f} acc {vacc:.4f} | lr {lr:.2e}"
              f" | {(time.time() - te) / 60:.1f} min", flush=True)
        score = -vl if v["select"] == "val_loss" else vauc
        if score > best + 1e-6:
            best, best_epoch, wait = score, epoch, 0
            torch.save(eval_model.state_dict(), out_dir / "best.pt")
        else:
            wait += 1
            if wait >= v["patience"]:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    info = dict(model=v["name"], seed=args.seed, parameters=count_parameters(model), best_epoch=best_epoch,
                best_score=best, select=v["select"], epochs_run=len(hist), train_time_sec=time.time() - t0,
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None, torch_version=torch.__version__)
    with open(out_dir / "train_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
