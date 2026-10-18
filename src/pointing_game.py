"""Step 6 - Quantitative explainability: Pointing Game (Zhang et al., ECCV 2016).

A "hit" is recorded when the maximum of the Grad-CAM heat-map falls inside a
radiologist bounding box (NIH BBox_List_2017.csv, coordinates in original
pixel space of the 1024x1024 PNGs), optionally with a tolerance margin (default 15 px, as in the
original Pointing Game protocol).

By default ONLY bounding-box images that belong to the held-out TEST patients
are used, so the explanation is evaluated on data the model never saw.
A trivial "image-centre" baseline is reported alongside for reference.

Usage:
    python -m src.pointing_game --model densenet121 --seed 42
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from .common import ensure_dir, get_device, index_images, load_config
from .dataset import preprocess
from .gradcam import GradCAM
from .models import build_model, gradcam_target_layer


def load_bboxes(bbox_csv: str) -> pd.DataFrame:
    b = pd.read_csv(bbox_csv)
    b = b.iloc[:, :6]
    b.columns = ["image", "finding", "x", "y", "w", "h"]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="densenet121")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tolerance", type=float, default=15.0, help="pixels, in original image space")
    ap.add_argument("--include-non-test", action="store_true",
                    help="also use bbox images of train/val patients (NOT recommended: leakage)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pc, tc = cfg["preprocessing"], cfg["training"]
    device = get_device()
    run_dir = f"{cfg['output_dir']}/{args.model}/seed{args.seed}"

    splits = pd.read_csv(f"{cfg['data']['split_dir']}/all_splits.csv")
    boxes = load_bboxes(cfg["data"]["bbox_csv"]).merge(splits[["image", "split"]], on="image", how="left")
    # NIH bounding boxes are given in the pixel space of the released 1024x1024 PNGs
    # (NOT the original scan size stored in Data_Entry_2017.csv).
    ref = float(cfg["data"].get("bbox_image_size", 1024))
    boxes["W"], boxes["H"] = ref, ref
    n_all = boxes.image.nunique()
    if not args.include_non_test:
        boxes = boxes[boxes.split == "test"]
    print(f"Bounding-box images: {n_all} in total, {boxes.image.nunique()} used "
          f"({len(boxes)} boxes).")
    if boxes.empty:
        raise SystemExit("No bounding-box images in the selected split.")

    model = build_model(args.model, pretrained=False, dropout=tc["dropout"]).to(device)
    model.load_state_dict(torch.load(f"{run_dir}/best.pt", map_location=device))
    cam_engine = GradCAM(model, gradcam_target_layer(model, args.model))
    index = index_images(cfg["data"]["image_root"])
    size = pc["image_size"]

    rows = []
    for image, grp in boxes.groupby("image"):
        x, _ = preprocess(index[image], size, pc["imagenet_normalize"])
        cam, prob = cam_engine(x.unsqueeze(0).to(device), target="abnormal")
        cy, cx = np.unravel_index(np.argmax(cam[0]), cam[0].shape)
        W, H = float(grp.W.iloc[0]), float(grp.H.iloc[0])
        px, py = (cx + 0.5) * W / size, (cy + 0.5) * H / size        # back to original pixels
        for _, b in grp.iterrows():
            t = args.tolerance
            hit = (b.x - t <= px <= b.x + b.w + t) and (b.y - t <= py <= b.y + b.h + t)
            centre_hit = (b.x - t <= W / 2 <= b.x + b.w + t) and (b.y - t <= H / 2 <= b.y + b.h + t)
            rows.append(dict(image=image, finding=b.finding, split=b.split, prob=float(prob[0]),
                             max_x=px, max_y=py, hit=int(hit), centre_baseline_hit=int(centre_hit)))
    cam_engine.remove()

    df = pd.DataFrame(rows)
    df.to_csv(f"{run_dir}/pointing_game_details.csv", index=False)
    per_class = df.groupby("finding").agg(boxes=("hit", "size"), hits=("hit", "sum"),
                                          accuracy=("hit", "mean"),
                                          centre_baseline=("centre_baseline_hit", "mean")).reset_index()
    per_class.to_csv(f"{run_dir}/pointing_game_per_class.csv", index=False)
    hits, n = int(df.hit.sum()), len(df)
    res = dict(model=args.model, seed=args.seed, tolerance_px=args.tolerance,
               images=int(df.image.nunique()), boxes=n, hits=hits, misses=n - hits,
               pointing_accuracy=hits / n, centre_baseline_accuracy=float(df.centre_baseline_hit.mean()),
               test_split_only=not args.include_non_test)
    with open(f"{run_dir}/pointing_game.json", "w") as f:
        json.dump(res, f, indent=2)
    print(per_class.to_string(index=False))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
