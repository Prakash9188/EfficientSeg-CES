"""Evaluate a checkpoint on a labelled image/mask folder; DUTS-TE only after design is frozen."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from model import TinySegNet
from predict import load_model, segment_bgr
from data_utils import _index_images, resize_image_256, resize_mask_256
from metrics import score_set, ces_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder of evaluation images")
    ap.add_argument("--masks", required=True, help="Folder of ground-truth masks")
    ap.add_argument("--model", default="checkpoints/tinysegnet.pt")
    ap.add_argument("--limit", type=int, default=0, help="Optional small debug limit; 0 means all")
    args = ap.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    image_map, mask_map = _index_images(args.images), _index_images(args.masks)
    keys = sorted(image_map.keys() & mask_map.keys())
    if not keys:
        raise SystemExit("No matching image/mask names found.")
    if args.limit > 0:
        keys = keys[:args.limit]
    model, input_size = load_model(args.model, torch.device("cpu"))
    predictions, truths = [], []
    for i, key in enumerate(keys, 1):
        image = cv2.imread(str(image_map[key]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_map[key]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"Skipping unreadable pair: {key}")
            continue
        image = resize_image_256(image)
        truth = resize_mask_256(mask) >= 128
        pred = segment_bgr(image, model, input_size).astype(bool)
        predictions.append(pred)
        truths.append(truth)
        if i % 100 == 0:
            print(f"Evaluated {i}/{len(keys)}")
    scores = score_set(predictions, truths)
    model_bytes = os.path.getsize(args.model)
    print(f"Images evaluated: {len(predictions)}")
    print(f"mIoU: {scores['mIoU']:.6f}")
    print(f"Boundary F1: {scores['Boundary_F1']:.6f}")
    print(f"Q: {scores['Q']:.6f}")
    print(f"Model bytes: {model_bytes}")
    print("CES requires mean CPU inference time from predict.py/your full-set timing run.")
    # Do not claim CES without a measured mean runtime on the target CPU.


if __name__ == "__main__":
    main()
