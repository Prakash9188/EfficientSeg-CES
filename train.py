"""Train TinySegNet from scratch on DUTS-TR. Example: python train.py --data data/DUTS-TR"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import TinySegNet
from data_utils import discover_pairs, SegmentationDataset
from metrics import confusion_iou, boundary_f1


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses, ious, bf1s = [], [], []
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(float(ce(logits, y).item()))
        probs = torch.argmax(logits, dim=1).cpu().numpy()
        truth = y.cpu().numpy()
        for p, t in zip(probs, truth):
            ious.append(confusion_iou(p, t))
            bf1s.append(boundary_f1(p, t))
    return (float(np.mean(losses)), float(np.mean(ious)), float(np.mean(bf1s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="DUTS-TR directory containing image and mask folders")
    ap.add_argument("--out", default="checkpoints/tinysegnet.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7360)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    torch.set_num_threads(1)
    seed_everything(args.seed)
    pairs = discover_pairs(args.data)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(round(len(pairs) * args.val_fraction)))
    val_ids, train_ids = order[:n_val], order[n_val:]
    train_pairs = [pairs[i] for i in train_ids]
    val_pairs = [pairs[i] for i in val_ids]
    train_ds = SegmentationDataset(train_pairs, args.size, augment=True)
    val_ds = SegmentationDataset(val_pairs, args.size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySegNet(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    best_val = float("inf")
    stale = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; pairs: {len(pairs)}; train: {len(train_pairs)}; validation: {len(val_pairs)}")
    print(f"Architecture: TinySegNet(width={args.width}); training resolution: {args.size}x{args.size}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            ce_loss = ce(logits, y)
            # Soft Dice loss on foreground probabilities.
            probs = torch.softmax(logits, dim=1)[:, 1]
            target = (y == 1).float()
            dims = (1, 2)
            intersection = (probs * target).sum(dims)
            dice = (2 * intersection + 1.0) / (probs.sum(dims) + target.sum(dims) + 1.0)
            loss = ce_loss + (1.0 - dice.mean())
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, val_miou, val_bf1 = validate(model, val_loader, device)
        print(f"Epoch {epoch:03d}/{args.epochs}: train_loss={np.mean(train_losses):.4f} "
              f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} val_BF1={val_bf1:.4f}")
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save({"model_state": model.cpu().state_dict(),
                        "width": args.width, "input_size": args.size,
                        "seed": args.seed, "best_val_loss": best_val,
                        "train_count": len(train_pairs), "val_count": len(val_pairs)},
                       out)
            model.to(device)
            print(f"  Saved best checkpoint: {out}")
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping: validation loss did not improve.")
                break
    print("Training complete. DUTS-TE was not used.")


if __name__ == "__main__":
    main()
