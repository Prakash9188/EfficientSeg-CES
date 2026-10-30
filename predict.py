"""Assignment inference entry point. Prints exactly the two required timing/size lines."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from model import TinySegNet


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TinySegNet(width=int(ckpt.get("width", 16)))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, int(ckpt.get("input_size", 128))


def segment_bgr(image_256_bgr, model, input_size):
    """Input: in-memory 256x256 BGR image. Output: uint8 mask with values 0/1."""
    small = cv2.resize(image_256_bgr, (input_size, input_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        low_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(low_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input photo path")
    ap.add_argument("--out", required=True, help="Output mask PNG path")
    ap.add_argument("--model", default="checkpoints/tinysegnet.pt")
    args = ap.parse_args()

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model, input_size = load_model(args.model, device)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    # Warm-up is outside the measured 20-run average.
    _ = segment_bgr(image, model, input_size)
    start = time.process_time()
    result = None
    for _ in range(20):
        result = segment_bgr(image, model, input_size)
    elapsed_ms = (time.process_time() - start) * 1000.0 / 20.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), result * 255)
    size_bytes = os.path.getsize(args.model)
    print(f"T CPU ms: {elapsed_ms:.2f}")
    print(f"S disk bytes: {size_bytes}")


if __name__ == "__main__":
    main()
