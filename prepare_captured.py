"""Resize your captured photos to assignment format. Masks must be drawn separately."""
import argparse
from pathlib import Path
import cv2
from data_utils import IMAGE_EXTS, resize_image_256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder containing your original phone photos")
    ap.add_argument("--out", default="captured")
    args = ap.parse_args()
    root, out = Path(args.input), Path(args.out)
    original = out / "original"
    resized = out / "images_256"
    masks = out / "masks_256"
    for d in (original, resized, masks):
        d.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images found in {root}")
    for p in paths:
        image = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable: {p}")
            continue
        cv2.imwrite(str(original / p.name), image)
        target = resized / (p.stem + ".png")
        cv2.imwrite(str(target), resize_image_256(image))
        print(f"Saved {target}. Draw its mask and save as {masks / (p.stem + '.png')}")
    print("Masks must be white (255) on the object and black (0) on the background.")


if __name__ == "__main__":
    main()
