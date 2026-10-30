# CSL7360 Computer Vision — Assignment 2
## Cost-Efficient Segmentation under a Compute Budget

This repository implements **Track B**, a small fully convolutional network trained from scratch in PyTorch. It uses no pretrained weights, `torchvision`, scikit-learn, scikit-image, SciPy, or third-party segmentation implementation.

## Project structure

```text
CES_Assignment2/
├── model.py                 # Tiny encoder-decoder CNN with skip connections
├── data_utils.py            # DUTS pair discovery, resize rules, dataset
├── train.py                 # Training, validation split, early stopping
├── predict.py               # Required command-line inference and timing
├── metrics.py               # mIoU, Boundary F1, Q and CES formulas
├── evaluate.py              # Labelled-set evaluation
├── prepare_captured.py      # Resize your own phone photos
├── captured/
│   ├── original/
│   ├── images_256/
│   └── masks_256/
├── checkpoints/             # Saved model parameters go here
└── results/                 # Logs, plots and report assets
```

## 1. Environment

Python 3.10+ is recommended. From this directory:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The code sets single-thread environment variables before importing NumPy/OpenCV/PyTorch in the training, evaluation and prediction entry points. `predict.py` also sets OpenCV and PyTorch thread counts to one.

## 2. Get DUTS-TR

Download the official DUTS dataset from http://saliencydetection.net/duts/ and extract the **DUTS-TR image and mask folders**. Common layout:

```text
data/DUTS-TR/
├── DUTS-TR-Image/
└── DUTS-TR-Mask/
```

If the extracted folders are named differently, pass the directory containing the image and mask subfolders to `--data`; `data_utils.py` also supports common `images/` and `masks/` layouts. Images and masks are paired by filename stem. Check that the dataset archive extracted correctly before training.

**Data discipline:** `train.py` creates a deterministic 90%/10% split from DUTS-TR (seed 7360 by default). Use validation results for architecture and hyperparameter choices. Do not use DUTS-TE for training, tuning, or early stopping. Only run final DUTS-TE evaluation after the design is frozen.

## 3. Train

```bash
python train.py --data data/DUTS-TR --epochs 25 --batch-size 32 --size 128
```

Useful options:

- `--epochs`: maximum epochs (default 25)
- `--batch-size`: batch size (default 32)
- `--lr`: Adam learning rate (default 0.001)
- `--size`: internal training/inference resolution (default 128)
- `--width`: base network width (default 16)
- `--val-fraction`: validation fraction (default 0.10)
- `--patience`: early-stopping patience (default 5)
- `--out`: checkpoint path (default `checkpoints/tinysegnet.pt`)

Training uses GPU if PyTorch detects one; inference is deliberately CPU-only. The objective is pixel-wise cross-entropy plus soft foreground Dice loss. Mild horizontal flips are used for training augmentation. The best checkpoint is selected using validation loss.

For a small smoke test before full training, use a small epoch count and a small copied subset of paired images. Do not treat subset results as final performance.

## 4. Predict one photo

```bash
python predict.py --image path/to/photo.jpg --out results/predicted_mask.png
```

To use another checkpoint:

```bash
python predict.py --image path/to/photo.jpg --out results/predicted_mask.png --model checkpoints/tinysegnet.pt
```

The output is a 256x256 PNG with values 0 (background) and 255 (object). The script warms up once, times 20 further runs with `time.process_time()`, and prints the required two lines:

```text
T CPU ms: 12.34
S disk bytes: 425767
```

Those numbers above are **format examples only**, not measured results. Report the values actually printed on your own machine. The assignment's timing rule starts with a 256x256 photo already in memory and includes the inference-time resize, colour conversion, tensor conversion and prediction. File I/O and checkpoint loading are excluded.

## 5. Evaluate a labelled set

The folder must contain corresponding image and mask filenames (same stem). Masks are thresholded at 128 after `INTER_NEAREST` resizing to 256x256.

```bash
python evaluate.py --images path/to/images --masks path/to/masks --model checkpoints/tinysegnet.pt
```

For a quick pipeline check only:

```bash
python evaluate.py --images path/to/images --masks path/to/masks --limit 10
```

Run on the full DUTS-TE only after finalizing all design choices. The reported mIoU follows the assignment's pooled confusion-matrix definition; Boundary F1 is computed per image and averaged. `Q = 0.70*mIoU + 0.30*Boundary_F1`.

To report CES, measure mean CPU time over the required 5,019 DUTS-TE images one at a time on your laptop, using `time.process_time()`, and use the saved parameter file size from `os.path.getsize()`. The score is zero if mean CPU time exceeds 50 ms. `metrics.ces_score()` implements the given formula, but do not substitute a single-image estimate for the required full-set timing.

## 6. Prepare your own 5–10 captured photos

Take 5–10 photos that collectively cover at least three required conditions: indoor/outdoor, daylight/tube light, near/far objects, clutter, and partial occlusion. Keep these photos out of training and validation.

Put originals in a folder, for example `my_phone_photos/`, then run:

```bash
python prepare_captured.py --input my_phone_photos --out captured
```

This creates:
- `captured/original/`: copies of the originals
- `captured/images_256/`: 256x256 PNGs
- `captured/masks_256/`: empty destination for your labels

Use Paint, GIMP, LabelMe or another tool to draw each binary mask on the corresponding resized image. Save the mask as `captured/masks_256/<same-stem>.png`, with object=255 (white) and background=0 (black). Make sure the filenames match exactly.

Then evaluate:

```bash
python evaluate.py --images captured/images_256 --masks captured/masks_256 --model checkpoints/tinysegnet.pt
```

## 7. GitHub upload

From inside the project directory:

```bash
git init
git add .
git commit -m "Implement cost-efficient object segmentation assignment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If `origin` already exists, do not add it again; use `git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git`. Do not commit the full DUTS dataset or large checkpoint unless your course repository requires it. Include the checkpoint separately if submission rules require the parameter file.

## 8. Report checklist

Your report should include:

1. Method and network diagram/description.
2. Exact train/validation counts, split seed, hyperparameters, best epoch and loss.
3. CPU model and OS; single-thread timing procedure.
4. One table with mIoU, Boundary F1, Q and, where measured, TCPU, Sdisk and CES for validation, DUTS-TE and captured photos.
5. True and predicted masks side by side as semi-transparent overlays for every captured photo.
6. At least two failure cases with likely causes (e.g., clutter, lighting, occlusion, small objects).
7. Reproduction commands and the final Git commit hash.

Do not fill in scores, runtime, CPU model, epoch counts or failure observations until you have actually measured them.

## Design notes and limitations

- This is a clear baseline, not a claim of top leaderboard performance. You should tune width, internal resolution, augmentation, learning rate and epoch count using the validation split only.
- A 128x128 internal resolution is chosen to reduce CPU cost. It can lose thin boundaries and small objects; try 160 or 192 only if measured CPU time stays under the limit.
- The script implements per-image boundary matching using a 5x5 neighbourhood filter, not a Python loop over every pixel.
- `evaluate.py` reports quality and model size. The assignment's official mean CPU time must be measured over the full 5,019-image DUTS-TE set on your own CPU. Do not claim that budget is met until you have done so.
