# Kulitan AI Scanner

This project trains an image classifier that predicts which Kulitan symbol a glyph image represents.

It is configured to focus on the glyph region only, so side text/labels are ignored during training and scanning.

## What you get
- Training pipeline with:
  - transfer learning (`ResNet18`)
  - class balancing (`WeightedRandomSampler` + class-weighted loss)
  - data augmentation
  - early stopping
  - test report export
- Scanner CLI for single-image prediction with confidence scores.

## 1) Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Put your dataset here

Expected structure:

```text
data/
  raw/
    na/
      img1.png
      img2.png
    ke/
      img1.png
    do/
      img1.png
    ...
```

Rules:
- Folder name = class label (example: `na`, `ke`, `do`, `ngi`, etc.)
- Include as many examples per label as possible.
- Keep images clear and tightly cropped around the symbol.

## 3) Train

```bash
python train.py --data-dir data/raw --epochs 40 --batch-size 32
```

Outputs are saved in `artifacts/`:
- `best_model.pt`
- `summary.json`
- `test_report.json`

## 4) Scan a new image

```bash
python scan.py --image path/to/new_symbol.png --checkpoint artifacts/best_model.pt --top-k 5
```

Show popup result (example: `po`):

```bash
python scan.py --image path/to/new_symbol.png --checkpoint artifacts/best_model.pt --popup
```

Strict scan mode (recommended):

```bash
python scan.py --image path/to/new_symbol.png --min-confidence 0.70
```

When confidence is too low or no clear symbol is detected, prediction becomes `unknown` instead of forcing a wrong label.

## 5) Live camera scanner

```bash
python live_scanner.py --checkpoint artifacts/best_model.pt --min-confidence 0.70 --show-crop
```

Controls:
- Press `q` to quit.

The live window continuously shows the detected Kulitan label (example: `po`) and confidence.

Example output:

```json
{
  "image": "path/to/new_symbol.png",
  "prediction": "na",
  "confidence": 0.97,
  "top_k": [
    {"label": "na", "confidence": 0.97},
    {"label": "ni", "confidence": 0.02},
    {"label": "nu", "confidence": 0.01}
  ]
}
```

## Accuracy tips (important)
- Keep class counts balanced. If one label has too few images, collect more for it.
- Use at least 40-100 images per class to start.
- Use consistent background and stroke thickness when possible.
- Add hard examples: blurry, rotated, thinner/thicker strokes.
- Keep only one Kulitan symbol per image for best results.
- If accuracy plateaus, try:
  - more epochs (`--epochs 60`)
  - larger image size (`--image-size 256`)
  - unfreezed training (default) and more data.

## Next step with your data
Once you send your full dataset, I can help you:
1. Auto-clean labels and duplicates.
2. Train and tune hyperparameters.
3. Add a confidence threshold so uncertain symbols return `unknown`.
4. Build a small desktop or web UI scanner.
