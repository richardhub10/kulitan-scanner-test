from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms

from .modeling import build_model
from .preprocess import extract_symbol_region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan a Kulitan image")
    parser.add_argument("--image", type=Path, required=True, help="Path to image file")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--unknown-label", type=str, default="unknown")
    parser.add_argument(
        "--allow-no-symbol",
        action="store_true",
        help="Allow prediction even when no valid glyph region is detected",
    )
    parser.add_argument("--popup", action="store_true", help="Show result in a popup window")
    return parser.parse_args()


def _make_transform(image_size: int, mean: List[float], std: List[float]):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def _popup_message(title: str, message: str) -> None:
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
    except Exception:
        # Silent fallback when UI popup is not available.
        return


def predict_single(
    image_path: Path,
    checkpoint_path: Path,
    top_k: int,
    min_confidence: float,
    unknown_label: str,
    allow_no_symbol: bool,
) -> Dict[str, object]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(checkpoint_path, device)

    idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}
    num_classes = len(idx_to_label)
    model = build_model(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    image_size = int(ckpt.get("image_size", 224))
    norm = ckpt.get("normalization", {})
    mean = norm.get("mean", [0.485, 0.456, 0.406])
    std = norm.get("std", [0.229, 0.224, 0.225])
    preprocessing = ckpt.get("preprocessing", {})
    isolate_symbol = bool(preprocessing.get("isolate_symbol", True))
    min_symbol_area_ratio = float(preprocessing.get("min_symbol_area_ratio", 0.002))
    tfm = _make_transform(image_size, mean, std)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        symbol_info = {
            "symbol_found": True,
            "symbol_area_ratio": 1.0,
            "components": 1,
        }
        if isolate_symbol:
            extraction = extract_symbol_region(
                image,
                min_symbol_area_ratio=min_symbol_area_ratio,
            )
            image = extraction.image
            symbol_info = {
                "symbol_found": extraction.symbol_found,
                "symbol_area_ratio": extraction.symbol_area_ratio,
                "components": extraction.components,
            }

        if not symbol_info["symbol_found"] and not allow_no_symbol:
            return {
                "image": str(image_path),
                "prediction": unknown_label,
                "confidence": 0.0,
                "top_k": [],
                "reason": "No clear Kulitan symbol detected",
                "symbol_info": symbol_info,
            }

        tensor = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, num_classes), dim=1)

    results = []
    for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        results.append(
            {
                "label": idx_to_label[idx],
                "confidence": prob,
            }
        )

    output = {
        "image": str(image_path),
        "prediction": results[0]["label"],
        "confidence": results[0]["confidence"],
        "top_k": results,
        "symbol_info": symbol_info,
    }

    if output["confidence"] < min_confidence:
        output["prediction"] = unknown_label
        output["reason"] = (
            f"Low confidence ({output['confidence']:.3f}) below threshold ({min_confidence:.3f})"
        )

    return output


def main() -> None:
    args = parse_args()
    output = predict_single(
        args.image,
        args.checkpoint,
        args.top_k,
        min_confidence=args.min_confidence,
        unknown_label=args.unknown_label,
        allow_no_symbol=args.allow_no_symbol,
    )
    print(json.dumps(output, indent=2))
    if args.popup:
        confidence = output.get("confidence", 0.0)
        _popup_message("Kulitan Scan Result", f"{output['prediction']} ({confidence:.3f})")


if __name__ == "__main__":
    main()
