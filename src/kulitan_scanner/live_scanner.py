from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .modeling import build_model
from .preprocess import extract_symbol_region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Kulitan camera scanner")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--unknown-label", type=str, default="unknown")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--scan-interval", type=int, default=6, help="Predict every N frames")
    parser.add_argument("--show-crop", action="store_true", help="Show extracted symbol crop")
    return parser.parse_args()


def _load_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}
    model = build_model(num_classes=len(idx_to_label))
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

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return model, idx_to_label, tfm, device, isolate_symbol, min_symbol_area_ratio


def _predict(
    pil_image: Image.Image,
    model,
    idx_to_label: Dict[int, str],
    tfm,
    device,
    top_k: int,
    min_confidence: float,
    unknown_label: str,
    isolate_symbol: bool,
    min_symbol_area_ratio: float,
) -> Tuple[str, float, List[Tuple[str, float]], np.ndarray | None, str]:
    symbol_crop_bgr = None
    reason = ""

    if isolate_symbol:
        extraction = extract_symbol_region(
            pil_image,
            min_symbol_area_ratio=min_symbol_area_ratio,
        )
        if not extraction.symbol_found:
            return unknown_label, 0.0, [], symbol_crop_bgr, "No clear Kulitan symbol detected"
        pil_image = extraction.image
        symbol_crop_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    tensor = tfm(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        k = min(top_k, len(idx_to_label))
        top_probs, top_indices = torch.topk(probs, k=k, dim=1)

    ranked: List[Tuple[str, float]] = []
    for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        ranked.append((idx_to_label[idx], float(prob)))

    pred_label, pred_conf = ranked[0]
    if pred_conf < min_confidence:
        reason = f"Low confidence ({pred_conf:.3f} < {min_confidence:.3f})"
        return unknown_label, pred_conf, ranked, symbol_crop_bgr, reason

    return pred_label, pred_conf, ranked, symbol_crop_bgr, reason


def main() -> None:
    args = parse_args()
    (
        model,
        idx_to_label,
        tfm,
        device,
        isolate_symbol,
        min_symbol_area_ratio,
    ) = _load_model(args.checkpoint)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    frame_count = 0
    latest_label = "-"
    latest_conf = 0.0
    latest_reason = ""
    latest_ranked: List[Tuple[str, float]] = []
    latest_crop = None

    print("Live scanner running")
    print("Controls: q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        if frame_count % max(1, args.scan_interval) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            (
                latest_label,
                latest_conf,
                latest_ranked,
                latest_crop,
                latest_reason,
            ) = _predict(
                pil_image,
                model,
                idx_to_label,
                tfm,
                device,
                args.top_k,
                args.min_confidence,
                args.unknown_label,
                isolate_symbol,
                min_symbol_area_ratio,
            )

        color = (0, 220, 0) if latest_label != args.unknown_label else (0, 165, 255)
        cv2.putText(
            frame,
            f"Kulitan: {latest_label}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Confidence: {latest_conf:.3f}",
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

        if latest_reason:
            cv2.putText(
                frame,
                latest_reason,
                (16, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 80, 220),
                2,
                cv2.LINE_AA,
            )

        y0 = 122
        for label, conf in latest_ranked[:3]:
            cv2.putText(
                frame,
                f"{label}: {conf:.3f}",
                (16, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y0 += 26

        cv2.imshow("Kulitan Live Scanner", frame)
        if args.show_crop and latest_crop is not None:
            cv2.imshow("Symbol Crop", latest_crop)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
