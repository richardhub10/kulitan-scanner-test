from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path
from typing import Dict, List

import requests
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

from .modeling import build_model
from .preprocess import extract_symbol_region


def _make_transform(image_size: int, mean: List[float], std: List[float]):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class KulitanPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        min_confidence: float,
        top_k: int,
        unknown_label: str,
        allow_no_symbol: bool,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = _ensure_checkpoint(checkpoint_path)

        self.ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.idx_to_label = {int(k): v for k, v in self.ckpt["idx_to_label"].items()}
        self.model = build_model(num_classes=len(self.idx_to_label))
        self.model.load_state_dict(self.ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        image_size = int(self.ckpt.get("image_size", 224))
        norm = self.ckpt.get("normalization", {})
        mean = norm.get("mean", [0.485, 0.456, 0.406])
        std = norm.get("std", [0.229, 0.224, 0.225])
        preprocessing = self.ckpt.get("preprocessing", {})

        self.tfm = _make_transform(image_size, mean, std)
        self.isolate_symbol = bool(preprocessing.get("isolate_symbol", True))
        self.min_symbol_area_ratio = float(preprocessing.get("min_symbol_area_ratio", 0.002))
        self.min_confidence = min_confidence
        self.top_k = top_k
        self.unknown_label = unknown_label
        self.allow_no_symbol = allow_no_symbol

    def predict_pil(self, image: Image.Image) -> Dict[str, object]:
        image = image.convert("RGB")

        symbol_info = {
            "symbol_found": True,
            "symbol_area_ratio": 1.0,
            "components": 1,
        }
        if self.isolate_symbol:
            extraction = extract_symbol_region(
                image,
                min_symbol_area_ratio=self.min_symbol_area_ratio,
            )
            image = extraction.image
            symbol_info = {
                "symbol_found": extraction.symbol_found,
                "symbol_area_ratio": extraction.symbol_area_ratio,
                "components": extraction.components,
            }

        if not symbol_info["symbol_found"] and not self.allow_no_symbol:
            return {
                "prediction": self.unknown_label,
                "confidence": 0.0,
                "top_k": [],
                "reason": "No clear Kulitan symbol detected",
                "symbol_info": symbol_info,
            }

        tensor = self.tfm(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, k=min(self.top_k, len(self.idx_to_label)), dim=1)

        results = []
        for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
            results.append({"label": self.idx_to_label[idx], "confidence": float(prob)})

        out = {
            "prediction": results[0]["label"],
            "confidence": results[0]["confidence"],
            "top_k": results,
            "symbol_info": symbol_info,
        }
        if out["confidence"] < self.min_confidence:
            out["prediction"] = self.unknown_label
            out["reason"] = (
                f"Low confidence ({out['confidence']:.3f}) below threshold ({self.min_confidence:.3f})"
            )

        return out


def _decode_data_url_to_image(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise ValueError("Invalid image data URL")
    _, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _ensure_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path

    model_url = os.getenv("MODEL_URL", "").strip()
    if not model_url:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Set MODEL_URL to auto-download it at startup."
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(model_url, timeout=120)
    response.raise_for_status()
    checkpoint_path.write_bytes(response.content)
    return checkpoint_path


def create_app(
    checkpoint: Path,
    min_confidence: float,
    top_k: int,
    unknown_label: str,
    allow_no_symbol: bool,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    predictor = None
    predictor_error = ""
    try:
        predictor = KulitanPredictor(
            checkpoint_path=checkpoint,
            min_confidence=min_confidence,
            top_k=top_k,
            unknown_label=unknown_label,
            allow_no_symbol=allow_no_symbol,
        )
    except Exception as exc:
        predictor_error = str(exc)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/scan")
    def scan_api():
        try:
            if predictor is None:
                return (
                    jsonify(
                        {
                            "error": "Model is not ready. Train first or set MODEL_URL.",
                            "details": predictor_error,
                        }
                    ),
                    503,
                )

            if "file" in request.files and request.files["file"].filename:
                file = request.files["file"]
                image = Image.open(file.stream).convert("RGB")
            else:
                payload = request.get_json(silent=True) or {}
                data_url = payload.get("image_data_url")
                if not data_url:
                    return jsonify({"error": "No image provided"}), 400
                image = _decode_data_url_to_image(data_url)

            output = predictor.predict_pil(image)
            return jsonify(output)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kulitan web scanner")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--unknown-label", type=str, default="unknown")
    parser.add_argument("--allow-no-symbol", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        checkpoint=args.checkpoint,
        min_confidence=args.min_confidence,
        top_k=args.top_k,
        unknown_label=args.unknown_label,
        allow_no_symbol=args.allow_no_symbol,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
