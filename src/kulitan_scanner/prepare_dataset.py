from __future__ import annotations

import argparse
import csv
import re
import shutil
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

DEFAULT_VOCAB = [
    "a", "e", "i", "o", "u",
    "ba", "be", "bi", "bo", "bu",
    "da", "de", "di", "do", "du",
    "ga", "ge", "gi", "go", "gu",
    "ka", "ke", "ki", "ko", "ku",
    "la", "le", "li", "lo", "lu",
    "ma", "me", "mi", "mo", "mu",
    "na", "ne", "ni", "no", "nu",
    "nga", "nge", "ngi", "ngo", "ngu",
    "pa", "pe", "pi", "po", "pu",
    "sa", "se", "si", "so", "su",
    "ta", "te", "ti", "to", "tu",
    "bang", "dang", "gang", "kank", "lang", "mang", "nang", "ngang", "pang", "sang", "tang",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kulitan dataset by OCR-labeling image text")
    parser.add_argument("--source", type=Path, required=True, help="Folder containing raw images")
    parser.add_argument("--output", type=Path, default=Path("data/raw"), help="Output training folder")
    parser.add_argument("--report", type=Path, default=Path("artifacts/dataset_label_report.csv"))
    parser.add_argument("--min-confidence", type=float, default=0.40)
    parser.add_argument("--disable-vocab-correction", action="store_true")
    parser.add_argument("--clean-output", action="store_true", help="Delete output folder before writing")
    return parser.parse_args()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def correct_with_vocab(label: str, vocab: List[str]) -> Optional[str]:
    if not label:
        return None
    if label in vocab:
        return label

    candidates = sorted((levenshtein(label, v), v) for v in vocab)
    best_dist, best = candidates[0]

    # Tight matching for short labels and slightly looser for longer labels.
    max_allowed = 1 if len(label) <= 3 else 2
    if best_dist <= max_allowed:
        return best
    return None


def normalize_label(raw: str) -> str:
    value = raw.strip().lower()
    value = value.replace("|", "i")
    value = value.replace("!", "i")
    value = value.replace("l", "i")

    value = "".join(ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z/]+", "", value)
    value = value.strip("/")

    # Resolve alternate-vowel notations like ku/u -> ku, ti/i -> ti.
    if "/" in value:
        value = value.split("/", 1)[0]

    return value


def extract_right_text(ocr: RapidOCR, image_path: Path) -> Tuple[Optional[str], float, str]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size

        # Most files place label text on the right side.
        crop = img.crop((int(w * 0.45), 0, w, h))

    crop_array = np.array(crop)
    result, _ = ocr(crop_array)
    if not result:
        return None, 0.0, ""

    best_text = ""
    best_conf = 0.0
    for line in result:
        text = str(line[1]).strip()
        conf = float(line[2])
        if conf > best_conf and text:
            best_conf = conf
            best_text = text

    norm = normalize_label(best_text)
    if not norm:
        return None, best_conf, best_text
    return norm, best_conf, best_text


def discover_images(folder: Path) -> List[Path]:
    paths = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(p)
    return sorted(paths)


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    images = discover_images(args.source)
    if not images:
        raise ValueError(f"No images found in {args.source}")

    if args.clean_output and args.output.exists():
        shutil.rmtree(args.output)

    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    unresolved_dir = args.output / "_unresolved"
    unresolved_dir.mkdir(parents=True, exist_ok=True)

    ocr = RapidOCR()

    rows = []
    labeled = 0
    unresolved = 0

    for idx, image_path in enumerate(images, start=1):
        label, conf, raw_text = extract_right_text(ocr, image_path)
        mapped_label = label
        if mapped_label and not args.disable_vocab_correction:
            mapped_label = correct_with_vocab(mapped_label, DEFAULT_VOCAB)

        if mapped_label is not None and conf >= args.min_confidence:
            target_dir = args.output / mapped_label
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / image_path.name
            labeled += 1
        else:
            target_path = unresolved_dir / image_path.name
            unresolved += 1

        shutil.copy2(image_path, target_path)

        rows.append(
            {
                "index": idx,
                "source": str(image_path),
                "target": str(target_path),
                "raw_text": raw_text,
                "label": label or "",
                "mapped_label": mapped_label or "",
                "confidence": f"{conf:.4f}",
                "status": "labeled" if mapped_label and conf >= args.min_confidence else "unresolved",
            }
        )

    with args.report.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["index", "source", "target", "raw_text", "label", "mapped_label", "confidence", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Images processed: {len(images)}")
    print(f"Labeled: {labeled}")
    print(f"Unresolved: {unresolved}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
