from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SymbolExtractionResult:
    image: Image.Image
    symbol_found: bool
    symbol_area_ratio: float
    bbox: Optional[Tuple[int, int, int, int]]
    components: int


def _to_grayscale_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    return gray / 255.0


def _connected_components(mask: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[Tuple[int, Tuple[int, int, int, int]]] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            min_x = max_x = x
            min_y = max_y = y

            while stack:
                cy, cx = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            components.append((area, (min_x, min_y, max_x, max_y)))

    return components


def extract_symbol_region(
    image: Image.Image,
    min_symbol_area_ratio: float = 0.002,
    padding_ratio: float = 0.20,
) -> SymbolExtractionResult:
    gray = _to_grayscale_array(image)
    h, w = gray.shape

    threshold = np.quantile(gray, 0.40)
    binary = gray <= threshold

    global_foreground = float(binary.mean())
    if global_foreground < min_symbol_area_ratio:
        return SymbolExtractionResult(
            image=image,
            symbol_found=False,
            symbol_area_ratio=global_foreground,
            bbox=None,
            components=0,
        )

    components = _connected_components(binary)
    min_pixels = max(12, int(0.0005 * h * w))
    candidates = [(area, bbox) for area, bbox in components if area >= min_pixels]

    if not candidates:
        return SymbolExtractionResult(
            image=image,
            symbol_found=False,
            symbol_area_ratio=global_foreground,
            bbox=None,
            components=0,
        )

    def score_component(item: Tuple[int, Tuple[int, int, int, int]]) -> float:
        area, bbox = item
        min_x, _, max_x, _ = bbox
        cx = (min_x + max_x) / 2.0
        left_bias = 1.0 - (cx / max(1.0, (w - 1)))
        return area * (1.0 + 0.45 * left_bias)

    best_area, (min_x, min_y, max_x, max_y) = max(candidates, key=score_component)

    pad_x = int((max_x - min_x + 1) * padding_ratio)
    pad_y = int((max_y - min_y + 1) * padding_ratio)

    x0 = max(0, min_x - pad_x)
    y0 = max(0, min_y - pad_y)
    x1 = min(w - 1, max_x + pad_x)
    y1 = min(h - 1, max_y + pad_y)

    cropped = image.crop((x0, y0, x1 + 1, y1 + 1))
    symbol_ratio = best_area / float(h * w)

    return SymbolExtractionResult(
        image=cropped,
        symbol_found=True,
        symbol_area_ratio=symbol_ratio,
        bbox=(x0, y0, x1, y1),
        components=len(candidates),
    )
