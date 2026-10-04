from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .preprocess import extract_symbol_region


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def discover_samples(data_root: Path) -> List[Sample]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data folder not found: {data_root}")

    samples: List[Sample] = []
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for file_path in class_dir.rglob("*"):
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS and file_path.is_file():
                samples.append(Sample(path=file_path, label=label))

    if not samples:
        raise ValueError(
            f"No image samples found in {data_root}. Expected structure: data/raw/<label>/*.png"
        )
    return samples


def build_label_map(samples: Sequence[Sample]) -> Dict[str, int]:
    labels = sorted({sample.label for sample in samples})
    return {label: idx for idx, label in enumerate(labels)}


def stratified_split(
    samples: Sequence[Sample],
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not 0.0 <= val_size < 1.0:
        raise ValueError("val_size must be in [0.0, 1.0)")
    if not 0.0 <= test_size < 1.0:
        raise ValueError("test_size must be in [0.0, 1.0)")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    labels = [s.label for s in samples]
    train_samples, temp_samples = train_test_split(
        list(samples),
        test_size=val_size + test_size,
        random_state=seed,
        stratify=labels,
    )

    if test_size == 0.0:
        return train_samples, temp_samples, []

    temp_labels = [s.label for s in temp_samples]
    relative_test_size = test_size / (val_size + test_size)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_labels,
    )
    return train_samples, val_samples, test_samples


class KulitanDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        label_to_idx: Dict[str, int],
        transform=None,
        isolate_symbol: bool = True,
        min_symbol_area_ratio: float = 0.002,
    ) -> None:
        self.samples = list(samples)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.isolate_symbol = isolate_symbol
        self.min_symbol_area_ratio = min_symbol_area_ratio

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.isolate_symbol:
                extraction = extract_symbol_region(
                    image,
                    min_symbol_area_ratio=self.min_symbol_area_ratio,
                )
                image = extraction.image
            if self.transform is not None:
                image = self.transform(image)
        return image, self.label_to_idx[sample.label]
