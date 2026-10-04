from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from .dataset import KulitanDataset, Sample, build_label_map, discover_samples, stratified_split
from .modeling import build_loss, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kulitan scanner model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Root dataset folder")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Output folder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only classification head")
    parser.add_argument(
        "--disable-symbol-isolation",
        action="store_true",
        help="Disable glyph extraction and train on full image",
    )
    parser.add_argument(
        "--min-symbol-area-ratio",
        type=float,
        default=0.002,
        help="Reject near-empty symbol detections below this foreground ratio",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_sampler(samples: List[Sample], label_to_idx: Dict[str, int]) -> WeightedRandomSampler:
    counts = Counter(s.label for s in samples)
    class_weights = {label: 1.0 / count for label, count in counts.items()}
    sample_weights = [class_weights[s.label] for s in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def compute_class_weights(samples: List[Sample], label_to_idx: Dict[str, int], device: torch.device) -> torch.Tensor:
    counts = Counter(s.label for s in samples)
    weights = np.zeros(len(label_to_idx), dtype=np.float32)
    for label, idx in label_to_idx.items():
        weights[idx] = 1.0 / counts[label]
    weights /= weights.sum()
    weights *= len(label_to_idx)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    running_loss = 0.0
    correct = 0
    total = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    iterator = tqdm(loader, leave=False)
    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, targets)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        correct += (preds == targets).sum().item()
        total += images.size(0)

        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc, all_targets, all_preds


def save_checkpoint(
    artifacts_dir: Path,
    model: nn.Module,
    label_to_idx: Dict[str, int],
    image_size: int,
    best_val_acc: float,
    isolate_symbol: bool,
    min_symbol_area_ratio: float,
) -> None:
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    ckpt = {
        "model_state": model.state_dict(),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "image_size": image_size,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "preprocessing": {
            "isolate_symbol": isolate_symbol,
            "min_symbol_area_ratio": min_symbol_area_ratio,
        },
        "best_val_acc": best_val_acc,
        "backbone": "resnet18",
    }
    torch.save(ckpt, artifacts_dir / "best_model.pt")


def export_report(
    artifacts_dir: Path,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    idx_to_label: Dict[int, str],
) -> None:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        with (artifacts_dir / "test_report.json").open("w", encoding="utf-8") as fp:
            json.dump({"message": "Test split is empty; no report generated."}, fp, indent=2)
        return

    labels = list(sorted(idx_to_label))
    target_names = [idx_to_label[idx] for idx in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)

    with (artifacts_dir / "test_report.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = discover_samples(args.data_dir)
    label_to_idx = build_label_map(samples)

    class_counts = Counter(s.label for s in samples)
    min_count = min(class_counts.values()) if class_counts else 0
    tiny_dataset_mode = min_count < 3
    val_size = args.val_size
    test_size = args.test_size
    if tiny_dataset_mode:
        # Prevent stratified split errors when many classes have too few examples.
        val_size = 0.0
        test_size = 0.0
        print("Tiny-dataset mode: using all data for training (no val/test split)")

    train_samples, val_samples, test_samples = stratified_split(
        samples,
        val_size=val_size,
        test_size=test_size,
        seed=args.seed,
    )

    train_tf, eval_tf = make_transforms(args.image_size)
    isolate_symbol = not args.disable_symbol_isolation

    train_ds = KulitanDataset(
        train_samples,
        label_to_idx,
        transform=train_tf,
        isolate_symbol=isolate_symbol,
        min_symbol_area_ratio=args.min_symbol_area_ratio,
    )
    val_ds = KulitanDataset(
        val_samples,
        label_to_idx,
        transform=eval_tf,
        isolate_symbol=isolate_symbol,
        min_symbol_area_ratio=args.min_symbol_area_ratio,
    )
    test_ds = KulitanDataset(
        test_samples,
        label_to_idx,
        transform=eval_tf,
        isolate_symbol=isolate_symbol,
        min_symbol_area_ratio=args.min_symbol_area_ratio,
    )

    train_sampler = build_sampler(train_samples, label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(label_to_idx), freeze_backbone=args.freeze_backbone).to(device)

    class_weights = compute_class_weights(train_samples, label_to_idx, device)
    criterion = build_loss(class_weights=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_score = -1.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        if len(val_ds) > 0:
            val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, None, device)
            score_for_checkpoint = val_acc
        else:
            val_loss, val_acc = 0.0, 0.0
            score_for_checkpoint = train_acc

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if score_for_checkpoint > best_score:
            best_score = score_for_checkpoint
            if len(val_ds) > 0:
                best_val_acc = val_acc
            else:
                best_val_acc = train_acc
            bad_epochs = 0
            save_checkpoint(
                args.artifacts_dir,
                model,
                label_to_idx,
                args.image_size,
                best_val_acc,
                isolate_symbol=isolate_symbol,
                min_symbol_area_ratio=args.min_symbol_area_ratio,
            )
        else:
            bad_epochs += 1

        if len(val_ds) > 0 and bad_epochs >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    checkpoint_path = args.artifacts_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    _, test_acc, y_true, y_pred = run_epoch(model, test_loader, criterion, None, device)
    idx_to_label = {int(idx): label for idx, label in checkpoint["idx_to_label"].items()}
    export_report(args.artifacts_dir, y_true, y_pred, idx_to_label)

    summary = {
        "num_classes": len(label_to_idx),
        "num_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "device": str(device),
    }
    with (args.artifacts_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("Training complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
