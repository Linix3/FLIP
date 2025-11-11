from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class FlowerDataset(Dataset):
    """Dataset that reads image paths and labels from a CSV file."""

    def __init__(
        self,
        csv_file: Path,
        image_root: Path,
        transform: Optional[Callable] = None,
        label_encoder: Optional[Dict[str, int]] = None,
        has_labels: bool = True,
    ) -> None:
        super().__init__()
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.annotations = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.label_encoder = label_encoder or {}
        self.has_labels = has_labels

        if "image_path" in self.annotations.columns:
            self.paths = self.annotations["image_path"].tolist()
        elif "img_name" in self.annotations.columns:
            self.paths = self.annotations["img_name"].tolist()
        else:
            raise ValueError("CSV must contain 'image_path' or 'img_name' column.")

        if has_labels:
            if "label" in self.annotations.columns:
                labels = self.annotations["label"]
            elif "class_id" in self.annotations.columns:
                labels = self.annotations["class_id"]
            else:
                raise ValueError("CSV must contain 'label' or 'class_id' column when has_labels=True.")

            if label_encoder:
                self.labels = [self.label_encoder[label] for label in labels]
            else:
                self.labels = labels.astype(int).tolist()
        else:
            self.labels = [-1] * len(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = Path(self.paths[idx])
        if not path.is_absolute():
            path = self.image_root / path

        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        item = {"image": image, "img_name": path.name}
        if self.has_labels:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_transforms(image_size: int, mean: Optional[Iterable[float]] = None, std: Optional[Iterable[float]] = None) -> Tuple[Callable, Callable, Callable]:
    """Return train/validation/test torchvision transform pipelines."""
    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.05)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.05)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform, test_transform


def save_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[List[float]],
    output_path: Path,
) -> None:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "top5_accuracy": top_k_accuracy_score(y_true, y_prob, k=5, labels=list(range(len(y_prob[0])))),
    }
    output_path.write_text(json.dumps(metrics, indent=2))


@dataclass
class ExponentialMovingAverage:
    beta: float = 0.98
    value: float = 0.0

    def update(self, x: float) -> float:
        self.value = self.beta * self.value + (1.0 - self.beta) * x
        bias_correction = 1.0 - self.beta
        return self.value / max(bias_correction, 1e-8)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def load_class_mapping(class_map_path: Optional[Path]) -> Optional[Dict[str, int]]:
    if class_map_path is None:
        return None
    if not class_map_path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {class_map_path}")
    with open(class_map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping


def inverse_class_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    return {idx: name for name, idx in mapping.items()}


def cosine_scheduler(
    base_lr: float,
    final_lr: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
) -> List[float]:
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    lr_schedule = []
    for step in range(total_steps):
        if step < warmup_steps:
            lr = base_lr * (step + 1) / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * progress))
        lr_schedule.append(lr)
    return lr_schedule


