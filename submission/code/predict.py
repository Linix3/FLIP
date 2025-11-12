from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.model import build_model, load_checkpoint
from model.utils import build_transforms, inverse_class_mapping, load_class_mapping


class InferenceDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with path.open("rb") as f:
            with Image.open(f) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)
        return {"image": tensor, "img_name": path.name}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions for flower dataset.")
    parser.add_argument("--config", type=Path, default=Path("model/config.json"), help="Path to config file.")
    parser.add_argument("--checkpoint", type=Path, default=Path("model/best_model.pth"), help="Trained checkpoint path.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images/test"), help="Directory containing test images.")
    parser.add_argument("--output", type=Path, default=Path("results/submission.csv"), help="Output CSV file path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    return parser.parse_args()


def collect_image_paths(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for inference; no GPU detected.")

    device = torch.device("cuda")
    _, _, test_transform = build_transforms(config.get("image_size", 600))

    image_paths = collect_image_paths(args.image_dir)
    dataset = InferenceDataset(image_paths, transform=test_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=config.get("num_workers", 4), pin_memory=True)

    config = {**config, "pretrained": False}
    model = build_model(config).to(device)
    load_checkpoint(model, args.checkpoint, device=device, strict=False)
    model.eval()

    class_map_path: Optional[Path] = None
    if config.get("class_map"):
        candidate = Path(config["class_map"])
        if candidate.exists():
            class_map_path = candidate

    default_map_path = Path("model/label_to_index.json")
    if class_map_path is None and default_map_path.exists():
        class_map_path = default_map_path

    label_to_id = load_class_mapping(class_map_path) if class_map_path else None
    id_to_label = inverse_class_mapping(label_to_id) if label_to_id else None

    img_names = []
    predictions = []
    confidences = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        img_names.extend(batch["img_name"])
        confidences.extend(conf.cpu().tolist())

        if id_to_label:
            for idx_val in pred.cpu().tolist():
                label_value = id_to_label[int(idx_val)]
                try:
                    predictions.append(int(label_value))
                except (ValueError, TypeError):
                    predictions.append(label_value)
        else:
            predictions.extend(pred.cpu().tolist())

    df = pd.DataFrame(
        {
            "img_name": img_names,
            "predicted_class": predictions,
            "confidence": confidences,
        }
    )
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()


