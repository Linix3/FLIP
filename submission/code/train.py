from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from model.model import build_model, load_checkpoint
from model.utils import (
    AverageMeter,
    FlowerDataset,
    build_dataloader,
    build_transforms,
    inverse_class_mapping,
    load_class_mapping,
    save_metrics,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flower classification model.")
    parser.add_argument("--config", type=Path, default=Path("model/config.json"), help="Path to config file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model"),
        help="Directory where checkpoints and logs will be saved.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    return parser.parse_args()


def build_criterion(label_smoothing: float) -> nn.Module:
    if label_smoothing > 0:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss()


def get_schedulers(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    epochs: int,
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, int]:
    scheduler_cfg = config.get("scheduler", {"type": "cosine"})
    min_lr = scheduler_cfg.get("min_lr", 1e-6)
    warmup_epochs = scheduler_cfg.get("warmup_epochs", 0)

    main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs or 1, eta_min=min_lr)
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=scheduler_cfg.get("warmup_start_factor", 0.1),
            total_iters=warmup_epochs,
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = main_scheduler
    return scheduler, warmup_epochs


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    accumulation_steps: int,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        with amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean().item()

        loss_meter.update(loss.item() * accumulation_steps, n=images.size(0))
        acc_meter.update(acc, n=images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, list, list, list]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_targets = []
    all_preds = []
    all_probs = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update((preds == targets).float().mean().item(), n=images.size(0))

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return loss_meter.avg, acc_meter.avg, all_targets, all_preds, all_probs


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (Path("results")).mkdir(parents=True, exist_ok=True)

    seed_everything(config.get("seed", 42))

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for training; no GPU detected.")

    device = torch.device("cuda")
    train_transform, val_transform, _ = build_transforms(config.get("image_size", 600))

    class_map_path = config.get("class_map")
    class_map = load_class_mapping(Path(class_map_path)) if class_map_path else None

    train_dataset = FlowerDataset(
        csv_file=Path(config["train_csv"]),
        image_root=Path(config["train_root"]),
        transform=train_transform,
        label_encoder=class_map,
        has_labels=True,
    )

    val_dataset = FlowerDataset(
        csv_file=Path(config["val_csv"]),
        image_root=Path(config["val_root"]),
        transform=val_transform,
        label_encoder=class_map,
        has_labels=True,
    )

    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    grad_accum = config.get("grad_accumulation", 1)

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    model = build_model(config).to(device)
    if args.resume:
        load_checkpoint(model, Path(args.resume), device=device, strict=False)

    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 5e-4),
        weight_decay=config.get("weight_decay", 1e-4),
        betas=config.get("adam_betas", (0.9, 0.999)),
    )

    epochs = config.get("num_epochs", 20)
    scheduler, warmup_epochs = get_schedulers(optimizer, config, epochs)
    criterion = build_criterion(config.get("label_smoothing", 0.1)).to(device)

    scaler = amp.GradScaler(enabled=config.get("amp", True))

    best_acc = 0.0
    best_epoch = 0
    metrics_path = Path("results") / "validation_metrics.json"

    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            accumulation_steps=grad_accum,
            use_amp=config.get("amp", True),
        )
        val_loss, val_acc, val_targets, val_preds, val_probs = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            checkpoint_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "val_accuracy": best_acc,
                    "config": config,
                },
                checkpoint_path,
            )
            if class_map:
                inv_map = inverse_class_mapping(class_map)
                (output_dir / "class_mapping.json").write_text(json.dumps(inv_map, indent=2), encoding="utf-8")

            save_metrics(val_targets, val_preds, val_probs, metrics_path)

    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()


