from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import timm


@dataclass
class ModelConfig:
    """Configuration container for model creation."""

    model_name: str = "convnext_base_in22ft1k"
    num_classes: int = 100
    drop_rate: float = 0.0
    drop_path_rate: float = 0.2
    pretrained: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        relevant = {
            "model_name": data.get("model_name", cls.model_name),
            "num_classes": data.get("num_classes", cls.num_classes),
            "drop_rate": data.get("drop_rate", cls.drop_rate),
            "drop_path_rate": data.get("drop_path_rate", cls.drop_path_rate),
            "pretrained": data.get("pretrained", cls.pretrained),
        }
        return cls(**relevant)


class FlowerClassifier(nn.Module):
    """Flower classifier built on top of a TIMM backbone."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.backbone_name = config.model_name
        self.model = timm.create_model(
            model_name=config.model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(config_dict: Dict[str, Any]) -> FlowerClassifier:
    """Create a FlowerClassifier instance from a config dictionary."""
    config = ModelConfig.from_dict(config_dict)
    return FlowerClassifier(config)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """Load a checkpoint into the given model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device if device is not None else "cpu"
    state = torch.load(checkpoint_path, map_location=map_location)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=strict)
    return model


