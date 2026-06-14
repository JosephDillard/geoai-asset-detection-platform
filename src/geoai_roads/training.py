from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml


@dataclass(frozen=True)
class BuildingTrainingConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def chips_dir(self) -> Path:
        return self._path("training", "chips_dir")

    @property
    def base_model_path(self) -> Path:
        return self._path("model", "base_path")

    @property
    def output_model_path(self) -> Path:
        return self._path("model", "output_path")

    @property
    def architecture(self) -> str:
        return str(self.raw["model"].get("architecture", "unetplusplus"))

    @property
    def encoder_name(self) -> str:
        return str(self.raw["model"].get("encoder_name", "efficientnet-b4"))

    @property
    def num_channels(self) -> int:
        return int(self.raw["model"].get("num_channels", 3))

    @property
    def num_classes(self) -> int:
        return int(self.raw["model"].get("num_classes", 2))

    @property
    def input_size(self) -> int:
        return int(self.raw["model"].get("input_size", 512))

    @property
    def mean(self) -> list[float]:
        return list(self.raw["model"].get("mean", [0.0, 0.0, 0.0]))

    @property
    def std(self) -> list[float]:
        return list(self.raw["model"].get("std", [1.0, 1.0, 1.0]))

    @property
    def epochs(self) -> int:
        return int(self.raw["training"].get("epochs", 20))

    @property
    def batch_size(self) -> int:
        return int(self.raw["training"].get("batch_size", 2))

    @property
    def learning_rate(self) -> float:
        return float(self.raw["training"].get("learning_rate", 1e-4))

    @property
    def weight_decay(self) -> float:
        return float(self.raw["training"].get("weight_decay", 1e-5))

    @property
    def num_workers(self) -> int:
        return int(self.raw["training"].get("num_workers", 0))

    @property
    def dice_weight(self) -> float:
        return float(self.raw["training"].get("dice_weight", 0.5))

    @property
    def ce_weight(self) -> float:
        return float(self.raw["training"].get("ce_weight", 0.5))

    @property
    def seed(self) -> int:
        return int(self.raw["training"].get("seed", 13))

    @property
    def device(self) -> str:
        return str(self.raw["training"].get("device", "auto"))

    @property
    def metrics_path(self) -> Path:
        value = self.raw["training"].get("metrics_path")
        if value:
            return self._resolve_path(value)
        return self.output_model_path.with_suffix(".metrics.json")

    def _path(self, section: str, key: str) -> Path:
        return self._resolve_path(self.raw[section][key])

    def _resolve_path(self, value: Any) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return (self.path.parent.parent / path).resolve()


def load_building_training_config(path: str | Path) -> BuildingTrainingConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return BuildingTrainingConfig(raw=raw, path=config_path)


def train_building_model(config: BuildingTrainingConfig) -> dict[str, Any]:
    torch, smp = _load_training_dependencies()
    _set_seed(torch, config.seed)
    device = _select_device(torch, config.device)

    train_dataset = BuildingChipDataset(config, split="train")
    val_dataset = BuildingChipDataset(config, split="val")
    if len(train_dataset) == 0:
        raise ValueError(f"No training chips found under {config.chips_dir / 'train'}")
    if len(val_dataset) == 0:
        raise ValueError(f"No validation chips found under {config.chips_dir / 'val'}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = _load_model(torch, smp, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = []
    best_iou = -1.0
    best_metrics: dict[str, Any] | None = None
    config.output_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(torch, model, train_loader, optimizer, device, config)
        metrics = _evaluate(torch, model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        history.append(metrics)

        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_metrics = metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": config.architecture,
                    "encoder_name": config.encoder_name,
                    "num_channels": config.num_channels,
                    "num_classes": config.num_classes,
                    "input_size": config.input_size,
                    "mean": config.mean,
                    "std": config.std,
                    "best_metrics": best_metrics,
                },
                config.output_model_path,
            )

    result = {
        "status": "succeeded",
        "best_model": str(config.output_model_path),
        "best_metrics": best_metrics,
        "history": history,
        "train_chips": len(train_dataset),
        "validation_chips": len(val_dataset),
        "device": str(device),
    }
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class BuildingChipDataset:
    def __init__(self, config: BuildingTrainingConfig, split: str) -> None:
        self.config = config
        self.split = split
        self.image_dir = config.chips_dir / split / "images"
        self.mask_dir = config.chips_dir / split / "masks"
        self.image_paths = sorted(self.image_dir.glob("*.tif"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        torch, _ = _load_training_dependencies()
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / f"{image_path.stem}_mask.tif"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask chip for {image_path.name}: {mask_path}")

        with rasterio.open(image_path) as image_dataset:
            image = image_dataset.read()
        with rasterio.open(mask_path) as mask_dataset:
            mask = mask_dataset.read(1)

        image = _to_float_rgb(image)
        mean = np.asarray(self.config.mean, dtype="float32")[:, None, None]
        std = np.asarray(self.config.std, dtype="float32")[:, None, None]
        image = (image - mean) / std

        image_tensor = torch.from_numpy(image.astype("float32"))
        mask_tensor = torch.from_numpy(mask.astype("int64"))
        image_tensor = _resize_image_tensor(torch, image_tensor, self.config.input_size)
        mask_tensor = _resize_mask_tensor(torch, mask_tensor, self.config.input_size)
        return image_tensor, mask_tensor


def _train_epoch(torch: Any, model: Any, loader: Any, optimizer: Any, device: Any, config: BuildingTrainingConfig) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = _combined_loss(torch, logits, masks, config.ce_weight, config.dice_weight)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1)


def _evaluate(torch: Any, model: Any, loader: Any, device: Any) -> dict[str, float]:
    model.eval()
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            predictions = torch.argmax(model(images), dim=1)
            true_positive += float(((predictions == 1) & (masks == 1)).sum().cpu())
            false_positive += float(((predictions == 1) & (masks == 0)).sum().cpu())
            false_negative += float(((predictions == 0) & (masks == 1)).sum().cpu())

    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1.0)
    dice = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1.0)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def _combined_loss(torch: Any, logits: Any, masks: Any, ce_weight: float, dice_weight: float):
    ce_loss = torch.nn.functional.cross_entropy(logits, masks)
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    targets = (masks == 1).float()
    intersection = (probabilities * targets).sum(dim=(1, 2))
    denominator = probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice_loss = 1 - ((2 * intersection + 1.0) / (denominator + 1.0)).mean()
    return ce_loss * ce_weight + dice_loss * dice_weight


def _load_model(torch: Any, smp: Any, config: BuildingTrainingConfig, device: Any):
    model = smp.create_model(
        arch=config.architecture,
        encoder_name=config.encoder_name,
        encoder_weights=None,
        in_channels=config.num_channels,
        classes=config.num_classes,
    )
    checkpoint = torch.load(config.base_model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def _load_training_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import segmentation_models_pytorch as smp
    except Exception as exc:
        raise RuntimeError(
            "Training requires PyTorch and segmentation-models-pytorch. "
            'Install with `python -m pip install -e ".[pytorch]"` or rebuild the GeoAI Docker image.'
        ) from exc
    return torch, smp


def _set_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(torch: Any, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _to_float_rgb(image: np.ndarray) -> np.ndarray:
    image = image.astype("float32")
    max_value = float(np.nanmax(image)) if image.size else 0
    if max_value > 1.0:
        image /= 255.0 if max_value <= 255 else max_value
    return image


def _resize_image_tensor(torch: Any, image_tensor: Any, input_size: int):
    if image_tensor.shape[-2:] == (input_size, input_size):
        return image_tensor
    resized = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0),
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def _resize_mask_tensor(torch: Any, mask_tensor: Any, input_size: int):
    if mask_tensor.shape[-2:] == (input_size, input_size):
        return mask_tensor
    resized = torch.nn.functional.interpolate(
        mask_tensor.float().unsqueeze(0).unsqueeze(0),
        size=(input_size, input_size),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).long()
