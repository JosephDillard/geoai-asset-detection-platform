from __future__ import annotations

from dataclasses import dataclass
import csv
import random
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box
import yaml

from geoai_roads.tiling import iter_windows


@dataclass(frozen=True)
class TrainingDataConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def imagery_source(self) -> Path:
        return self._path("imagery", "source")

    @property
    def imagery_bands(self) -> list[int]:
        return list(self.raw["imagery"].get("bands", [1, 2, 3]))

    @property
    def label_source(self) -> Path:
        return self._path("labels", "source")

    @property
    def label_layer(self) -> str | None:
        value = self.raw["labels"].get("layer")
        return str(value) if value else None

    @property
    def output_dir(self) -> Path:
        return self._path("training_data", "output_dir")

    @property
    def tile_size(self) -> int:
        return int(self.raw["training_data"].get("tile_size", 256))

    @property
    def overlap(self) -> int:
        return int(self.raw["training_data"].get("overlap", 64))

    @property
    def validation_fraction(self) -> float:
        return float(self.raw["training_data"].get("validation_fraction", 0.2))

    @property
    def include_empty_fraction(self) -> float:
        return float(self.raw["training_data"].get("include_empty_fraction", 0.1))

    @property
    def min_mask_pixels(self) -> int:
        return int(self.raw["training_data"].get("min_mask_pixels", 16))

    @property
    def all_touched(self) -> bool:
        return bool(self.raw["training_data"].get("all_touched", False))

    @property
    def skip_partial_tiles(self) -> bool:
        return bool(self.raw["training_data"].get("skip_partial_tiles", True))

    @property
    def seed(self) -> int:
        return int(self.raw["training_data"].get("seed", 13))

    @property
    def class_name(self) -> str:
        return str(self.raw.get("asset", {}).get("class_name", "building"))

    def _path(self, section: str, key: str) -> Path:
        value = Path(str(self.raw[section][key]))
        if value.is_absolute():
            return value
        return (self.path.parent.parent / value).resolve()


def load_training_data_config(path: str | Path) -> TrainingDataConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return TrainingDataConfig(raw=raw, path=config_path)


def export_training_chips(config: TrainingDataConfig) -> dict[str, int]:
    labels = _read_labels(config)
    rng = random.Random(config.seed)
    summary = {
        "train_chips": 0,
        "validation_chips": 0,
        "positive_chips": 0,
        "empty_chips": 0,
        "skipped_partial_tiles": 0,
    }

    manifest_path = config.output_dir / "manifest.csv"
    for split in ("train", "val"):
        (config.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (config.output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    with rasterio.open(config.imagery_source) as dataset:
        labels = labels.to_crs(dataset.crs)
        labels["geometry"] = labels.geometry.buffer(0)
        labels = labels[~labels.geometry.is_empty & labels.geometry.notnull()].copy()

        windows = iter_windows(dataset.width, dataset.height, config.tile_size, config.overlap)
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=["split", "image", "mask", "positive_pixels", "bounds"],
            )
            writer.writeheader()

            for index, window in enumerate(windows):
                if config.skip_partial_tiles and (
                    int(window.width) != config.tile_size or int(window.height) != config.tile_size
                ):
                    summary["skipped_partial_tiles"] += 1
                    continue

                transform = dataset.window_transform(window)
                tile_bounds = window_bounds(window, dataset.transform)
                tile_polygon = box(*tile_bounds)
                candidates = labels[labels.intersects(tile_polygon)]
                mask = _rasterize_labels(candidates, window, transform, config)
                positive_pixels = int(mask.sum())

                if positive_pixels < config.min_mask_pixels:
                    if rng.random() > config.include_empty_fraction:
                        continue
                    summary["empty_chips"] += 1
                else:
                    summary["positive_chips"] += 1

                split = "val" if rng.random() < config.validation_fraction else "train"
                image_name = f"{config.class_name}_{index:06d}.tif"
                mask_name = f"{config.class_name}_{index:06d}_mask.tif"
                image_path = config.output_dir / split / "images" / image_name
                mask_path = config.output_dir / split / "masks" / mask_name

                image = dataset.read(config.imagery_bands, window=window)
                _write_image_chip(image_path, image, dataset, transform)
                _write_mask_chip(mask_path, mask, dataset.crs, transform)

                summary[f"{'validation' if split == 'val' else 'train'}_chips"] += 1
                writer.writerow(
                    {
                        "split": split,
                        "image": image_path.relative_to(config.output_dir).as_posix(),
                        "mask": mask_path.relative_to(config.output_dir).as_posix(),
                        "positive_pixels": positive_pixels,
                        "bounds": ",".join(f"{value:.8f}" for value in tile_bounds),
                    }
                )

    return summary


def _read_labels(config: TrainingDataConfig) -> gpd.GeoDataFrame:
    if not config.label_source.exists():
        raise FileNotFoundError(
            f"Training label file not found: {config.label_source}. "
            "Create/correct building polygons in QGIS and save them there."
        )
    labels = gpd.read_file(config.label_source, layer=config.label_layer)
    if labels.empty:
        raise ValueError(f"Training label file contains no features: {config.label_source}")
    if labels.crs is None:
        raise ValueError("Training labels must have a CRS.")
    return labels


def _rasterize_labels(
    labels: gpd.GeoDataFrame,
    window,
    transform,
    config: TrainingDataConfig,
) -> np.ndarray:
    if labels.empty:
        return np.zeros((int(window.height), int(window.width)), dtype="uint8")

    shapes = ((geometry, 1) for geometry in labels.geometry if geometry and not geometry.is_empty)
    return rasterize(
        shapes,
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=config.all_touched,
    )


def _write_image_chip(path: Path, image: np.ndarray, source, transform) -> None:
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        height=image.shape[1],
        width=image.shape[2],
        count=image.shape[0],
        transform=transform,
        compress="deflate",
    )
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(image)


def _write_mask_chip(path: Path, mask: np.ndarray, crs, transform) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=0,
        compress="deflate",
    ) as dataset:
        dataset.write(mask, 1)
