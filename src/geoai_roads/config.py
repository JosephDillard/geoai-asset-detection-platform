from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RoadConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def imagery_source(self) -> Path:
        return self._path("imagery", "source")

    @property
    def imagery_bands(self) -> list[int]:
        return list(self.raw["imagery"].get("bands", [1, 2, 3]))

    @property
    def tile_size(self) -> int:
        return int(self.raw["tiling"].get("tile_size", 1024))

    @property
    def tile_overlap(self) -> int:
        return int(self.raw["tiling"].get("overlap", 128))

    @property
    def tile_dir(self) -> Path:
        return self._path("tiling", "output_dir")

    @property
    def model_path(self) -> Path:
        return self._path("model", "path")

    @property
    def model_backend(self) -> str:
        value = self.raw["model"].get("backend")
        if value:
            backend = str(value).strip().lower()
        else:
            suffix = self.model_path.suffix.lower()
            backend = "keras" if suffix in {".keras", ".h5", ".hdf5"} else "onnx"

        if backend not in {"onnx", "keras", "pytorch"}:
            raise ValueError("model.backend must be one of: keras, onnx, pytorch")
        return backend

    @property
    def model_architecture(self) -> str:
        return str(self.raw["model"].get("architecture", "unet"))

    @property
    def model_encoder_name(self) -> str:
        return str(self.raw["model"].get("encoder_name", "resnet34"))

    @property
    def model_num_channels(self) -> int:
        return int(self.raw["model"].get("num_channels", 3))

    @property
    def model_num_classes(self) -> int:
        return int(self.raw["model"].get("num_classes", 2))

    @property
    def model_input_size(self) -> int:
        return int(self.raw["model"].get("input_size", self.tile_size))

    @property
    def model_mean(self) -> list[float]:
        return list(self.raw["model"].get("mean", [0.0, 0.0, 0.0]))

    @property
    def model_std(self) -> list[float]:
        return list(self.raw["model"].get("std", [1.0, 1.0, 1.0]))

    @property
    def model_output_name(self) -> str | None:
        value = self.raw["model"].get("output_name")
        return str(value) if value else None

    @property
    def road_threshold(self) -> float:
        return float(self.raw["inference"].get("threshold", 0.5))

    @property
    def class_name(self) -> str:
        asset = self.raw.get("asset") or {}
        project = self.raw.get("project") or {}
        return str(asset.get("class_name") or project.get("class_name") or "road")

    @property
    def mask_dir(self) -> Path:
        return self._path("inference", "mask_dir")

    @property
    def save_probability(self) -> bool:
        return bool(self.raw["inference"].get("save_probability", False))

    @property
    def probability_dir(self) -> Path | None:
        value = self.raw["inference"].get("probability_dir")
        if value:
            return self._resolve_path(value)
        if self.save_probability:
            return self.mask_dir.parent / f"{self.mask_dir.name}_probabilities"
        return None

    @property
    def inference_augmentations(self) -> list[str]:
        value = self.raw["inference"].get("augmentations", ["none"])
        if isinstance(value, bool):
            return ["none", "hflip", "vflip", "hvflip"] if value else ["none"]
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    @property
    def preserve_model_resolution(self) -> bool:
        return bool(self.raw["inference"].get("preserve_model_resolution", False))

    @property
    def vector_output(self) -> Path:
        return self._path("vectorization", "output")

    @property
    def min_area_m2(self) -> float:
        return float(self.raw["vectorization"].get("min_area_m2", 0))

    @property
    def simplify_tolerance_m(self) -> float:
        return float(self.raw["vectorization"].get("simplify_tolerance_m", 0))

    @property
    def smooth_tolerance_m(self) -> float:
        return float(self.raw["vectorization"].get("smooth_tolerance_m", 0))

    @property
    def rectangularize(self) -> bool:
        return bool(self.raw["vectorization"].get("rectangularize", False))

    @property
    def rectangularize_min_area_ratio(self) -> float:
        return float(self.raw["vectorization"].get("rectangularize_min_area_ratio", 0.9))

    @property
    def dissolve_overlaps(self) -> bool:
        return bool(self.raw["vectorization"].get("dissolve_overlaps", False))

    @property
    def max_mask_coverage(self) -> float:
        return float(self.raw["vectorization"].get("max_mask_coverage", 0))

    @property
    def max_source_pixel_size_m(self) -> float:
        return float(self.raw["vectorization"].get("max_source_pixel_size_m", 0))

    @property
    def output_crs(self) -> str:
        return str(self.raw["project"].get("output_crs", "EPSG:3857"))

    @property
    def processing_crs(self) -> str:
        return str(self.raw["project"].get("processing_crs", self.output_crs))

    @property
    def postgis_url(self) -> str:
        override = os.getenv("GEOAI_POSTGIS_URL")
        if override:
            return override
        return str(self.raw["postgis"]["url"])

    @property
    def postgis_schema(self) -> str:
        return str(self.raw["postgis"].get("schema", "public"))

    @property
    def postgis_table(self) -> str:
        return str(self.raw["postgis"].get("table", "detected_roads"))

    def _path(self, section: str, key: str) -> Path:
        return self._resolve_path(self.raw[section][key])

    def _resolve_path(self, value: Any) -> Path:
        value = Path(str(value))
        if value.is_absolute():
            return value
        return (self.path.parent.parent / value).resolve()


def load_config(path: str | Path) -> RoadConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return RoadConfig(raw=raw, path=config_path)
