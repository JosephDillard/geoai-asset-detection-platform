from __future__ import annotations

from dataclasses import dataclass
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
    def mask_dir(self) -> Path:
        return self._path("inference", "mask_dir")

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
    def output_crs(self) -> str:
        return str(self.raw["project"].get("output_crs", "EPSG:3857"))

    @property
    def processing_crs(self) -> str:
        return str(self.raw["project"].get("processing_crs", self.output_crs))

    @property
    def postgis_url(self) -> str:
        return str(self.raw["postgis"]["url"])

    @property
    def postgis_schema(self) -> str:
        return str(self.raw["postgis"].get("schema", "public"))

    @property
    def postgis_table(self) -> str:
        return str(self.raw["postgis"].get("table", "detected_roads"))

    def _path(self, section: str, key: str) -> Path:
        value = Path(str(self.raw[section][key]))
        if value.is_absolute():
            return value
        return (self.path.parent.parent / value).resolve()


def load_config(path: str | Path) -> RoadConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return RoadConfig(raw=raw, path=config_path)
