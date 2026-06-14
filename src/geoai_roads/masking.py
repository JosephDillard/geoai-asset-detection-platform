from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import sieve
from rasterio.merge import merge


@dataclass(frozen=True)
class MaskCleanupConfig:
    close_pixels: int = 0
    fill_holes_pixels: int = 0
    remove_objects_pixels: int = 0

    @property
    def enabled(self) -> bool:
        return any(
            value > 0
            for value in (
                self.close_pixels,
                self.fill_holes_pixels,
                self.remove_objects_pixels,
            )
        )


def threshold_probability_rasters(
    probability_dir: Path,
    mask_dir: Path,
    threshold: float,
    class_name: str,
    average_overlaps: bool = False,
    cleanup: MaskCleanupConfig | None = None,
) -> int:
    probability_paths = _source_probability_paths(probability_dir)
    if not probability_paths:
        raise ValueError(f"No probability rasters found in {probability_dir}")

    cleanup = cleanup or MaskCleanupConfig()
    _prepare_mask_dir(mask_dir)

    if average_overlaps:
        probability, profile = average_probability_rasters(probability_paths)
        mask = clean_binary_mask(probability >= threshold, cleanup)
        output_path = mask_dir / f"{class_name}_merged_mask.tif"
        _write_mask(output_path, mask, profile, class_name, threshold, "merged_probability")
        return 1

    for probability_path in probability_paths:
        with rasterio.open(probability_path) as dataset:
            probability = dataset.read(1)
            profile = dataset.profile.copy()
            source_tile = dataset.tags().get("source_tile", probability_path.name)
        mask = clean_binary_mask(probability >= threshold, cleanup)
        output_path = mask_dir / probability_path.name.replace("_probability.tif", "_mask.tif")
        _write_mask(output_path, mask, profile, class_name, threshold, source_tile)

    return len(probability_paths)


def average_probability_rasters(probability_paths: list[Path]) -> tuple[np.ndarray, dict]:
    datasets = [rasterio.open(path) for path in probability_paths]
    try:
        probability_sum, transform = merge(datasets, method="sum")
        probability_count, _ = merge(datasets, method="count")
        profile = datasets[0].profile.copy()
    finally:
        for dataset in datasets:
            dataset.close()

    count = probability_count[0].astype("float32")
    probability = np.divide(
        probability_sum[0].astype("float32"),
        count,
        out=np.zeros_like(probability_sum[0], dtype="float32"),
        where=count > 0,
    )
    profile.update(
        driver="GTiff",
        height=probability.shape[0],
        width=probability.shape[1],
        count=1,
        dtype="float32",
        transform=transform,
        nodata=None,
        compress="deflate",
    )
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    return probability, profile


def clean_binary_mask(mask: np.ndarray, cleanup: MaskCleanupConfig | None = None) -> np.ndarray:
    cleanup = cleanup or MaskCleanupConfig()
    mask = np.asarray(mask).astype(bool)

    if cleanup.close_pixels > 0:
        mask = _binary_closing(mask, cleanup.close_pixels)

    if cleanup.fill_holes_pixels > 0:
        background = (~mask).astype("uint8")
        background = sieve(background, size=int(cleanup.fill_holes_pixels), connectivity=8)
        mask = ~background.astype(bool)

    if cleanup.remove_objects_pixels > 0:
        mask = sieve(mask.astype("uint8"), size=int(cleanup.remove_objects_pixels), connectivity=8)
        mask = mask.astype(bool)

    return mask.astype("uint8")


def threshold_tag(threshold: float) -> str:
    return f"t{int(round(threshold * 100)):03d}"


def _source_probability_paths(probability_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in probability_dir.glob("*_probability.tif")
        if "_merged_" not in path.name
    )


def _prepare_mask_dir(mask_dir: Path) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in mask_dir.glob("*_mask.tif"):
        if mask_path.is_file():
            mask_path.unlink()


def _write_mask(
    output_path: Path,
    mask: np.ndarray,
    profile: dict,
    class_name: str,
    threshold: float,
    source_tile: str,
) -> None:
    mask_profile = profile.copy()
    mask_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
    mask_profile.pop("blockxsize", None)
    mask_profile.pop("blockysize", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **mask_profile) as dataset:
        dataset.write(mask.astype("uint8"), 1)
        dataset.update_tags(
            source_tile=source_tile,
            threshold=str(threshold),
            class_name=class_name,
        )


def _binary_closing(mask: np.ndarray, iterations: int) -> np.ndarray:
    closed = mask
    for _ in range(iterations):
        closed = _binary_dilation(closed)
    for _ in range(iterations):
        closed = _binary_erosion(closed)
    return closed


def _binary_dilation(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for row_offset in range(3):
        for col_offset in range(3):
            result |= padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
    return result


def _binary_erosion(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    for row_offset in range(3):
        for col_offset in range(3):
            result &= padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
    return result
