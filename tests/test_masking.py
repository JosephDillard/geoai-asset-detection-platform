from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geoai_roads.masking import (
    MaskCleanupConfig,
    clean_binary_mask,
    threshold_probability_rasters,
)


def _write_probability(path: Path, probability: np.ndarray, transform) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=probability.shape[0],
        width=probability.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:26913",
        transform=transform,
        nodata=None,
    ) as dataset:
        dataset.write(probability.astype("float32"), 1)
        dataset.update_tags(source_tile=path.name, class_name="building")


def test_clean_binary_mask_fills_holes_and_removes_specks() -> None:
    mask = np.zeros((8, 8), dtype="uint8")
    mask[1:6, 1:6] = 1
    mask[3, 3] = 0
    mask[7, 7] = 1

    cleaned = clean_binary_mask(
        mask,
        MaskCleanupConfig(fill_holes_pixels=2, remove_objects_pixels=2),
    )

    assert cleaned[3, 3] == 1
    assert cleaned[7, 7] == 0


def test_threshold_probability_rasters_can_average_overlaps(tmp_path: Path) -> None:
    probability_dir = tmp_path / "probabilities"
    mask_dir = tmp_path / "masks"
    _write_probability(
        probability_dir / "tile_1_building_probability.tif",
        np.full((4, 4), 0.2, dtype="float32"),
        from_origin(0, 4, 1, 1),
    )
    _write_probability(
        probability_dir / "tile_2_building_probability.tif",
        np.full((4, 4), 0.8, dtype="float32"),
        from_origin(2, 4, 1, 1),
    )

    count = threshold_probability_rasters(
        probability_dir=probability_dir,
        mask_dir=mask_dir,
        threshold=0.5,
        class_name="building",
        average_overlaps=True,
    )

    assert count == 1
    with rasterio.open(mask_dir / "building_merged_mask.tif") as dataset:
        mask = dataset.read(1)

    assert mask.shape == (4, 6)
    assert mask[:, :2].sum() == 0
    assert mask[:, 2:4].sum() == 8
    assert mask[:, 4:].sum() == 8
