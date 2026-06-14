from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from geoai_roads.vectorize import vectorize_masks


def _write_mask(path: Path, mask: np.ndarray, crs: str, transform) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    ) as dataset:
        dataset.write(mask.astype("uint8"), 1)


def _write_tagged_mask(path: Path, mask: np.ndarray, crs: str, transform, class_name: str) -> None:
    _write_mask(path, mask, crs, transform)
    with rasterio.open(path, "r+") as dataset:
        dataset.update_tags(class_name=class_name)


def test_vectorize_skips_masks_above_max_coverage(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    output = tmp_path / "roads.gpkg"
    mask = np.ones((10, 10), dtype="uint8")
    _write_mask(mask_dir / "full_road_mask.tif", mask, "EPSG:26913", from_origin(0, 10, 1, 1))

    count = vectorize_masks(
        mask_dir=mask_dir,
        output_path=output,
        processing_crs="EPSG:26913",
        output_crs="EPSG:4326",
        min_area_m2=0,
        simplify_tolerance_m=0,
        max_mask_coverage=0.5,
    )

    assert count == 0
    assert gpd.read_file(output).empty


def test_vectorize_skips_masks_above_max_pixel_size(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    output = tmp_path / "roads.gpkg"
    mask = np.zeros((10, 10), dtype="uint8")
    mask[4:6, :] = 1
    _write_mask(
        mask_dir / "coarse_road_mask.tif",
        mask,
        "EPSG:4326",
        from_origin(-106.85, 35.05, 0.001, 0.001),
    )

    count = vectorize_masks(
        mask_dir=mask_dir,
        output_path=output,
        processing_crs="EPSG:3857",
        output_crs="EPSG:4326",
        min_area_m2=0,
        simplify_tolerance_m=0,
        max_source_pixel_size_m=5,
    )

    assert count == 0
    assert gpd.read_file(output).empty


def test_vectorize_keeps_reasonable_mask(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    output = tmp_path / "roads.gpkg"
    mask = np.zeros((10, 10), dtype="uint8")
    mask[4:6, :] = 1
    _write_mask(mask_dir / "road_mask.tif", mask, "EPSG:26913", from_origin(0, 10, 1, 1))

    count = vectorize_masks(
        mask_dir=mask_dir,
        output_path=output,
        processing_crs="EPSG:26913",
        output_crs="EPSG:4326",
        min_area_m2=0,
        simplify_tolerance_m=0,
        max_mask_coverage=0.5,
        max_source_pixel_size_m=5,
    )

    roads = gpd.read_file(output)
    assert count == 1
    assert len(roads) == 1


def test_vectorize_uses_mask_class_name_tag(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    output = tmp_path / "buildings.gpkg"
    mask = np.zeros((10, 10), dtype="uint8")
    mask[2:5, 2:5] = 1
    _write_tagged_mask(
        mask_dir / "building_mask.tif",
        mask,
        "EPSG:26913",
        from_origin(0, 10, 1, 1),
        "building",
    )

    count = vectorize_masks(
        mask_dir=mask_dir,
        output_path=output,
        processing_crs="EPSG:26913",
        output_crs="EPSG:4326",
        min_area_m2=0,
        simplify_tolerance_m=0,
    )

    buildings = gpd.read_file(output)
    assert count == 1
    assert buildings.loc[0, "class_name"] == "building"
