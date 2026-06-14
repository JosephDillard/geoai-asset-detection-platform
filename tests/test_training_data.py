from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geoai_roads.training_data import (
    export_training_chips,
    load_training_data_config,
)


def test_export_training_chips_from_qgis_labels(tmp_path: Path) -> None:
    imagery = tmp_path / "imagery.tif"
    labels = tmp_path / "labels.gpkg"
    config_path = tmp_path / "config" / "training.yaml"
    output_dir = tmp_path / "training"

    image = np.zeros((3, 8, 8), dtype="uint8")
    image[0, :, :] = 100
    with rasterio.open(
        imagery,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=3,
        dtype="uint8",
        crs="EPSG:26913",
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(image)

    labels_frame = gpd.GeoDataFrame(
        {"name": ["building"]},
        geometry=[box(1, 3, 5, 7)],
        crs="EPSG:26913",
    )
    labels_frame.to_file(labels, layer="buildings", driver="GPKG")

    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        f"""
project:
  name: test
asset:
  class_name: building
imagery:
  source: {imagery.as_posix()}
  bands: [1, 2, 3]
labels:
  source: {labels.as_posix()}
  layer: buildings
training_data:
  output_dir: {output_dir.as_posix()}
  tile_size: 4
  overlap: 0
  validation_fraction: 0
  include_empty_fraction: 0
  min_mask_pixels: 1
  skip_partial_tiles: true
  seed: 13
model:
  base_path: models/base.pth
  output_path: models/output.pth
training:
  chips_dir: {output_dir.as_posix()}
""",
        encoding="utf-8",
    )

    config = load_training_data_config(config_path)
    summary = export_training_chips(config)

    assert summary["train_chips"] > 0
    assert summary["positive_chips"] > 0
    image_chips = sorted((output_dir / "train" / "images").glob("*.tif"))
    mask_chips = sorted((output_dir / "train" / "masks").glob("*.tif"))
    assert len(image_chips) == len(mask_chips) == summary["train_chips"]
    with rasterio.open(mask_chips[0]) as mask_dataset:
        assert mask_dataset.read(1).sum() > 0
    assert (output_dir / "manifest.csv").exists()
