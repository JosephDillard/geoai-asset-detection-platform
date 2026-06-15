from pathlib import Path
import zipfile

import geopandas as gpd
import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from shapely.geometry import box

from geoai_roads.api import create_app
from geoai_roads.training_data import load_training_data_config
from geoai_roads.training_web import build_label_export_package, save_uploaded_labels


def _write_imagery(path: Path) -> None:
    image = np.zeros((3, 8, 8), dtype="uint8")
    with rasterio.open(
        path,
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


def _write_labels(path: Path) -> None:
    labels = gpd.GeoDataFrame(
        {"name": ["training"]},
        geometry=[box(1, 3, 5, 7)],
        crs="EPSG:26913",
    )
    labels.to_file(path, layer="buildings", driver="GPKG")


def _write_training_config(tmp_path: Path) -> Path:
    imagery = tmp_path / "imagery.tif"
    labels = tmp_path / "labels" / "taos_building_labels.gpkg"
    output_dir = tmp_path / "training"
    _write_imagery(imagery)
    labels.parent.mkdir(parents=True)
    _write_labels(labels)

    config_path = tmp_path / "config" / "training.yaml"
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
    return config_path


def test_training_pages_are_registered(tmp_path: Path) -> None:
    config_path = _write_training_config(tmp_path)
    app = create_app(default_catalog="config/workflows.example.yaml", allowed_origins=[])
    app.state.default_training_config = str(config_path)
    client = TestClient(app)

    response = client.get("/training")

    assert response.status_code == 200
    assert "Building Model Training" in response.text
    assert "/training/export" in response.text


def test_training_export_page_explains_downloads(tmp_path: Path) -> None:
    config_path = _write_training_config(tmp_path)
    app = create_app(default_catalog="config/workflows.example.yaml", allowed_origins=[])
    app.state.default_training_config = str(config_path)
    client = TestClient(app)

    response = client.get("/training/export")

    assert response.status_code == 200
    assert "Label Package" in response.text
    assert "Imagery COG" in response.text
    assert "Training Chips" in response.text
    assert "taos_building_labels.gpkg" in response.text
    assert "manifest.csv" in response.text
    assert "/training/export/imagery" in response.text


def test_training_imagery_download_returns_configured_cog(tmp_path: Path) -> None:
    config_path = _write_training_config(tmp_path)
    app = create_app(default_catalog="config/workflows.example.yaml", allowed_origins=[])
    client = TestClient(app)

    response = client.get("/training/export/imagery", params={"config_path": str(config_path)})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/tiff")
    assert "imagery.tif" in response.headers["content-disposition"]
    assert len(response.content) > 0


def test_build_label_export_package_contains_qgis_package(tmp_path: Path) -> None:
    config_path = _write_training_config(tmp_path)

    package_path = build_label_export_package(config_path)

    assert package_path.exists()
    with zipfile.ZipFile(package_path) as archive:
        names = set(archive.namelist())
        readme = archive.read("README.txt").decode("utf-8")
    assert "taos_building_labels.gpkg" in names
    assert "README.txt" in names
    assert "The imagery COG is not included in this ZIP" in readme


def test_build_label_export_package_allows_empty_starting_labels(tmp_path: Path) -> None:
    config_path = _write_training_config(tmp_path)
    config = load_training_data_config(config_path)
    config.label_source.unlink()

    package_path = build_label_export_package(config_path)

    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(package_path) as archive:
        archive.extract("taos_building_labels.gpkg", extract_dir)
    labels = gpd.read_file(extract_dir / "taos_building_labels.gpkg", layer="buildings")
    assert labels.empty
    assert labels.crs is not None


def test_save_uploaded_labels_normalizes_to_training_target(tmp_path: Path) -> None:
    config_path = _write_training_config(tmp_path)
    config = load_training_data_config(config_path)
    upload_path = tmp_path / "upload.geojson"
    labels = gpd.GeoDataFrame(
        {"name": ["upload"]},
        geometry=[box(2, 2, 4, 4)],
        crs="EPSG:26913",
    )
    labels.to_file(upload_path, driver="GeoJSON")

    result = save_uploaded_labels(config, upload_path.name, upload_path.read_bytes())

    assert result["features"] == 1
    saved = gpd.read_file(config.label_source, layer="buildings")
    assert len(saved) == 1
    assert saved.loc[0, "name"] == "upload"
