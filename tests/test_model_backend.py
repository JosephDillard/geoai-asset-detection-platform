from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geoai_roads.config import RoadConfig
from geoai_roads.inference import preprocess_tile_keras


def _road_config(model: dict[str, object]) -> RoadConfig:
    return RoadConfig(
        raw={"model": model},
        path=Path("config/test.yaml").resolve(),
    )


def test_model_backend_uses_explicit_value() -> None:
    config = _road_config({"path": "models/road-segmentation.onnx", "backend": "keras"})

    assert config.model_backend == "keras"


def test_model_backend_detects_keras_file_suffix() -> None:
    config = _road_config({"path": "models/aerial-image-road-segmentation-xp.keras"})

    assert config.model_backend == "keras"


def test_model_backend_defaults_to_onnx() -> None:
    config = _road_config({"path": "models/road-segmentation.onnx"})

    assert config.model_backend == "onnx"


def test_model_backend_rejects_unknown_value() -> None:
    config = _road_config({"path": "models/road-segmentation.onnx", "backend": "pytorch"})

    with pytest.raises(ValueError, match="model.backend"):
        config.model_backend


def test_postgis_url_can_be_overridden_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RoadConfig(
        raw={
            "model": {"path": "models/road-segmentation.onnx"},
            "postgis": {"url": "postgresql+psycopg://gsb:gsb@localhost:5432/geostatusboard"},
        },
        path=Path("config/test.yaml").resolve(),
    )
    monkeypatch.setenv("GEOAI_POSTGIS_URL", "postgresql+psycopg://gsb:gsb@postgis:5432/geostatusboard")

    assert config.postgis_url == "postgresql+psycopg://gsb:gsb@postgis:5432/geostatusboard"


def test_preprocess_tile_keras_returns_nhwc_float_batch(tmp_path: Path) -> None:
    tile_path = tmp_path / "tile.tif"
    image = np.stack(
        [
            np.full((4, 4), 255, dtype="uint8"),
            np.full((4, 4), 128, dtype="uint8"),
            np.zeros((4, 4), dtype="uint8"),
        ]
    )

    with rasterio.open(
        tile_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=3,
        dtype="uint8",
        crs="EPSG:26913",
        transform=from_origin(0, 4, 1, 1),
    ) as dataset:
        dataset.write(image)

    tensor = preprocess_tile_keras(
        tile_path=tile_path,
        input_size=2,
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
    )

    assert tensor.shape == (1, 2, 2, 3)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0
