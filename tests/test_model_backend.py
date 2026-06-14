from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.coords import BoundingBox
from rasterio.transform import from_origin

from geoai_roads.config import RoadConfig
from geoai_roads.inference import _predict_probability_with_tta, _write_mask, preprocess_tile_keras


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


def test_model_backend_accepts_pytorch() -> None:
    config = _road_config({"path": "models/road-segmentation.onnx", "backend": "pytorch"})

    assert config.model_backend == "pytorch"


def test_model_backend_rejects_unknown_value() -> None:
    config = _road_config({"path": "models/road-segmentation.onnx", "backend": "tensorflow-lite"})

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


def test_inference_probability_config_defaults_to_sibling_dir() -> None:
    config = RoadConfig(
        raw={
            "model": {"path": "models/road-segmentation.onnx"},
            "inference": {
                "mask_dir": "outputs/building_masks",
                "save_probability": True,
                "augmentations": "none,hflip,vflip,hvflip",
            },
        },
        path=Path("config/test.yaml").resolve(),
    )

    assert config.probability_dir == config.mask_dir.parent / "building_masks_probabilities"
    assert config.inference_augmentations == ["none", "hflip", "vflip", "hvflip"]


def test_predict_probability_with_tta_inverts_flips() -> None:
    input_tensor = np.arange(6, dtype="float32").reshape(1, 1, 2, 3) / 10.0

    probability = _predict_probability_with_tta(
        input_tensor=input_tensor,
        predict=lambda tensor: tensor[0, 0],
        layout="nchw",
        augmentations=("none", "hflip", "vflip", "hvflip"),
    )

    assert np.allclose(probability, input_tensor[0, 0])


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


def test_write_mask_can_preserve_model_resolution(tmp_path: Path) -> None:
    tile_path = tmp_path / "tile.tif"
    mask_dir = tmp_path / "masks"
    image = np.zeros((3, 2, 2), dtype="uint8")
    transform = from_origin(10, 20, 1, 1)

    with rasterio.open(
        tile_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=3,
        dtype="uint8",
        crs="EPSG:26913",
        transform=transform,
    ) as dataset:
        dataset.write(image)

    probability = np.ones((4, 4), dtype="float32")
    _write_mask(
        tile_path=tile_path,
        mask_dir=mask_dir,
        probability=probability,
        threshold=0.5,
        class_name="building",
        preserve_model_resolution=True,
    )

    with rasterio.open(mask_dir / "tile_building_mask.tif") as mask:
        assert mask.width == 4
        assert mask.height == 4
        assert mask.res == (0.5, 0.5)
        assert mask.bounds == BoundingBox(left=10, bottom=18, right=12, top=20)


def test_write_mask_can_save_probability_raster(tmp_path: Path) -> None:
    tile_path = tmp_path / "tile.tif"
    mask_dir = tmp_path / "masks"
    probability_dir = tmp_path / "probabilities"
    image = np.zeros((3, 2, 2), dtype="uint8")

    with rasterio.open(
        tile_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=3,
        dtype="uint8",
        crs="EPSG:26913",
        transform=from_origin(10, 20, 1, 1),
    ) as dataset:
        dataset.write(image)

    probability = np.array([[0.25, 0.75], [0.4, 0.9]], dtype="float32")
    _write_mask(
        tile_path=tile_path,
        mask_dir=mask_dir,
        probability=probability,
        threshold=0.5,
        class_name="building",
        probability_dir=probability_dir,
    )

    with rasterio.open(mask_dir / "tile_building_mask.tif") as mask:
        assert mask.read(1).tolist() == [[0, 1], [0, 1]]

    with rasterio.open(probability_dir / "tile_building_probability.tif") as probability_raster:
        assert probability_raster.dtypes == ("float32",)
        assert np.allclose(probability_raster.read(1), probability)
