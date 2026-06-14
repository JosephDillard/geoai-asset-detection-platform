from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "imagery" / "example-cog.tif"
MODEL_PATH = ROOT / "models" / "road-segmentation.onnx"


def main() -> None:
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_demo_imagery(IMAGE_PATH)
    write_demo_model(MODEL_PATH)

    print(f"Wrote {IMAGE_PATH}")
    print(f"Wrote {MODEL_PATH}")


def write_demo_imagery(path: Path) -> None:
    width = 256
    height = 256
    image = np.zeros((3, height, width), dtype="uint8")
    image[0, :, :] = 76
    image[1, :, :] = 110
    image[2, :, :] = 86

    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]

    horizontal = np.abs(rows - 128) <= 4
    vertical = np.abs(cols - 92) <= 4
    diagonal = np.abs(rows - (0.55 * cols + 40)) <= 4
    road_mask = horizontal | vertical | diagonal

    image[:, road_mask] = np.array([230, 230, 225], dtype="uint8")[:, None]

    transform = from_origin(-106.85, 35.05, 0.001, 0.001)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        tiled=True,
        compress="deflate",
    ) as dataset:
        dataset.write(image)


def write_demo_model(path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise SystemExit(
            "The demo model generator needs onnx. Install it with: "
            "python -m pip install -e .[demo]"
        ) from exc

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1024, 1024])
    output_tensor = helper.make_tensor_value_info(
        "road_probability",
        TensorProto.FLOAT,
        [1, 1, 1024, 1024],
    )
    reduce_mean = helper.make_node(
        "ReduceMean",
        inputs=["input"],
        outputs=["road_probability"],
        axes=[1],
        keepdims=1,
    )
    graph = helper.make_graph(
        [reduce_mean],
        "DemoRoadSegmentation",
        [input_tensor],
        [output_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="geoai_roads_demo",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    main()
