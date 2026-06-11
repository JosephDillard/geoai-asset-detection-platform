from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.enums import Resampling


def _to_float_rgb(image: np.ndarray) -> np.ndarray:
    image = image.astype("float32")
    max_value = float(np.nanmax(image)) if image.size else 0
    if max_value > 1.0:
        image /= 255.0 if max_value <= 255 else max_value
    return image


def preprocess_tile(
    tile_path: Path,
    input_size: int,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    with rasterio.open(tile_path) as dataset:
        image = dataset.read(
            out_shape=(dataset.count, input_size, input_size),
            resampling=Resampling.bilinear,
        )

    image = _to_float_rgb(image)
    mean_array = np.asarray(mean, dtype="float32")[:, None, None]
    std_array = np.asarray(std, dtype="float32")[:, None, None]
    image = (image - mean_array) / std_array
    return image[None, :, :, :].astype("float32")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def road_probability(output: np.ndarray) -> np.ndarray:
    output = np.asarray(output)
    output = np.squeeze(output)

    if output.ndim == 3:
        if output.shape[0] in {1, 2}:
            output = _class_probability(output, axis=0)
        elif output.shape[-1] in {1, 2}:
            output = _class_probability(output, axis=-1)
        else:
            raise ValueError(f"Unsupported model output shape for road mask: {output.shape}")

    if output.min() < 0 or output.max() > 1:
        output = sigmoid(output)

    return output.astype("float32")


def _class_probability(output: np.ndarray, axis: int) -> np.ndarray:
    class_count = output.shape[axis]
    road_index = 0 if class_count == 1 else 1

    if class_count == 1:
        return np.take(output, road_index, axis=axis)

    channel_sum = np.sum(output, axis=axis)
    is_probability = output.min() >= 0 and output.max() <= 1 and np.allclose(
        channel_sum,
        1.0,
        atol=1e-3,
    )
    probabilities = output if is_probability else softmax(output, axis=axis)
    return np.take(probabilities, road_index, axis=axis)


def infer_tiles(
    tile_dir: Path,
    mask_dir: Path,
    model_path: Path,
    input_size: int,
    mean: list[float],
    std: list[float],
    threshold: float,
    output_name: str | None = None,
) -> int:
    mask_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    tile_paths = sorted(tile_dir.glob("*.tif"))
    for tile_path in tile_paths:
        input_tensor = preprocess_tile(tile_path, input_size, mean, std)
        outputs = session.run([output_name] if output_name else None, {input_name: input_tensor})
        probability = road_probability(outputs[0])

        with rasterio.open(tile_path) as tile:
            probability = _resize_to_tile(probability, tile.height, tile.width)
            profile = tile.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")

        mask = (probability >= threshold).astype("uint8")
        mask_path = mask_dir / f"{tile_path.stem}_road_mask.tif"
        with rasterio.open(mask_path, "w", **profile) as dataset:
            dataset.write(mask, 1)
            dataset.update_tags(source_tile=tile_path.name, threshold=str(threshold))

    return len(tile_paths)


def _resize_to_tile(probability: np.ndarray, height: int, width: int) -> np.ndarray:
    if probability.shape == (height, width):
        return probability

    row_idx = np.linspace(0, probability.shape[0] - 1, height).round().astype(int)
    col_idx = np.linspace(0, probability.shape[1] - 1, width).round().astype(int)
    return probability[row_idx][:, col_idx]
