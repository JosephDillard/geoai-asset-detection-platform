from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.enums import Resampling

MODEL_BACKENDS = {"onnx", "keras", "pytorch"}


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


def preprocess_tile_keras(
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
    image = np.moveaxis(image, 0, -1)
    mean_array = np.asarray(mean, dtype="float32")[None, None, :]
    std_array = np.asarray(std, dtype="float32")[None, None, :]
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
    backend: str = "onnx",
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    num_channels: int = 3,
    num_classes: int = 2,
    class_name: str = "road",
) -> int:
    backend = _normalize_backend(backend, model_path)
    tile_paths = sorted(tile_dir.glob("*.tif"))

    _prepare_mask_dir(mask_dir)
    if backend == "onnx":
        return _infer_tiles_onnx(
            tile_paths=tile_paths,
            mask_dir=mask_dir,
            model_path=model_path,
            input_size=input_size,
            mean=mean,
            std=std,
            threshold=threshold,
            output_name=output_name,
            class_name=class_name,
        )
    if backend == "keras":
        return _infer_tiles_keras(
            tile_paths=tile_paths,
            mask_dir=mask_dir,
            model_path=model_path,
            input_size=input_size,
            mean=mean,
            std=std,
            threshold=threshold,
            output_name=output_name,
            class_name=class_name,
        )
    if backend == "pytorch":
        return _infer_tiles_pytorch(
            tile_paths=tile_paths,
            mask_dir=mask_dir,
            model_path=model_path,
            input_size=input_size,
            mean=mean,
            std=std,
            threshold=threshold,
            architecture=architecture,
            encoder_name=encoder_name,
            num_channels=num_channels,
            num_classes=num_classes,
            class_name=class_name,
        )
    raise ValueError(f"Unsupported road segmentation model backend: {backend}")


def _prepare_mask_dir(mask_dir: Path) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in mask_dir.glob("*_mask.tif"):
        if mask_path.is_file():
            mask_path.unlink()


def _infer_tiles_onnx(
    tile_paths: list[Path],
    mask_dir: Path,
    model_path: Path,
    input_size: int,
    mean: list[float],
    std: list[float],
    threshold: float,
    output_name: str | None = None,
    class_name: str = "road",
) -> int:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for tile_path in tile_paths:
        input_tensor = preprocess_tile(tile_path, input_size, mean, std)
        outputs = session.run([output_name] if output_name else None, {input_name: input_tensor})
        probability = road_probability(outputs[0])
        _write_mask(tile_path, mask_dir, probability, threshold, class_name)

    return len(tile_paths)


def _infer_tiles_keras(
    tile_paths: list[Path],
    mask_dir: Path,
    model_path: Path,
    input_size: int,
    mean: list[float],
    std: list[float],
    threshold: float,
    output_name: str | None = None,
    class_name: str = "road",
) -> int:
    model = _load_keras_model(model_path)
    for tile_path in tile_paths:
        input_tensor = preprocess_tile_keras(tile_path, input_size, mean, std)
        outputs = model.predict(input_tensor, verbose=0)
        probability = road_probability(_select_model_output(outputs, output_name))
        _write_mask(tile_path, mask_dir, probability, threshold, class_name)

    return len(tile_paths)


def _infer_tiles_pytorch(
    tile_paths: list[Path],
    mask_dir: Path,
    model_path: Path,
    input_size: int,
    mean: list[float],
    std: list[float],
    threshold: float,
    architecture: str,
    encoder_name: str,
    num_channels: int,
    num_classes: int,
    class_name: str = "road",
) -> int:
    torch, model, device = _load_pytorch_smp_model(
        model_path=model_path,
        architecture=architecture,
        encoder_name=encoder_name,
        num_channels=num_channels,
        num_classes=num_classes,
    )

    for tile_path in tile_paths:
        input_tensor = preprocess_tile(tile_path, input_size, mean, std)
        with torch.no_grad():
            output = model(torch.from_numpy(input_tensor).to(device))
        probability = road_probability(output.detach().cpu().numpy())
        _write_mask(tile_path, mask_dir, probability, threshold, class_name)

    return len(tile_paths)


def _load_keras_model(model_path: Path) -> Any:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except Exception as exc:
        raise RuntimeError(
            "The Keras model backend requires TensorFlow/Keras. Use a Python 3.10-3.12 "
            'environment and install with `python -m pip install -e ".[keras]"`, or use '
            "an ONNX model backend."
        ) from exc

    return load_model(
        str(model_path),
        custom_objects=_keras_custom_objects(tf),
        compile=False,
    )


def _load_pytorch_smp_model(
    model_path: Path,
    architecture: str,
    encoder_name: str,
    num_channels: int,
    num_classes: int,
) -> tuple[Any, Any, Any]:
    try:
        import torch
        import segmentation_models_pytorch as smp
    except Exception as exc:
        raise RuntimeError(
            "The PyTorch model backend requires torch and segmentation-models-pytorch. "
            'Install with `python -m pip install -e ".[pytorch]"`, or rebuild the '
            "GeoAI Docker image."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.create_model(
        arch=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=num_channels,
        classes=num_classes,
    )

    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return torch, model, device


def _keras_custom_objects(tf: Any) -> dict[str, Any]:
    def iou_coef(y_true: Any, y_pred: Any, smooth: float = 1e-6) -> Any:
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)

    def dice_coef(y_true: Any, y_pred: Any, smooth: float = 1e-6) -> Any:
        intersection = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return (2.0 * intersection + smooth) / (total + smooth)

    def soft_dice_loss(y_true: Any, y_pred: Any) -> Any:
        return 1 - dice_coef(y_true, y_pred)

    return {
        "soft_dice_loss": soft_dice_loss,
        "dice_coef": dice_coef,
        "iou_coef": iou_coef,
    }


def _select_model_output(outputs: Any, output_name: str | None) -> np.ndarray:
    if isinstance(outputs, dict):
        if output_name:
            if output_name not in outputs:
                raise ValueError(f"Keras model did not return output named {output_name}")
            return np.asarray(outputs[output_name])
        return np.asarray(next(iter(outputs.values())))

    if isinstance(outputs, (list, tuple)):
        if output_name:
            raise ValueError("Named Keras outputs must be returned as a mapping")
        return np.asarray(outputs[0])

    return np.asarray(outputs)


def _write_mask(
    tile_path: Path,
    mask_dir: Path,
    probability: np.ndarray,
    threshold: float,
    class_name: str = "road",
) -> None:
    with rasterio.open(tile_path) as tile:
        probability = _resize_to_tile(probability, tile.height, tile.width)
        profile = tile.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")

    mask = (probability >= threshold).astype("uint8")
    mask_path = mask_dir / f"{tile_path.stem}_{class_name}_mask.tif"
    with rasterio.open(mask_path, "w", **profile) as dataset:
        dataset.write(mask, 1)
        dataset.update_tags(
            source_tile=tile_path.name,
            threshold=str(threshold),
            class_name=class_name,
        )


def _normalize_backend(backend: str | None, model_path: Path) -> str:
    if not backend:
        suffix = model_path.suffix.lower()
        backend = "keras" if suffix in {".keras", ".h5", ".hdf5"} else "onnx"
    backend = backend.lower().strip()
    if backend not in MODEL_BACKENDS:
        raise ValueError(
            f"Unsupported model backend: {backend}. Expected one of: "
            f"{', '.join(sorted(MODEL_BACKENDS))}"
        )
    return backend


def _resize_to_tile(probability: np.ndarray, height: int, width: int) -> np.ndarray:
    if probability.shape == (height, width):
        return probability

    row_idx = np.linspace(0, probability.shape[0] - 1, height).round().astype(int)
    col_idx = np.linspace(0, probability.shape[1] - 1, width).round().astype(int)
    return probability[row_idx][:, col_idx]
