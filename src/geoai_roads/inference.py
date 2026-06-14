from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort
import rasterio
from affine import Affine
from rasterio.enums import Resampling

MODEL_BACKENDS = {"onnx", "keras", "pytorch"}
TTA_AUGMENTATIONS = {"none", "hflip", "vflip", "hvflip"}


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
    preserve_model_resolution: bool = False,
    probability_dir: Path | None = None,
    augmentations: Iterable[str] | None = None,
    output_name: str | None = None,
    backend: str = "onnx",
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    num_channels: int = 3,
    num_classes: int = 2,
    class_name: str = "road",
) -> int:
    backend = _normalize_backend(backend, model_path)
    augmentations = _normalize_augmentations(augmentations)
    tile_paths = sorted(tile_dir.glob("*.tif"))

    _prepare_mask_dir(mask_dir)
    if probability_dir:
        _prepare_probability_dir(probability_dir)
    if backend == "onnx":
        return _infer_tiles_onnx(
            tile_paths=tile_paths,
            mask_dir=mask_dir,
            probability_dir=probability_dir,
            model_path=model_path,
            input_size=input_size,
            mean=mean,
            std=std,
            threshold=threshold,
            preserve_model_resolution=preserve_model_resolution,
            augmentations=augmentations,
            output_name=output_name,
            class_name=class_name,
        )
    if backend == "keras":
        return _infer_tiles_keras(
            tile_paths=tile_paths,
            mask_dir=mask_dir,
            probability_dir=probability_dir,
            model_path=model_path,
            input_size=input_size,
            mean=mean,
            std=std,
            threshold=threshold,
            preserve_model_resolution=preserve_model_resolution,
            augmentations=augmentations,
            output_name=output_name,
            class_name=class_name,
        )
    if backend == "pytorch":
        return _infer_tiles_pytorch(
            tile_paths=tile_paths,
            mask_dir=mask_dir,
            probability_dir=probability_dir,
            model_path=model_path,
            input_size=input_size,
            mean=mean,
            std=std,
            threshold=threshold,
            preserve_model_resolution=preserve_model_resolution,
            augmentations=augmentations,
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


def _prepare_probability_dir(probability_dir: Path) -> None:
    probability_dir.mkdir(parents=True, exist_ok=True)
    for probability_path in probability_dir.glob("*_probability.tif"):
        if probability_path.is_file():
            probability_path.unlink()


def _infer_tiles_onnx(
    tile_paths: list[Path],
    mask_dir: Path,
    probability_dir: Path | None,
    model_path: Path,
    input_size: int,
    mean: list[float],
    std: list[float],
    threshold: float,
    preserve_model_resolution: bool = False,
    augmentations: tuple[str, ...] = ("none",),
    output_name: str | None = None,
    class_name: str = "road",
) -> int:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for tile_path in tile_paths:
        input_tensor = preprocess_tile(tile_path, input_size, mean, std)
        probability = _predict_probability_with_tta(
            input_tensor=input_tensor,
            predict=lambda tensor: session.run(
                [output_name] if output_name else None,
                {input_name: tensor},
            )[0],
            layout="nchw",
            augmentations=augmentations,
        )
        _write_mask(
            tile_path,
            mask_dir,
            probability,
            threshold,
            class_name,
            preserve_model_resolution,
            probability_dir,
        )

    return len(tile_paths)


def _infer_tiles_keras(
    tile_paths: list[Path],
    mask_dir: Path,
    probability_dir: Path | None,
    model_path: Path,
    input_size: int,
    mean: list[float],
    std: list[float],
    threshold: float,
    preserve_model_resolution: bool = False,
    augmentations: tuple[str, ...] = ("none",),
    output_name: str | None = None,
    class_name: str = "road",
) -> int:
    model = _load_keras_model(model_path)
    for tile_path in tile_paths:
        input_tensor = preprocess_tile_keras(tile_path, input_size, mean, std)
        probability = _predict_probability_with_tta(
            input_tensor=input_tensor,
            predict=lambda tensor: _select_model_output(model.predict(tensor, verbose=0), output_name),
            layout="nhwc",
            augmentations=augmentations,
        )
        _write_mask(
            tile_path,
            mask_dir,
            probability,
            threshold,
            class_name,
            preserve_model_resolution,
            probability_dir,
        )

    return len(tile_paths)


def _infer_tiles_pytorch(
    tile_paths: list[Path],
    mask_dir: Path,
    probability_dir: Path | None,
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
    preserve_model_resolution: bool = False,
    augmentations: tuple[str, ...] = ("none",),
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
        probability = _predict_probability_with_tta(
            input_tensor=input_tensor,
            predict=lambda tensor: _predict_pytorch_probability(torch, model, device, tensor),
            layout="nchw",
            augmentations=augmentations,
        )
        _write_mask(
            tile_path,
            mask_dir,
            probability,
            threshold,
            class_name,
            preserve_model_resolution,
            probability_dir,
        )

    return len(tile_paths)


def _predict_pytorch_probability(torch: Any, model: Any, device: Any, input_tensor: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        output = model(torch.from_numpy(np.ascontiguousarray(input_tensor)).to(device))
    return output.detach().cpu().numpy()


def _predict_probability_with_tta(
    input_tensor: np.ndarray,
    predict: Any,
    layout: str,
    augmentations: Iterable[str],
) -> np.ndarray:
    probabilities = []
    for augmentation in augmentations:
        augmented_tensor = _augment_input(input_tensor, augmentation, layout)
        probability = road_probability(predict(augmented_tensor))
        probabilities.append(_invert_probability_augmentation(probability, augmentation))

    return np.mean(probabilities, axis=0).astype("float32")


def _normalize_augmentations(augmentations: Iterable[str] | None) -> tuple[str, ...]:
    normalized = tuple(str(item).strip().lower() for item in (augmentations or ("none",)) if str(item).strip())
    if not normalized:
        normalized = ("none",)

    invalid = sorted(set(normalized) - TTA_AUGMENTATIONS)
    if invalid:
        expected = ", ".join(sorted(TTA_AUGMENTATIONS))
        raise ValueError(f"Unsupported inference augmentation(s): {', '.join(invalid)}. Expected: {expected}")

    return normalized


def _augment_input(input_tensor: np.ndarray, augmentation: str, layout: str) -> np.ndarray:
    if layout == "nchw":
        vertical_axis = -2
        horizontal_axis = -1
    elif layout == "nhwc":
        vertical_axis = 1
        horizontal_axis = 2
    else:
        raise ValueError(f"Unsupported model input layout for augmentation: {layout}")

    augmented = input_tensor
    if augmentation in {"vflip", "hvflip"}:
        augmented = np.flip(augmented, axis=vertical_axis)
    if augmentation in {"hflip", "hvflip"}:
        augmented = np.flip(augmented, axis=horizontal_axis)
    return np.ascontiguousarray(augmented)


def _invert_probability_augmentation(probability: np.ndarray, augmentation: str) -> np.ndarray:
    restored = probability
    if augmentation in {"vflip", "hvflip"}:
        restored = np.flip(restored, axis=0)
    if augmentation in {"hflip", "hvflip"}:
        restored = np.flip(restored, axis=1)
    return np.ascontiguousarray(restored)


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
    preserve_model_resolution: bool = False,
    probability_dir: Path | None = None,
) -> None:
    with rasterio.open(tile_path) as tile:
        profile = tile.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
        if not profile.get("tiled"):
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        if preserve_model_resolution:
            probability = np.asarray(probability)
            profile.update(
                height=probability.shape[0],
                width=probability.shape[1],
                transform=_scaled_transform(tile.transform, tile.width, tile.height, probability),
            )
        else:
            probability = _resize_to_tile(probability, tile.height, tile.width)

    mask = (probability >= threshold).astype("uint8")
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{tile_path.stem}_{class_name}_mask.tif"
    with rasterio.open(mask_path, "w", **profile) as dataset:
        dataset.write(mask, 1)
        dataset.update_tags(
            source_tile=tile_path.name,
            threshold=str(threshold),
            class_name=class_name,
        )

    if probability_dir:
        probability_profile = profile.copy()
        probability_profile.update(dtype="float32", nodata=None)
        probability_dir.mkdir(parents=True, exist_ok=True)
        probability_path = probability_dir / f"{tile_path.stem}_{class_name}_probability.tif"
        with rasterio.open(probability_path, "w", **probability_profile) as dataset:
            dataset.write(probability.astype("float32"), 1)
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


def _scaled_transform(transform: Affine, tile_width: int, tile_height: int, probability: np.ndarray) -> Affine:
    scale_x = tile_width / probability.shape[1]
    scale_y = tile_height / probability.shape[0]
    return transform * Affine.scale(scale_x, scale_y)
