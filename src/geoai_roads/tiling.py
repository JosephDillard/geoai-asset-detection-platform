from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.windows import Window


def iter_windows(width: int, height: int, tile_size: int, overlap: int) -> list[Window]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be greater than or equal to 0 and smaller than tile_size")

    stride = tile_size - overlap
    windows: list[Window] = []

    y_offsets = list(range(0, max(height - tile_size, 0) + 1, stride))
    x_offsets = list(range(0, max(width - tile_size, 0) + 1, stride))

    if not y_offsets or y_offsets[-1] != max(height - tile_size, 0):
        y_offsets.append(max(height - tile_size, 0))
    if not x_offsets or x_offsets[-1] != max(width - tile_size, 0):
        x_offsets.append(max(width - tile_size, 0))

    for y in y_offsets:
        for x in x_offsets:
            windows.append(Window(x, y, min(tile_size, width - x), min(tile_size, height - y)))

    return windows


def extract_tiles(
    source: Path,
    output_dir: Path,
    bands: list[int],
    tile_size: int,
    overlap: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source) as dataset:
        windows = iter_windows(dataset.width, dataset.height, tile_size, overlap)
        profile = dataset.profile.copy()
        profile.update(driver="GTiff", tiled=True, compress="deflate", count=len(bands))

        stem = source.stem
        for index, window in enumerate(windows):
            transform = dataset.window_transform(window)
            tile_profile = profile.copy()
            tile_profile.update(
                height=int(window.height),
                width=int(window.width),
                transform=transform,
            )
            tile_path = output_dir / f"{stem}_tile_{index:06d}.tif"
            data = dataset.read(bands, window=window, boundless=False)

            with rasterio.open(tile_path, "w", **tile_profile) as tile:
                tile.write(data)

    return len(windows)
