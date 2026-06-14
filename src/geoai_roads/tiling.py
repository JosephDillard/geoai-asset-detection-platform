from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box, shape
from shapely.ops import unary_union


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
    bbox: list[float] | None = None,
    aoi_geojson: dict[str, Any] | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source) as dataset:
        stem = source.stem
        _clear_matching(output_dir, f"{stem}_tile_*.tif")

        windows = iter_windows(dataset.width, dataset.height, tile_size, overlap)
        area = _map_context_geometry(dataset, bbox, aoi_geojson)
        if area is not None:
            windows = [
                window
                for window in windows
                if box(*window_bounds(window, dataset.transform)).intersects(area)
            ]

        profile = dataset.profile.copy()
        profile.update(driver="GTiff", tiled=True, compress="deflate", count=len(bands))

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


def _clear_matching(output_dir: Path, pattern: str) -> None:
    for path in output_dir.glob(pattern):
        if path.is_file():
            path.unlink()


def _map_context_geometry(
    dataset: rasterio.io.DatasetReader,
    bbox: list[float] | None,
    aoi_geojson: dict[str, Any] | None,
):
    geometry = _geojson_geometry(aoi_geojson) if aoi_geojson else None
    if geometry is None and bbox:
        if len(bbox) != 4:
            raise ValueError("bbox must be [west, south, east, north]")
        geometry = box(*bbox)

    if geometry is None or geometry.is_empty:
        return None

    if not dataset.crs:
        return geometry

    if aoi_geojson:
        transformed = transform_geom("EPSG:4326", dataset.crs, geometry.__geo_interface__)
        return shape(transformed)

    west, south, east, north = geometry.bounds
    return box(*transform_bounds("EPSG:4326", dataset.crs, west, south, east, north))


def _geojson_geometry(value: dict[str, Any] | None):
    if not value:
        return None

    geojson_type = value.get("type")
    if geojson_type == "FeatureCollection":
        geometries = [
            shape(feature["geometry"])
            for feature in value.get("features", [])
            if feature.get("geometry")
        ]
        return unary_union(geometries) if geometries else None
    if geojson_type == "Feature":
        geometry = value.get("geometry")
        return shape(geometry) if geometry else None
    return shape(value)
