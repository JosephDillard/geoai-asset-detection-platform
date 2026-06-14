from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, Polygon, shape


def vectorize_masks(
    mask_dir: Path,
    output_path: Path,
    processing_crs: str,
    output_crs: str,
    min_area_m2: float,
    simplify_tolerance_m: float,
    smooth_tolerance_m: float = 0,
) -> int:
    records = []

    for mask_path in sorted(mask_dir.glob("*.tif")):
        with rasterio.open(mask_path) as dataset:
            mask = dataset.read(1)
            source_tile = dataset.tags().get("source_tile", mask_path.name)

            for geometry, value in shapes(mask, mask=mask == 1, transform=dataset.transform):
                if int(value) != 1:
                    continue
                records.append(
                    {
                        "source_tile": source_tile,
                        "class_name": "road",
                        "confidence": None,
                        "geometry": shape(geometry),
                        "source_crs": dataset.crs,
                    }
                )

    if not records:
        empty = gpd.GeoDataFrame(
            columns=["source_tile", "class_name", "confidence", "geometry"],
            geometry="geometry",
            crs=output_crs,
        )
        _write_vector(empty, output_path)
        return 0

    source_crs = records[0].pop("source_crs")
    for record in records:
        record.pop("source_crs", None)

    roads = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs)
    roads = roads.to_crs(processing_crs)

    roads["geometry"] = roads.geometry.buffer(0)
    roads = roads[~roads.geometry.is_empty & roads.geometry.notnull()].copy()

    if min_area_m2 > 0:
        roads = roads[roads.geometry.area >= min_area_m2].copy()

    if simplify_tolerance_m > 0 and not roads.empty:
        roads["geometry"] = roads.geometry.simplify(simplify_tolerance_m, preserve_topology=True)

    if roads.empty:
        empty = gpd.GeoDataFrame(
            columns=["source_tile", "class_name", "confidence", "geometry"],
            geometry="geometry",
            crs=output_crs,
        )
        _write_vector(empty, output_path)
        return 0

    roads = roads.dissolve(by=["source_tile", "class_name"], as_index=False)

    if smooth_tolerance_m > 0 and not roads.empty:
        roads["geometry"] = roads.geometry.buffer(smooth_tolerance_m).buffer(-smooth_tolerance_m)
        roads = roads[~roads.geometry.is_empty & roads.geometry.notnull()].copy()

    roads["geometry"] = roads.geometry.apply(_as_multipolygon)
    roads = roads.to_crs(output_crs)
    _write_vector(roads, output_path)
    return len(roads)


def _as_multipolygon(geometry: Polygon | MultiPolygon) -> MultiPolygon:
    if isinstance(geometry, MultiPolygon):
        return geometry
    if isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    polygons = [part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon)]
    return MultiPolygon(polygons)


def _write_vector(frame: gpd.GeoDataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix in {".gpkg", ".geopackage"}:
        frame.to_file(output_path, layer="roads", driver="GPKG")
    elif suffix in {".geojson", ".json"}:
        frame.to_file(output_path, driver="GeoJSON")
    else:
        raise ValueError("Vector output must be .geojson or .gpkg")
