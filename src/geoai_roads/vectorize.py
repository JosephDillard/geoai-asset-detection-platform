from __future__ import annotations

import logging
import math
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_bounds
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

LOGGER = logging.getLogger(__name__)


def vectorize_masks(
    mask_dir: Path,
    output_path: Path,
    processing_crs: str,
    output_crs: str,
    min_area_m2: float,
    simplify_tolerance_m: float,
    smooth_tolerance_m: float = 0,
    rectangularize: bool = False,
    rectangularize_min_area_ratio: float = 0.9,
    dissolve_overlaps: bool = False,
    regularize: bool = False,
    regularize_tolerance_m: float = 0,
    regularize_angle_tolerance_degrees: float = 12,
    regularize_min_area_ratio: float = 0.7,
    regularize_max_area_ratio: float = 1.35,
    max_mask_coverage: float = 0,
    max_source_pixel_size_m: float = 0,
    class_name: str = "road",
) -> int:
    records = []

    for mask_path in sorted(mask_dir.glob("*.tif")):
        with rasterio.open(mask_path) as dataset:
            mask = dataset.read(1)
            if _skip_for_pixel_size(mask_path, dataset, processing_crs, max_source_pixel_size_m):
                continue
            if _skip_for_mask_coverage(mask_path, mask, max_mask_coverage):
                continue

            source_tile = dataset.tags().get("source_tile", mask_path.name)
            mask_class_name = dataset.tags().get("class_name", class_name)

            for geometry, value in shapes(mask, mask=mask == 1, transform=dataset.transform):
                if int(value) != 1:
                    continue
                records.append(
                    {
                        "source_tile": source_tile,
                        "class_name": mask_class_name,
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

    if rectangularize and not roads.empty:
        roads["geometry"] = roads.geometry.apply(
            lambda geometry: _rectangularize_geometry(geometry, rectangularize_min_area_ratio)
        )
        roads["geometry"] = roads.geometry.buffer(0)
        roads = roads[~roads.geometry.is_empty & roads.geometry.notnull()].copy()

    if roads.empty:
        empty = gpd.GeoDataFrame(
            columns=["source_tile", "class_name", "confidence", "geometry"],
            geometry="geometry",
            crs=output_crs,
        )
        _write_vector(empty, output_path)
        return 0

    if dissolve_overlaps:
        roads = _dissolve_overlapping_polygons(roads)
    elif not rectangularize:
        roads = roads.dissolve(by=["source_tile", "class_name"], as_index=False)

    if smooth_tolerance_m > 0 and not roads.empty:
        roads["geometry"] = roads.geometry.buffer(smooth_tolerance_m).buffer(-smooth_tolerance_m)
        roads = roads[~roads.geometry.is_empty & roads.geometry.notnull()].copy()

    if regularize and not roads.empty:
        roads["geometry"] = roads.geometry.apply(
            lambda geometry: _regularize_geometry(
                geometry=geometry,
                simplify_tolerance=regularize_tolerance_m,
                angle_tolerance_degrees=regularize_angle_tolerance_degrees,
                min_area_ratio=regularize_min_area_ratio,
                max_area_ratio=regularize_max_area_ratio,
            )
        )
        roads["geometry"] = roads.geometry.buffer(0)
        roads = roads[~roads.geometry.is_empty & roads.geometry.notnull()].copy()

    roads["geometry"] = roads.geometry.apply(_as_multipolygon)
    roads = roads.to_crs(output_crs)
    _write_vector(roads, output_path)
    return len(roads)


def _skip_for_pixel_size(
    mask_path: Path,
    dataset: rasterio.io.DatasetReader,
    processing_crs: str,
    max_source_pixel_size_m: float,
) -> bool:
    if max_source_pixel_size_m <= 0:
        return False
    if not dataset.crs:
        LOGGER.warning("Skipping %s because it has no CRS.", mask_path.name)
        return True

    pixel_size_m = _pixel_size_in_processing_crs(dataset, processing_crs)
    if pixel_size_m <= max_source_pixel_size_m:
        return False

    LOGGER.warning(
        "Skipping %s because source pixel size %.2fm exceeds %.2fm.",
        mask_path.name,
        pixel_size_m,
        max_source_pixel_size_m,
    )
    return True


def _skip_for_mask_coverage(
    mask_path: Path,
    mask,
    max_mask_coverage: float,
) -> bool:
    if max_mask_coverage <= 0:
        return False

    coverage = float((mask == 1).sum()) / float(mask.size) if mask.size else 0.0
    if coverage <= max_mask_coverage:
        return False

    LOGGER.warning(
        "Skipping %s because mask coverage %.1f%% exceeds %.1f%%.",
        mask_path.name,
        coverage * 100.0,
        max_mask_coverage * 100.0,
    )
    return True


def _pixel_size_in_processing_crs(
    dataset: rasterio.io.DatasetReader,
    processing_crs: str,
) -> float:
    left, bottom, right, top = transform_bounds(
        dataset.crs,
        processing_crs,
        *dataset.bounds,
        densify_pts=21,
    )
    pixel_width = abs(right - left) / max(dataset.width, 1)
    pixel_height = abs(top - bottom) / max(dataset.height, 1)
    return max(pixel_width, pixel_height)


def _as_multipolygon(geometry: Polygon | MultiPolygon) -> MultiPolygon:
    if isinstance(geometry, MultiPolygon):
        return geometry
    if isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    polygons = [part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon)]
    return MultiPolygon(polygons)


def _dissolve_overlapping_polygons(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    records = []
    for class_name, group in frame.groupby("class_name", dropna=False):
        merged = unary_union([geometry for geometry in group.geometry if geometry and not geometry.is_empty])
        for polygon in _iter_polygons(merged):
            records.append(
                {
                    "source_tile": "merged",
                    "class_name": class_name,
                    "confidence": None,
                    "geometry": polygon,
                }
            )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=frame.crs)


def _iter_polygons(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if not polygon.is_empty]
    return [
        polygon
        for part in getattr(geometry, "geoms", [])
        for polygon in _iter_polygons(part)
    ]


def _rectangularize_geometry(geometry, min_area_ratio: float):
    if geometry is None or geometry.is_empty:
        return geometry
    if isinstance(geometry, Polygon):
        return _rectangularize_polygon(geometry, min_area_ratio)
    if isinstance(geometry, MultiPolygon):
        polygons = [
            _rectangularize_polygon(part, min_area_ratio)
            for part in geometry.geoms
            if isinstance(part, Polygon) and not part.is_empty
        ]
        return MultiPolygon([polygon for polygon in polygons if not polygon.is_empty])
    polygons = [
        _rectangularize_polygon(part, min_area_ratio)
        for part in getattr(geometry, "geoms", [])
        if isinstance(part, Polygon) and not part.is_empty
    ]
    return MultiPolygon([polygon for polygon in polygons if not polygon.is_empty])


def _rectangularize_polygon(polygon: Polygon, min_area_ratio: float) -> Polygon:
    polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return polygon

    rectangle = polygon.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon) or rectangle.is_empty or rectangle.area <= 0:
        return polygon

    area_ratio = polygon.area / rectangle.area
    if area_ratio < min_area_ratio:
        return polygon
    return rectangle


def _regularize_geometry(
    geometry,
    simplify_tolerance: float,
    angle_tolerance_degrees: float,
    min_area_ratio: float,
    max_area_ratio: float,
):
    if geometry is None or geometry.is_empty:
        return geometry
    if isinstance(geometry, Polygon):
        return _regularize_polygon(
            geometry,
            simplify_tolerance,
            angle_tolerance_degrees,
            min_area_ratio,
            max_area_ratio,
        )
    polygons = [
        _regularize_polygon(
            polygon,
            simplify_tolerance,
            angle_tolerance_degrees,
            min_area_ratio,
            max_area_ratio,
        )
        for polygon in _iter_polygons(geometry)
    ]
    return MultiPolygon([polygon for polygon in polygons if polygon and not polygon.is_empty])


def _regularize_polygon(
    polygon: Polygon,
    simplify_tolerance: float,
    angle_tolerance_degrees: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> Polygon:
    original = polygon.buffer(0)
    if original.is_empty or original.area <= 0:
        return original

    working = original
    if simplify_tolerance > 0:
        working = working.simplify(simplify_tolerance, preserve_topology=True).buffer(0)
        if working.is_empty or working.area <= 0:
            return original

    angle = _dominant_polygon_angle(original)
    rotated = affinity.rotate(working, -angle, origin="centroid", use_radians=False)
    snapped = _snap_polygon_axis_edges(rotated, angle_tolerance_degrees)
    if snapped.is_empty:
        return working

    restored = affinity.rotate(snapped, angle, origin="centroid", use_radians=False).buffer(0)
    if restored.is_empty or restored.area <= 0:
        return working

    area_ratio = restored.area / original.area
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return working
    return restored


def _dominant_polygon_angle(polygon: Polygon) -> float:
    rectangle = polygon.minimum_rotated_rectangle
    if not isinstance(rectangle, Polygon) or rectangle.is_empty:
        return 0.0

    coords = list(rectangle.exterior.coords)
    longest_length = 0.0
    longest_angle = 0.0
    for start, end in zip(coords, coords[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > longest_length:
            longest_length = length
            longest_angle = math.degrees(math.atan2(dy, dx))
    return longest_angle


def _snap_polygon_axis_edges(polygon: Polygon, angle_tolerance_degrees: float) -> Polygon:
    exterior = _snap_ring_axis_edges(list(polygon.exterior.coords), angle_tolerance_degrees)
    if len(exterior) < 4:
        return polygon

    interiors = []
    for interior in polygon.interiors:
        snapped = _snap_ring_axis_edges(list(interior.coords), angle_tolerance_degrees)
        if len(snapped) >= 4:
            interiors.append(snapped)

    try:
        return Polygon(exterior, interiors).buffer(0)
    except Exception:
        return polygon


def _snap_ring_axis_edges(
    coords: list[tuple[float, float]],
    angle_tolerance_degrees: float,
) -> list[tuple[float, float]]:
    if len(coords) < 4:
        return coords

    tangent = math.tan(math.radians(angle_tolerance_degrees))
    snapped = [coords[0]]
    for x, y in coords[1:]:
        previous_x, previous_y = snapped[-1]
        dx = x - previous_x
        dy = y - previous_y
        if abs(dx) > 0 and abs(dy) <= abs(dx) * tangent:
            y = previous_y
        elif abs(dy) > 0 and abs(dx) <= abs(dy) * tangent:
            x = previous_x
        snapped.append((x, y))

    snapped[-1] = snapped[0]
    return _drop_consecutive_duplicate_coords(snapped)


def _drop_consecutive_duplicate_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    for coord in coords:
        if not result or coord != result[-1]:
            result.append(coord)
    if result and result[0] != result[-1]:
        result.append(result[0])
    return result


def _write_vector(frame: gpd.GeoDataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix in {".gpkg", ".geopackage"}:
        frame.to_file(output_path, layer="roads", driver="GPKG")
    elif suffix in {".geojson", ".json"}:
        frame.to_file(output_path, driver="GeoJSON")
    else:
        raise ValueError("Vector output must be .geojson or .gpkg")
