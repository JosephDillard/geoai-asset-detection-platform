from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

from geoai_roads.postgis import load_vectors_to_postgis


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGERY_DIR = ROOT / "data" / "imagery"
DEFAULT_OUTPUT = ROOT / "outputs" / "cog_footprints.gpkg"
DEFAULT_DATABASE_URL = "postgresql+psycopg://gsb:gsb@localhost:5432/geostatusboard"


def main() -> None:
    args = parse_args()
    imagery_dir = Path(args.imagery_dir).resolve()
    output_path = Path(args.output).resolve()

    footprints = build_footprints(imagery_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    footprints.to_file(output_path, layer="cog_footprints", driver="GPKG")
    print(f"Wrote {len(footprints)} COG footprint(s) to {output_path}")

    if args.no_postgis:
        return

    count = load_vectors_to_postgis(
        output_path,
        database_url=args.database_url,
        schema=args.schema,
        table=args.table,
        if_exists=args.if_exists,
    )
    print(f"Loaded {count} COG footprint(s) into {args.schema}.{args.table}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GeoAI COG footprints and optionally load them into PostGIS."
    )
    parser.add_argument("--imagery-dir", default=str(DEFAULT_IMAGERY_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument("--schema", default="public")
    parser.add_argument("--table", default="geoai_cog_footprints")
    parser.add_argument(
        "--if-exists",
        choices=["fail", "replace", "append"],
        default="replace",
    )
    parser.add_argument("--no-postgis", action="store_true")
    return parser.parse_args()


def build_footprints(imagery_dir: Path) -> gpd.GeoDataFrame:
    paths = sorted(
        path
        for pattern in ("*.tif", "*.tiff")
        for path in imagery_dir.glob(pattern)
        if path.is_file()
    )
    if not paths:
        raise RuntimeError(f"No GeoTIFF imagery found in {imagery_dir}")

    records = []
    for path in paths:
        with rasterio.open(path) as dataset:
            if not dataset.crs:
                raise RuntimeError(f"Imagery has no CRS: {path}")
            bounds = transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21)
            tags = dataset.tags()
            resolution = dataset.res
            records.append(
                {
                    "cog_id": path.stem,
                    "file_name": path.name,
                    "file_path": _relative_path(path),
                    "source_collection": tags.get("source_collection", ""),
                    "source_item": tags.get("source_item", ""),
                    "source_href": tags.get("source_href", ""),
                    "native_crs": str(dataset.crs),
                    "width": int(dataset.width),
                    "height": int(dataset.height),
                    "band_count": int(dataset.count),
                    "pixel_size_x": float(resolution[0]),
                    "pixel_size_y": float(abs(resolution[1])),
                    "geometry": box(*bounds),
                }
            )

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
