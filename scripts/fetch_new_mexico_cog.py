from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import rasterio
from rasterio.shutil import copy as copy_raster
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]
STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DEFAULT_BBOX = [-106.78, 34.90, -106.58, 35.10]
DEFAULT_OUTPUT = ROOT / "data" / "imagery" / "new-mexico-naip-abq-cog.tif"


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    feature = find_naip_feature(args.bbox, args.item_id)
    image_href = feature["assets"]["image"]["href"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_cog_subset(
        source_href=image_href,
        output_path=output_path,
        crop_size=args.crop_size,
        bands=args.bands,
        source_item_id=feature["id"],
    )

    print(f"Wrote {output_path}")
    print(f"Source item: {feature['id']}")
    print(f"Source COG: {image_href}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a small New Mexico NAIP Cloud Optimized GeoTIFF test image."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=DEFAULT_BBOX,
        help="Longitude/latitude bounding box used for the NAIP STAC search.",
    )
    parser.add_argument(
        "--item-id",
        default="",
        help="Optional exact NAIP STAC item id to select from the search results.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=3072,
        help="Square crop size in source pixels.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Source bands to write to the local COG.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output COG path.",
    )
    return parser.parse_args()


def find_naip_feature(bbox: list[float], item_id: str = "") -> dict:
    payload = {
        "collections": ["naip"],
        "bbox": bbox,
        "limit": 20,
    }
    request = Request(
        STAC_SEARCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        data = json.load(response)

    features = data.get("features") or []
    if item_id:
        features = [feature for feature in features if feature.get("id") == item_id]
    if not features:
        raise RuntimeError("No NAIP COG items found for the requested New Mexico search area.")

    return features[0]


def write_cog_subset(
    source_href: str,
    output_path: Path,
    crop_size: int,
    bands: list[int],
    source_item_id: str,
) -> None:
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    with rasterio.open(source_href) as source:
        width = min(crop_size, source.width)
        height = min(crop_size, source.height)
        col_off = max((source.width - width) // 2, 0)
        row_off = max((source.height - height) // 2, 0)
        window = Window(col_off, row_off, width, height)
        data = source.read(bands, window=window)

        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            count=len(bands),
            transform=source.window_transform(window),
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
            photometric="rgb" if len(bands) >= 3 else "minisblack",
        )

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "new-mexico-naip-subset.tif"
        with rasterio.open(temp_path, "w", **profile) as dataset:
            dataset.write(data)
            dataset.update_tags(
                source_collection="naip",
                source_item=source_item_id,
                source_href=source_href,
            )

        if output_path.exists():
            output_path.unlink()

        copy_raster(
            temp_path,
            output_path,
            driver="COG",
            compress="deflate",
            blocksize=512,
            overview_resampling="nearest",
        )


if __name__ == "__main__":
    main()
