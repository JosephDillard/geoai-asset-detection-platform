from __future__ import annotations

from pathlib import Path

import click

from geoai_roads.config import load_config
from geoai_roads.inference import infer_tiles
from geoai_roads.postgis import load_vectors_to_postgis
from geoai_roads.tiling import extract_tiles
from geoai_roads.vectorize import vectorize_masks


@click.group()
def main() -> None:
    """Run the road detection pipeline."""


@main.command()
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
def tile(config_path: str) -> None:
    """Extract georeferenced tiles from the configured imagery."""
    config = load_config(config_path)
    count = extract_tiles(
        source=config.imagery_source,
        output_dir=config.tile_dir,
        bands=config.imagery_bands,
        tile_size=config.tile_size,
        overlap=config.tile_overlap,
    )
    click.echo(f"Extracted {count} tile(s) to {config.tile_dir}")


@main.command()
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
def infer(config_path: str) -> None:
    """Run ONNX road segmentation over extracted tiles."""
    config = load_config(config_path)
    count = infer_tiles(
        tile_dir=config.tile_dir,
        mask_dir=config.mask_dir,
        model_path=config.model_path,
        input_size=config.model_input_size,
        mean=config.model_mean,
        std=config.model_std,
        threshold=config.road_threshold,
        output_name=config.model_output_name,
    )
    click.echo(f"Wrote {count} road mask(s) to {config.mask_dir}")


@main.command()
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
def vectorize(config_path: str) -> None:
    """Convert road masks to GeoJSON or GeoPackage polygons."""
    config = load_config(config_path)
    count = vectorize_masks(
        mask_dir=config.mask_dir,
        output_path=config.vector_output,
        processing_crs=config.processing_crs,
        output_crs=config.output_crs,
        min_area_m2=config.min_area_m2,
        simplify_tolerance_m=config.simplify_tolerance_m,
    )
    click.echo(f"Wrote {count} road feature group(s) to {config.vector_output}")


@main.command("load-postgis")
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
@click.option(
    "--if-exists",
    type=click.Choice(["fail", "replace", "append"]),
    default="append",
    show_default=True,
)
def load_postgis(config_path: str, if_exists: str) -> None:
    """Load vectorized roads into PostGIS."""
    config = load_config(config_path)
    count = load_vectors_to_postgis(
        vector_path=config.vector_output,
        database_url=config.postgis_url,
        schema=config.postgis_schema,
        table=config.postgis_table,
        if_exists=if_exists,
    )
    click.echo(f"Loaded {count} road feature(s) into {config.postgis_schema}.{config.postgis_table}")


@main.command()
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
def run(config_path: str) -> None:
    """Run tile, infer, and vectorize in order."""
    config_file = Path(config_path)
    ctx = click.get_current_context()
    ctx.invoke(tile, config_path=str(config_file))
    ctx.invoke(infer, config_path=str(config_file))
    ctx.invoke(vectorize, config_path=str(config_file))


if __name__ == "__main__":
    main()
