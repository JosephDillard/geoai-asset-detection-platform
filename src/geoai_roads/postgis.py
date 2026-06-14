from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

from geoalchemy2 import Geometry
import geopandas as gpd
from sqlalchemy import create_engine, text

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_vectors_to_postgis(
    vector_path: Path,
    database_url: str,
    schema: str,
    table: str,
    if_exists: str = "append",
    job_id: str | None = None,
    metadata: dict[str, str] | None = None,
) -> int:
    frame = gpd.read_file(vector_path)
    if frame.crs is None:
        raise ValueError("Vector output must define a CRS before loading into PostGIS.")

    _validate_identifier(schema, "schema")
    _validate_identifier(table, "table")

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        if frame.empty:
            return 0

    frame = frame.rename_geometry("geom")
    if job_id:
        frame["job_id"] = job_id
    for key, value in (metadata or {}).items():
        _validate_identifier(key, "metadata column")
        frame[key] = value
    if job_id or metadata:
        frame["loaded_at"] = datetime.now(timezone.utc).isoformat()

    geometry_dtype = Geometry("GEOMETRY", srid=frame.crs.to_epsg() or -1)

    frame.to_postgis(
        table,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        dtype={"geom": geometry_dtype},
    )
    return len(frame)


def _validate_identifier(value: str, label: str) -> None:
    if not IDENTIFIER_PATTERN.match(value):
        raise ValueError(f"Invalid PostGIS {label} name: {value!r}")
