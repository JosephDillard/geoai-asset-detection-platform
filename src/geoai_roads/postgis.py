from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
from sqlalchemy import create_engine, text

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_vectors_to_postgis(
    vector_path: Path,
    database_url: str,
    schema: str,
    table: str,
    if_exists: str = "append",
) -> int:
    roads = gpd.read_file(vector_path)
    if roads.empty:
        return 0

    _validate_identifier(schema, "schema")
    _validate_identifier(table, "table")

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

    roads.to_postgis(table, engine, schema=schema, if_exists=if_exists, index=False)
    return len(roads)


def _validate_identifier(value: str, label: str) -> None:
    if not IDENTIFIER_PATTERN.match(value):
        raise ValueError(f"Invalid PostGIS {label} name: {value!r}")
