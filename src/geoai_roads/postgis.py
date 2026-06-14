from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

from geoalchemy2 import Geometry
import geopandas as gpd
from sqlalchemy import create_engine, inspect, text

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
        _ensure_job_table(connection, schema)
        if frame.empty:
            _record_geoai_job(
                connection=connection,
                schema=schema,
                job_id=job_id,
                metadata=metadata,
                feature_count=0,
            )
            return 0

    frame = frame.rename_geometry("geom")
    if job_id:
        frame["job_id"] = job_id
    for key, value in (metadata or {}).items():
        _validate_identifier(key, "metadata column")
        frame[key] = value
    if job_id or metadata:
        frame["loaded_at"] = datetime.now(timezone.utc)

    geometry_dtype = Geometry("GEOMETRY", srid=frame.crs.to_epsg() or -1)

    if if_exists == "append":
        with engine.begin() as connection:
            _ensure_append_columns(connection, schema, table, frame.columns)

    frame.to_postgis(
        table,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        dtype={"geom": geometry_dtype},
    )
    with engine.begin() as connection:
        _ensure_job_table(connection, schema)
        _record_geoai_job(
            connection=connection,
            schema=schema,
            job_id=job_id,
            metadata=metadata,
            feature_count=len(frame),
        )
    return len(frame)


def _validate_identifier(value: str, label: str) -> None:
    if not IDENTIFIER_PATTERN.match(value):
        raise ValueError(f"Invalid PostGIS {label} name: {value!r}")


def _ensure_append_columns(connection, schema: str, table: str, columns) -> None:
    inspector = inspect(connection)
    if not inspector.has_table(table, schema=schema):
        return

    existing = {column["name"] for column in inspector.get_columns(table, schema=schema)}
    text_columns = ["job_id", "workflow_id"]
    for column in text_columns:
        if column in columns and column not in existing:
            connection.execute(text(f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS {column} text"))
    if "loaded_at" in columns and "loaded_at" not in existing:
        connection.execute(
            text(f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS loaded_at timestamptz")
        )
    elif "loaded_at" in existing:
        loaded_at_column = next(
            (column for column in inspector.get_columns(table, schema=schema) if column["name"] == "loaded_at"),
            None,
        )
        loaded_at_type = str((loaded_at_column or {}).get("type", "")).upper()
        if "TIMESTAMP" not in loaded_at_type:
            connection.execute(
                text(
                    f"ALTER TABLE {schema}.{table} ALTER COLUMN loaded_at "
                    "TYPE timestamptz USING NULLIF(loaded_at::text, '')::timestamptz"
                )
            )
    if "created_at" not in existing:
        connection.execute(
            text(
                f"ALTER TABLE {schema}.{table} "
                "ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now()"
            )
        )
    if "job_id" in columns or "job_id" in existing:
        connection.execute(
            text(f"CREATE INDEX IF NOT EXISTS {table}_job_id_idx ON {schema}.{table} (job_id)")
        )


def _ensure_job_table(connection, schema: str) -> None:
    connection.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.geoai_jobs (
                job_id text PRIMARY KEY,
                workflow_id text,
                status text NOT NULL DEFAULT 'loaded',
                feature_count bigint NOT NULL DEFAULT 0,
                loaded_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )


def _record_geoai_job(
    connection,
    schema: str,
    job_id: str | None,
    metadata: dict[str, str] | None,
    feature_count: int,
) -> None:
    if not job_id:
        return

    workflow_id = (metadata or {}).get("workflow_id")
    connection.execute(
        text(
            f"""
            INSERT INTO {schema}.geoai_jobs AS jobs
                (job_id, workflow_id, status, feature_count, loaded_at)
            VALUES
                (:job_id, :workflow_id, 'loaded', :feature_count, :loaded_at)
            ON CONFLICT (job_id) DO UPDATE SET
                workflow_id = COALESCE(EXCLUDED.workflow_id, jobs.workflow_id),
                status = EXCLUDED.status,
                feature_count = jobs.feature_count + EXCLUDED.feature_count,
                loaded_at = EXCLUDED.loaded_at
            """
        ),
        {
            "job_id": job_id,
            "workflow_id": workflow_id,
            "feature_count": feature_count,
            "loaded_at": datetime.now(timezone.utc),
        },
    )
