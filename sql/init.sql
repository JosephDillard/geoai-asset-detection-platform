CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS detected_roads (
    id bigserial PRIMARY KEY,
    source_tile text,
    class_name text NOT NULL DEFAULT 'road',
    confidence double precision,
    geom geometry(MultiPolygon, 3857),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS detected_roads_geom_idx
    ON detected_roads
    USING gist (geom);
