CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS detected_roads (
    id bigserial PRIMARY KEY,
    job_id text,
    workflow_id text,
    source_tile text,
    class_name text NOT NULL DEFAULT 'road',
    confidence double precision,
    geom geometry(MultiPolygon, 4326),
    loaded_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS detected_roads_geom_idx
    ON detected_roads
    USING gist (geom);

CREATE INDEX IF NOT EXISTS detected_roads_job_id_idx
    ON detected_roads (job_id);

CREATE TABLE IF NOT EXISTS geoai_jobs (
    job_id text PRIMARY KEY,
    workflow_id text,
    status text NOT NULL DEFAULT 'loaded',
    feature_count bigint NOT NULL DEFAULT 0,
    loaded_at timestamptz NOT NULL DEFAULT now()
);
