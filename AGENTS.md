# Agent Guide

This repo is the GeoAI workflow layer in the companion geospatial stack.
It turns imagery into GIS-ready vector detections and can publish results to
PostGIS for the status-board map.

## Stack Context

Related sibling repos:

- `geospatial-status-board` visualizes PostGIS/GeoServer layers in a Grails and
  MapLibre web app.
- `geospatial-data-gateway` ingests geospatial files into PostGIS and can notify
  map clients when layers refresh.
- `geospatial-mcp-services` provides map-aware assistant tools, starting with a
  GeoNames/Wikipedia MCP server.

When changing cross-repo docs or integration instructions, prefer full GitHub
URLs for links that point outside this repo.

## What This Repo Owns

- Python package: `src/geoai_roads/`
- Workflow configuration: `config/*.yaml`
- Local helpers and data scripts: `scripts/`
- SQL/PostGIS helpers: `sql/`
- Tests: `tests/`
- Ignored local artifacts: `data/`, `models/`, `outputs/`, and `logs/`

The main pipeline stages are tiling, model inference, mask cleanup,
vectorization, optional quality filtering, and optional PostGIS loading.

## Development Notes

- Keep large imagery, downloaded models, generated masks, vectors, and logs out
  of Git.
- Keep TensorFlow/Keras and PyTorch support optional. The default dev path should
  still work without installing every model backend.
- Use EPSG:4326 for web-map-ready vector outputs unless a workflow explicitly
  documents another CRS.
- The status-board Docker `geoai` profile expects this repo to live beside it as
  `../geoai-asset-detection-platform`.
- Do not treat demo detections as authoritative analysis; they are for pipeline
  and map integration testing.

## Useful Commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests
```

For optional model backends:

```powershell
python -m pip install -e ".[dev,keras]"
python -m pip install -e ".[dev,pytorch]"
```

For local demo assets:

```powershell
python scripts\create_demo_assets.py
```

## Before Finishing Changes

- Run the focused tests for touched pipeline areas.
- Run the full `python -m pytest` suite when changing shared pipeline behavior.
- Update `README.md` or workflow examples when commands, model expectations,
  output tables, or integration contracts change.
