from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import os
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from geoai_roads.orchestrator import (
    MAX_WORKFLOWS,
    WorkflowDefinition,
    WorkflowResult,
    load_workflow_catalog,
    normalize_road_stages,
    run_workflow,
)

DEFAULT_CATALOG = "config/workflows.example.yaml"
DEFAULT_CORS_ORIGINS = "http://localhost:8080,http://127.0.0.1:8080"


class RequestSource(str, Enum):
    manual = "manual"
    external_app = "external_app"


class RoadStage(str, Enum):
    tile = "tile"
    infer = "infer"
    vectorize = "vectorize"
    load_postgis = "load-postgis"


class MapContext(BaseModel):
    source_app: str | None = Field(
        default=None,
        description="App that captured the map context, for example geospatial-status-board.",
    )
    aoi_geojson: dict[str, Any] | None = Field(
        default=None,
        description="Optional GeoJSON geometry or FeatureCollection drawn in the map viewer.",
    )
    bbox: list[float] | None = Field(
        default=None,
        description="Optional [west, south, east, north] bounds in EPSG:4326.",
        min_length=4,
        max_length=4,
    )
    map_center: list[float] | None = Field(
        default=None,
        description="Optional [longitude, latitude] map center in EPSG:4326.",
        min_length=2,
        max_length=2,
    )
    zoom: float | None = Field(
        default=None,
        description="Optional MapLibre zoom level when the request was submitted.",
    )
    selected_layer: str | None = Field(
        default=None,
        description="Optional status-board layer key active when the request was submitted.",
    )
    selected_feature_ids: list[str] | None = Field(
        default=None,
        description="Optional selected feature ids from the map viewer.",
    )

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_time(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv("GEOAI_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class RunRequest(BaseModel):
    request_source: RequestSource = Field(
        default=RequestSource.manual,
        description="Where the request came from.",
    )
    submitted_by: str | None = Field(
        default=None,
        description="Optional user, service, or app name submitting the request.",
    )
    external_request_id: str | None = Field(
        default=None,
        description="Optional id supplied by an upstream app for traceability.",
    )
    catalog_path: str | None = Field(
        default=None,
        description="Workflow catalog path. Defaults to the server catalog.",
    )
    workflow_ids: list[str] | None = Field(
        default=None,
        description="Workflow ids to run. Omit to run enabled workflows from the catalog.",
    )
    stages: list[RoadStage] | None = Field(
        default=None,
        description="Optional stage override for every selected workflow.",
    )
    map_context: MapContext | None = Field(
        default=None,
        description="Optional MapLibre viewer context such as drawn AOI, bbox, or selected layer.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional operator or upstream app notes for this run.",
    )


@dataclass
class RunRecord:
    id: str
    status: str
    request_source: str
    submitted_by: str | None
    external_request_id: str | None
    notes: str | None
    map_context: dict[str, Any] | None
    catalog_path: str
    requested_workflows: list[str]
    stages_override: list[str] | None
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    results: list[WorkflowResult] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "request_source": self.request_source,
            "submitted_by": self.submitted_by,
            "external_request_id": self.external_request_id,
            "notes": self.notes,
            "map_context": self.map_context,
            "catalog_path": self.catalog_path,
            "requested_workflows": self.requested_workflows,
            "stages_override": self.stages_override,
            "created_at": _format_time(self.created_at),
            "started_at": _format_time(self.started_at),
            "finished_at": _format_time(self.finished_at),
            "results": [result.as_dict() for result in self.results],
            "error": self.error,
        }


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = Lock()

    def create(
        self,
        catalog_path: Path,
        workflows: list[WorkflowDefinition],
        stages_override: list[str] | None,
        request: RunRequest,
    ) -> RunRecord:
        record = RunRecord(
            id=uuid4().hex,
            status="queued",
            request_source=request.request_source.value,
            submitted_by=request.submitted_by,
            external_request_id=request.external_request_id,
            notes=request.notes,
            map_context=request.map_context.as_dict() if request.map_context else None,
            catalog_path=str(catalog_path),
            requested_workflows=[workflow.id for workflow in workflows],
            stages_override=stages_override,
        )
        with self._lock:
            self._runs[record.id] = record
        return record

    def update(self, run_id: str, **changes: Any) -> RunRecord:
        with self._lock:
            record = self._get_locked(run_id)
            for key, value in changes.items():
                setattr(record, key, value)
            return record

    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            return self._get_locked(run_id)

    def list(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda record: record.created_at, reverse=True)

    def _get_locked(self, run_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record


def create_app(
    default_catalog: str | Path = DEFAULT_CATALOG,
    max_workers: int = 2,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    app = FastAPI(
        title="GeoAI Workflow API",
        version="0.1.0",
        description=f"Run up to {MAX_WORKFLOWS} configured GeoAI workflows from YAML catalogs.",
    )
    origins = allowed_origins if allowed_origins is not None else _cors_origins_from_env()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.state.default_catalog = str(default_catalog)
    app.state.run_store = RunStore()
    app.state.executor = ThreadPoolExecutor(max_workers=max_workers)

    @app.on_event("shutdown")
    def _shutdown_executor() -> None:
        app.state.executor.shutdown(wait=False, cancel_futures=True)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/workflows")
    def list_workflows(catalog_path: str | None = None) -> dict[str, Any]:
        catalog = _load_catalog_or_400(catalog_path or app.state.default_catalog)
        return catalog.as_dict()

    @app.get("/run-options")
    def run_options(catalog_path: str | None = None) -> dict[str, Any]:
        catalog = _load_catalog_or_400(catalog_path or app.state.default_catalog)
        return {
            "request_sources": [source.value for source in RequestSource],
            "stages": [stage.value for stage in RoadStage],
            "max_workflows": MAX_WORKFLOWS,
            "workflows": [
                {
                    "id": workflow.id,
                    "name": workflow.name,
                    "enabled": workflow.enabled,
                    "type": workflow.workflow_type,
                    "default_stages": list(workflow.stages),
                }
                for workflow in catalog.workflows
            ],
        }

    @app.post("/runs", status_code=202)
    def create_run(request: RunRequest) -> dict[str, Any]:
        catalog = _load_catalog_or_400(request.catalog_path or app.state.default_catalog)
        try:
            workflows = catalog.select(request.workflow_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            stages_override = (
                list(normalize_road_stages(stage.value for stage in request.stages))
                if request.stages
                else None
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        store: RunStore = app.state.run_store
        record = store.create(catalog.path, workflows, stages_override, request)
        app.state.executor.submit(_execute_run, store, record.id, workflows, stages_override)
        return record.as_dict()

    @app.get("/runs")
    def list_runs() -> dict[str, Any]:
        store: RunStore = app.state.run_store
        return {"runs": [record.as_dict() for record in store.list()]}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        store: RunStore = app.state.run_store
        try:
            return store.get(run_id).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc

    return app


app = create_app()


def _execute_run(
    store: RunStore,
    run_id: str,
    workflows: list[WorkflowDefinition],
    stages_override: list[str] | None,
) -> None:
    store.update(run_id, status="running", started_at=_now())
    try:
        results = [run_workflow(workflow, stages_override) for workflow in workflows]
        status = "failed" if any(result.status == "failed" for result in results) else "succeeded"
        store.update(run_id, status=status, results=results, finished_at=_now())
    except Exception as exc:
        store.update(run_id, status="failed", error=str(exc), finished_at=_now())


def _load_catalog_or_400(path: str | Path):
    try:
        return load_workflow_catalog(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
