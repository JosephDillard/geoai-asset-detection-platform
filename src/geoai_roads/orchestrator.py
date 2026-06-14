from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from geoai_roads.config import RoadConfig, load_config
from geoai_roads.inference import infer_tiles
from geoai_roads.postgis import load_vectors_to_postgis
from geoai_roads.tiling import extract_tiles
from geoai_roads.vectorize import vectorize_masks

MAX_WORKFLOWS = 10
ROAD_STAGES = ("tile", "infer", "vectorize", "load-postgis")
DEFAULT_ROAD_STAGES = ("tile", "infer", "vectorize")
POSTGIS_IF_EXISTS = ("fail", "replace", "append")


@dataclass(frozen=True)
class StageResult:
    stage: str
    count: int
    message: str
    output_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "count": self.count,
            "message": self.message,
            "output_path": str(self.output_path) if self.output_path else None,
        }


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    name: str
    workflow_type: str
    config_path: Path
    enabled: bool = True
    stages: tuple[str, ...] = DEFAULT_ROAD_STAGES
    postgis_if_exists: str = "append"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.workflow_type,
            "config_path": str(self.config_path),
            "enabled": self.enabled,
            "stages": list(self.stages),
            "postgis_if_exists": self.postgis_if_exists,
        }


@dataclass(frozen=True)
class WorkflowResult:
    workflow_id: str
    name: str
    status: str
    stages: list[StageResult]
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status,
            "stages": [stage.as_dict() for stage in self.stages],
            "error": self.error,
        }


@dataclass(frozen=True)
class WorkflowCatalog:
    path: Path
    workflows: tuple[WorkflowDefinition, ...]

    def select(self, workflow_ids: Iterable[str] | None = None) -> list[WorkflowDefinition]:
        if workflow_ids:
            workflow_map = {workflow.id: workflow for workflow in self.workflows}
            selected = []
            missing = []
            for workflow_id in workflow_ids:
                workflow = workflow_map.get(workflow_id)
                if workflow is None:
                    missing.append(workflow_id)
                else:
                    selected.append(workflow)
            if missing:
                raise ValueError(f"Unknown workflow id(s): {', '.join(missing)}")
        else:
            selected = [workflow for workflow in self.workflows if workflow.enabled]

        if not selected:
            raise ValueError("No workflows selected")
        if len(selected) > MAX_WORKFLOWS:
            raise ValueError(f"A run can include at most {MAX_WORKFLOWS} workflows")
        return selected

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "max_workflows": MAX_WORKFLOWS,
            "workflows": [workflow.as_dict() for workflow in self.workflows],
        }


def load_workflow_catalog(path: str | Path) -> WorkflowCatalog:
    catalog_path = Path(path).resolve()
    with catalog_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    workflow_items = raw.get("workflows")
    if not isinstance(workflow_items, list):
        raise ValueError("Workflow catalog must contain a 'workflows' list")
    if len(workflow_items) > MAX_WORKFLOWS:
        raise ValueError(f"Workflow catalog can define at most {MAX_WORKFLOWS} workflows")

    workflows = tuple(_parse_workflow(item, catalog_path.parent) for item in workflow_items)
    workflow_ids = [workflow.id for workflow in workflows]
    duplicates = sorted(
        {workflow_id for workflow_id in workflow_ids if workflow_ids.count(workflow_id) > 1}
    )
    if duplicates:
        raise ValueError(f"Duplicate workflow id(s): {', '.join(duplicates)}")

    return WorkflowCatalog(path=catalog_path, workflows=workflows)


def run_road_stage(
    config: RoadConfig,
    stage: str,
    postgis_if_exists: str = "append",
) -> StageResult:
    if stage == "tile":
        count = extract_tiles(
            source=config.imagery_source,
            output_dir=config.tile_dir,
            bands=config.imagery_bands,
            tile_size=config.tile_size,
            overlap=config.tile_overlap,
        )
        return StageResult(
            stage,
            count,
            f"Extracted {count} tile(s) to {config.tile_dir}",
            config.tile_dir,
        )

    if stage == "infer":
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
        return StageResult(
            stage,
            count,
            f"Wrote {count} road mask(s) to {config.mask_dir}",
            config.mask_dir,
        )

    if stage == "vectorize":
        count = vectorize_masks(
            mask_dir=config.mask_dir,
            output_path=config.vector_output,
            processing_crs=config.processing_crs,
            output_crs=config.output_crs,
            min_area_m2=config.min_area_m2,
            simplify_tolerance_m=config.simplify_tolerance_m,
        )
        return StageResult(
            stage,
            count,
            f"Wrote {count} road feature group(s) to {config.vector_output}",
            config.vector_output,
        )

    if stage == "load-postgis":
        count = load_vectors_to_postgis(
            vector_path=config.vector_output,
            database_url=config.postgis_url,
            schema=config.postgis_schema,
            table=config.postgis_table,
            if_exists=postgis_if_exists,
        )
        return StageResult(
            stage,
            count,
            f"Loaded {count} road feature(s) into {config.postgis_schema}.{config.postgis_table}",
        )

    raise ValueError(f"Unsupported road stage: {stage}")


def run_road_pipeline(
    config_path: str | Path,
    stages: Iterable[str] | None = None,
    postgis_if_exists: str = "append",
) -> list[StageResult]:
    selected_stages = normalize_road_stages(stages or DEFAULT_ROAD_STAGES)
    _validate_postgis_if_exists(postgis_if_exists)

    config = load_config(config_path)
    return [run_road_stage(config, stage, postgis_if_exists) for stage in selected_stages]


def run_workflow(
    workflow: WorkflowDefinition,
    stages: Iterable[str] | None = None,
) -> WorkflowResult:
    stage_results: list[StageResult] = []
    try:
        if workflow.workflow_type != "roads":
            raise ValueError(f"Unsupported workflow type: {workflow.workflow_type}")

        selected_stages = normalize_road_stages(stages or workflow.stages)
        _validate_postgis_if_exists(workflow.postgis_if_exists)
        config = load_config(workflow.config_path)

        for stage in selected_stages:
            stage_results.append(run_road_stage(config, stage, workflow.postgis_if_exists))

        return WorkflowResult(
            workflow_id=workflow.id,
            name=workflow.name,
            status="succeeded",
            stages=stage_results,
        )
    except Exception as exc:
        return WorkflowResult(
            workflow_id=workflow.id,
            name=workflow.name,
            status="failed",
            stages=stage_results,
            error=str(exc),
        )


def run_workflow_catalog(
    catalog_path: str | Path,
    workflow_ids: Iterable[str] | None = None,
    stages: Iterable[str] | None = None,
) -> list[WorkflowResult]:
    catalog = load_workflow_catalog(catalog_path)
    selected_workflows = catalog.select(workflow_ids)
    return [run_workflow(workflow, stages) for workflow in selected_workflows]


def _parse_workflow(item: Any, catalog_dir: Path) -> WorkflowDefinition:
    if not isinstance(item, dict):
        raise ValueError("Each workflow must be a mapping")

    workflow_id = str(item.get("id") or "").strip()
    if not workflow_id:
        raise ValueError("Each workflow must define an id")

    workflow_type = str(item.get("type", "roads")).strip()
    if workflow_type != "roads":
        raise ValueError(f"Unsupported workflow type for {workflow_id}: {workflow_type}")

    config_value = item.get("config")
    if not config_value:
        raise ValueError(f"Workflow {workflow_id} must define a config path")

    postgis = item.get("postgis") or {}
    if not isinstance(postgis, dict):
        raise ValueError(f"Workflow {workflow_id} postgis settings must be a mapping")

    postgis_if_exists = str(postgis.get("if_exists", item.get("postgis_if_exists", "append")))
    _validate_postgis_if_exists(postgis_if_exists)

    stages = normalize_road_stages(item.get("stages") or DEFAULT_ROAD_STAGES)
    return WorkflowDefinition(
        id=workflow_id,
        name=str(item.get("name", workflow_id)),
        workflow_type=workflow_type,
        config_path=_resolve_relative_path(config_value, catalog_dir),
        enabled=bool(item.get("enabled", True)),
        stages=stages,
        postgis_if_exists=postgis_if_exists,
    )


def normalize_road_stages(stages: Iterable[str]) -> tuple[str, ...]:
    if isinstance(stages, (str, bytes)):
        raise ValueError("stages must be a list of stage names")

    selected = tuple(str(stage) for stage in stages)
    invalid = [stage for stage in selected if stage not in ROAD_STAGES]
    if invalid:
        raise ValueError(f"Unsupported road stage(s): {', '.join(invalid)}")
    return selected


def _validate_postgis_if_exists(value: str) -> None:
    if value not in POSTGIS_IF_EXISTS:
        raise ValueError(f"postgis if_exists must be one of: {', '.join(POSTGIS_IF_EXISTS)}")


def _resolve_relative_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
