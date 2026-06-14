from __future__ import annotations

from pathlib import Path

import click

from geoai_roads.config import load_config
from geoai_roads.orchestrator import (
    ROAD_STAGES,
    StageResult,
    WorkflowResult,
    run_road_pipeline,
    run_road_stage,
    run_workflow_catalog,
)


@click.group()
def main() -> None:
    """Run the road detection pipeline."""


@main.command()
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
def tile(config_path: str) -> None:
    """Extract georeferenced tiles from the configured imagery."""
    config = load_config(config_path)
    _echo_stage_result(run_road_stage(config, "tile"))


@main.command()
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
def infer(config_path: str) -> None:
    """Run ONNX road segmentation over extracted tiles."""
    config = load_config(config_path)
    _echo_stage_result(run_road_stage(config, "infer"))


@main.command()
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
def vectorize(config_path: str) -> None:
    """Convert road masks to GeoJSON or GeoPackage polygons."""
    config = load_config(config_path)
    _echo_stage_result(run_road_stage(config, "vectorize"))


@main.command("load-postgis")
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
@click.option(
    "--if-exists",
    type=click.Choice(["fail", "replace", "append"]),
    default="append",
    show_default=True,
)
@click.option("--job-id", default=None, help="Optional job id written to loaded features.")
def load_postgis(config_path: str, if_exists: str, job_id: str | None) -> None:
    """Load vectorized roads into PostGIS."""
    config = load_config(config_path)
    _echo_stage_result(run_road_stage(config, "load-postgis", if_exists, job_id=job_id))


@main.command()
@click.option("--config", "config_path", default="config/roads.example.yaml", show_default=True)
def run(config_path: str) -> None:
    """Run tile, infer, and vectorize in order."""
    config_file = Path(config_path)
    for result in run_road_pipeline(config_file):
        _echo_stage_result(result)


@main.command("run-workflows")
@click.option(
    "--catalog",
    "catalog_path",
    default="config/workflows.example.yaml",
    show_default=True,
)
@click.option(
    "--workflow",
    "workflow_ids",
    multiple=True,
    help="Workflow id to run. Repeat to run multiple workflows. Defaults to enabled workflows.",
)
@click.option(
    "--stage",
    "stages",
    multiple=True,
    type=click.Choice(ROAD_STAGES),
    help="Override stages for every selected workflow. Repeat in execution order.",
)
def run_workflows(
    catalog_path: str,
    workflow_ids: tuple[str, ...],
    stages: tuple[str, ...],
) -> None:
    """Run one or more workflows from a workflow catalog."""
    results = run_workflow_catalog(
        catalog_path,
        workflow_ids=workflow_ids or None,
        stages=stages or None,
    )
    _echo_workflow_results(results)

    failed = [result for result in results if result.status == "failed"]
    if failed:
        raise click.ClickException(f"{len(failed)} workflow(s) failed")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option(
    "--catalog",
    "catalog_path",
    default="config/workflows.example.yaml",
    show_default=True,
    envvar="GEOAI_WORKFLOW_CATALOG",
)
def serve(host: str, port: int, catalog_path: str) -> None:
    """Start the REST API and interactive Swagger UI."""
    try:
        import uvicorn

        from geoai_roads.api import create_app
    except ImportError as exc:
        raise click.ClickException(
            "REST dependencies are missing. Install the project dependencies and try again."
        ) from exc

    click.echo(f"Serving GeoAI Workflow API at http://{host}:{port}")
    click.echo(f"Interactive API UI: http://{host}:{port}/docs")
    uvicorn.run(create_app(default_catalog=catalog_path), host=host, port=port)


def _echo_stage_result(result: StageResult) -> None:
    click.echo(result.message)


def _echo_workflow_results(results: list[WorkflowResult]) -> None:
    for result in results:
        click.echo(f"[{result.workflow_id}] {result.status}: {result.name}")
        for stage in result.stages:
            click.echo(f"  - {stage.message}")
        if result.error:
            click.echo(f"  ! {result.error}")


if __name__ == "__main__":
    main()
