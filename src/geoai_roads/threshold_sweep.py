from __future__ import annotations

from pathlib import Path
from typing import Any

from geoai_roads.config import RoadConfig
from geoai_roads.masking import threshold_probability_rasters, threshold_tag
from geoai_roads.postgis import load_vectors_to_postgis
from geoai_roads.vectorize import vectorize_masks


def run_threshold_sweep(
    config: RoadConfig,
    postgis_if_exists: str = "append",
    job_id: str | None = None,
    workflow_id: str | None = None,
) -> list[dict[str, Any]]:
    if not config.probability_dir:
        raise ValueError("Threshold sweep requires inference.probability_dir or save_probability.")

    thresholds = config.threshold_sweep or [config.road_threshold]
    results = []
    base_job_id = job_id or workflow_id or config.raw.get("project", {}).get("name") or "threshold-sweep"

    for threshold in thresholds:
        tag = threshold_tag(threshold)
        mask_dir = _tagged_dir(config.mask_dir, tag)
        vector_output = _tagged_path(config.vector_output, tag)
        sweep_job_id = f"{base_job_id}-{tag}"
        sweep_workflow_id = f"{workflow_id}-{tag}" if workflow_id else tag

        threshold_probability_rasters(
            probability_dir=config.probability_dir,
            mask_dir=mask_dir,
            threshold=threshold,
            class_name=config.class_name,
            average_overlaps=config.average_probability_overlaps,
            cleanup=config.mask_cleanup,
        )
        feature_count = vectorize_masks(
            mask_dir=mask_dir,
            output_path=vector_output,
            processing_crs=config.processing_crs,
            output_crs=config.output_crs,
            min_area_m2=config.min_area_m2,
            simplify_tolerance_m=config.simplify_tolerance_m,
            smooth_tolerance_m=config.smooth_tolerance_m,
            rectangularize=config.rectangularize,
            rectangularize_min_area_ratio=config.rectangularize_min_area_ratio,
            dissolve_overlaps=config.dissolve_overlaps,
            regularize=config.regularize,
            regularize_tolerance_m=config.regularize_tolerance_m,
            regularize_angle_tolerance_degrees=config.regularize_angle_tolerance_degrees,
            regularize_min_area_ratio=config.regularize_min_area_ratio,
            regularize_max_area_ratio=config.regularize_max_area_ratio,
            max_mask_coverage=config.max_mask_coverage,
            max_source_pixel_size_m=config.max_source_pixel_size_m,
            class_name=config.class_name,
        )
        loaded_count = load_vectors_to_postgis(
            vector_path=vector_output,
            database_url=config.postgis_url,
            schema=config.postgis_schema,
            table=config.postgis_table,
            if_exists=postgis_if_exists,
            job_id=sweep_job_id,
            metadata={"workflow_id": sweep_workflow_id},
        )
        results.append(
            {
                "threshold": threshold,
                "job_id": sweep_job_id,
                "workflow_id": sweep_workflow_id,
                "feature_count": loaded_count,
                "vector_feature_count": feature_count,
                "mask_dir": str(mask_dir),
                "vector_output": str(vector_output),
            }
        )

    return results


def _tagged_dir(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.name}_{tag}")


def _tagged_path(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")
