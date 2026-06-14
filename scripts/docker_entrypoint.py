from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

HF_MODEL_PATH = Path("models/aerial-image-road-segmentation-xp.keras")
SAMPLE_COG_PATH = Path("data/imagery/new-mexico-naip-taos-cog.tif")


def main() -> None:
    maybe_download_hf_model()
    maybe_fetch_sample_cog()

    command = sys.argv[1:] or [
        sys.executable,
        "-m",
        "geoai_roads.cli",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--catalog",
        os.getenv("GEOAI_WORKFLOW_CATALOG", "config/workflows.example.yaml"),
    ]
    os.execvp(command[0], command)


def maybe_download_hf_model() -> None:
    if not enabled("GEOAI_DOWNLOAD_HF_MODEL", default=True):
        return

    model_path = Path(os.getenv("GEOAI_HF_MODEL_PATH", str(HF_MODEL_PATH)))
    if model_path.exists():
        print(f"HF road model found: {model_path}", flush=True)
        return

    print(f"HF road model missing, downloading to {model_path}", flush=True)
    run([sys.executable, "scripts/download_hf_road_model.py", "--output", str(model_path)])


def maybe_fetch_sample_cog() -> None:
    if not enabled("GEOAI_FETCH_SAMPLE_COG", default=True):
        return

    cog_path = Path(os.getenv("GEOAI_SAMPLE_COG_PATH", str(SAMPLE_COG_PATH)))
    if cog_path.exists():
        print(f"Sample New Mexico COG found: {cog_path}", flush=True)
        return

    print(f"Sample New Mexico COG missing, fetching to {cog_path}", flush=True)
    run([sys.executable, "scripts/fetch_new_mexico_cog.py", "--output", str(cog_path)])


def enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
