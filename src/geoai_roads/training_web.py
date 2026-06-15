from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import shutil
from urllib.parse import quote
import zipfile

import geopandas as gpd
import rasterio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from geoai_roads.training_data import (
    TrainingDataConfig,
    export_training_chips,
    load_training_data_config,
)

DEFAULT_TRAINING_CONFIG = "config/training.whu-taos.example.yaml"
DEFAULT_SEED_LABELS = "outputs/whu_taos_buildings.gpkg"


def register_training_routes(
    app: FastAPI,
    default_training_config: str | Path = DEFAULT_TRAINING_CONFIG,
) -> None:
    app.state.default_training_config = str(default_training_config)

    @app.get("/training", response_class=HTMLResponse)
    def training_home(config_path: str | None = None) -> HTMLResponse:
        config = _load_config(config_path or app.state.default_training_config)
        return _html_response(
            "Training",
            _home_body(config),
            active="home",
        )

    @app.get("/training/export", response_class=HTMLResponse)
    def training_export(config_path: str | None = None) -> HTMLResponse:
        config = _load_config(config_path or app.state.default_training_config)
        return _html_response(
            "Export",
            _export_body(config),
            active="export",
        )

    @app.get("/training/export/package")
    def download_label_package(config_path: str | None = None) -> FileResponse:
        config_file = config_path or app.state.default_training_config
        try:
            package_path = build_label_export_package(config_file)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(
            package_path,
            media_type="application/zip",
            filename=package_path.name,
        )

    @app.get("/training/export/imagery")
    def download_imagery(config_path: str | None = None) -> FileResponse:
        config = _load_config(config_path or app.state.default_training_config)
        imagery_path = config.imagery_source
        if not imagery_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Imagery COG not found: {imagery_path}",
            )
        return FileResponse(
            imagery_path,
            media_type="image/tiff",
            filename=imagery_path.name,
        )

    @app.post("/training/export/chips", response_class=HTMLResponse)
    async def export_chips(request: Request) -> HTMLResponse:
        config_path = await _form_value(request, "config_path", app.state.default_training_config)
        config = _load_config(config_path)
        try:
            summary = export_training_chips(config)
        except Exception as exc:
            return _html_response(
                "Export",
                _export_body(config, error=str(exc)),
                active="export",
                status_code=400,
            )
        return _html_response(
            "Export",
            _export_body(config, summary=summary),
            active="export",
        )

    @app.get("/training/export/chips.zip")
    def download_chips(config_path: str | None = None) -> FileResponse:
        config = _load_config(config_path or app.state.default_training_config)
        try:
            package_path = build_training_chips_package(config)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(
            package_path,
            media_type="application/zip",
            filename=package_path.name,
        )

    @app.get("/training/import", response_class=HTMLResponse)
    def training_import(config_path: str | None = None) -> HTMLResponse:
        config = _load_config(config_path or app.state.default_training_config)
        return _html_response(
            "Import",
            _import_body(config),
            active="import",
        )

    @app.post("/training/import", response_class=HTMLResponse)
    async def upload_labels(request: Request) -> HTMLResponse:
        form = await request.form()
        config_path = str(form.get("config_path") or app.state.default_training_config)
        config = _load_config(config_path)
        upload = form.get("label_file")
        if not upload or not getattr(upload, "filename", ""):
            return _html_response(
                "Import",
                _import_body(config, error="Choose a GeoPackage, GeoJSON, or ZIP file."),
                active="import",
                status_code=400,
            )

        try:
            content = await upload.read()
            result = save_uploaded_labels(
                config=config,
                filename=str(upload.filename),
                content=content,
            )
        except Exception as exc:
            return _html_response(
                "Import",
                _import_body(config, error=str(exc)),
                active="import",
                status_code=400,
            )

        return _html_response(
            "Import",
            _import_body(config, result=result),
            active="import",
        )


def build_label_export_package(config_path: str | Path) -> Path:
    config = load_training_data_config(config_path)
    export_dir = config.output_dir.parent / "exports"
    package_dir = export_dir / _timestamp_slug("label_package")
    package_dir.mkdir(parents=True, exist_ok=True)

    labels = _label_frame_for_export(config)
    label_path = package_dir / "taos_building_labels.gpkg"
    _write_label_frame(labels, label_path, config.label_layer or "buildings")

    readme_path = package_dir / "README.txt"
    readme_path.write_text(_label_package_readme(config), encoding="utf-8")
    shutil.copy2(config.path, package_dir / config.path.name)

    zip_path = export_dir / f"{package_dir.name}.zip"
    _zip_directory(package_dir, zip_path)
    return zip_path


def build_training_chips_package(config: TrainingDataConfig) -> Path:
    if not config.output_dir.exists():
        raise FileNotFoundError(
            f"Training chips not found: {config.output_dir}. Export chips before downloading them."
        )
    export_dir = config.output_dir.parent / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    zip_path = export_dir / f"{_timestamp_slug('training_chips')}.zip"
    _zip_directory(config.output_dir, zip_path)
    return zip_path


def save_uploaded_labels(
    config: TrainingDataConfig,
    filename: str,
    content: bytes,
) -> dict[str, str | int]:
    if not content:
        raise ValueError("Uploaded label file is empty.")

    upload_dir = config.label_source.parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    source_path = _store_upload(upload_dir, filename, content)
    labels = _read_uploaded_labels(source_path)
    if labels.empty:
        raise ValueError("Uploaded label file contains no features.")
    if labels.crs is None:
        raise ValueError("Uploaded label file must have a CRS.")

    config.label_source.parent.mkdir(parents=True, exist_ok=True)
    if config.label_source.exists():
        config.label_source.unlink()
    _write_label_frame(labels, config.label_source, config.label_layer or "buildings")
    return {
        "filename": Path(filename).name,
        "features": len(labels),
        "saved_to": str(config.label_source),
        "layer": config.label_layer or "buildings",
    }


def _home_body(config: TrainingDataConfig) -> str:
    status = _status_rows(config)
    return f"""
    <section class="training-band">
      <div>
        <p class="eyebrow">GeoAI Training</p>
        <h1>Building Model Training</h1>
      </div>
      <div class="training-actions">
        <a class="button primary" href="/training/export">Export</a>
        <a class="button" href="/training/import">Import</a>
      </div>
    </section>
    <section class="training-grid">
      <div class="training-panel">
        <h2>Workspace</h2>
        <dl class="status-list">{status}</dl>
      </div>
      <div class="training-panel">
        <h2>Flow</h2>
        <ol class="flow-list">
          <li>Download the label package.</li>
          <li>Edit building polygons in QGIS.</li>
          <li>Upload the corrected labels.</li>
          <li>Export chips and train from the CLI or container.</li>
        </ol>
      </div>
    </section>
    """


def _export_body(
    config: TrainingDataConfig,
    summary: dict[str, int] | None = None,
    error: str | None = None,
) -> str:
    summary_html = _summary_panel(summary) if summary else ""
    error_html = _notice(error, "error") if error else ""
    config_query = _url_value(config.path)
    chips_ready = config.output_dir.exists()
    imagery_ready = config.imagery_source.exists()
    download_chips = (
        f'<a class="button" href="/training/export/chips.zip?config_path={config_query}">'
        "Download chips</a>"
        if chips_ready
        else '<span class="button disabled">Download chips</span>'
    )
    download_imagery = (
        f'<a class="button primary" href="/training/export/imagery?config_path={config_query}">'
        "Download imagery COG</a>"
        if imagery_ready
        else '<span class="button disabled">Download imagery COG</span>'
    )
    return f"""
    <section class="training-band">
      <div>
        <p class="eyebrow">Export</p>
        <h1>Download Training Inputs</h1>
      </div>
      <a class="button" href="/training">Overview</a>
    </section>
    {error_html}
    {summary_html}
    <section class="training-grid">
      <form class="training-panel" method="get" action="/training/export/package">
        <h2>Label Package</h2>
        <p class="panel-copy">ZIP for QGIS label editing. It contains:</p>
        <ul class="content-list">
          <li><strong>taos_building_labels.gpkg</strong> - editable building polygons layer.</li>
          <li><strong>{escape(config.path.name)}</strong> - training config with source and output paths.</li>
          <li><strong>README.txt</strong> - quick notes for the QGIS correction workflow.</li>
        </ul>
        <p class="panel-copy">Imagery is downloaded separately so the label package stays small.</p>
        <label>Config
          <input name="config_path" value="{escape(str(config.path))}">
        </label>
        <button class="button primary" type="submit">Download package</button>
      </form>
      <section class="training-panel">
        <h2>Imagery COG</h2>
        <p class="panel-copy">Single source raster from the training config. Use this as the QGIS
        base image when correcting labels.</p>
        <dl class="status-list compact">
          <dt>File</dt><dd>{escape(config.imagery_source.name)}</dd>
          <dt>Path</dt><dd>{escape(str(config.imagery_source))}</dd>
          <dt>Size</dt><dd>{escape(_file_size(config.imagery_source))}</dd>
        </dl>
        {download_imagery}
      </section>
      <form class="training-panel" method="post" action="/training/export/chips">
        <h2>Training Chips</h2>
        <p class="panel-copy">Generated model-training ZIP. After export, it contains:</p>
        <ul class="content-list">
          <li><strong>manifest.csv</strong> - chip index, split, mask path, and bounds.</li>
          <li><strong>train/images</strong> and <strong>train/masks</strong> - training tiles.</li>
          <li><strong>val/images</strong> and <strong>val/masks</strong> - validation tiles.</li>
        </ul>
        <input type="hidden" name="config_path" value="{escape(str(config.path))}">
        <dl class="status-list">{_status_rows(config)}</dl>
        <div class="training-actions">
          <button class="button primary" type="submit">Export chips</button>
          {download_chips}
        </div>
      </form>
    </section>
    """


def _import_body(
    config: TrainingDataConfig,
    result: dict[str, str | int] | None = None,
    error: str | None = None,
) -> str:
    result_html = _summary_panel(result, title="Imported") if result else ""
    error_html = _notice(error, "error") if error else ""
    return f"""
    <section class="training-band">
      <div>
        <p class="eyebrow">Import</p>
        <h1>Upload Corrected Labels</h1>
      </div>
      <a class="button" href="/training">Overview</a>
    </section>
    {error_html}
    {result_html}
    <section class="training-grid">
      <form class="training-panel wide" method="post" action="/training/import" enctype="multipart/form-data">
        <h2>Label Upload</h2>
        <input type="hidden" name="config_path" value="{escape(str(config.path))}">
        <label>Corrected label file
          <input type="file" name="label_file" accept=".gpkg,.geojson,.json,.zip" required>
        </label>
        <dl class="status-list">{_status_rows(config)}</dl>
        <button class="button primary" type="submit">Upload labels</button>
      </form>
    </section>
    """


def _html_response(
    title: str,
    body: str,
    active: str,
    status_code: int = 200,
) -> HTMLResponse:
    return HTMLResponse(
        content=f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} - GeoAI Training</title>
  <style>{_css()}</style>
</head>
<body>
  <header class="topbar">
    <a class="brand" href="/training">GeoAI Training</a>
    <nav>
      {_nav_link("Overview", "/training", active == "home")}
      {_nav_link("Export", "/training/export", active == "export")}
      {_nav_link("Import", "/training/import", active == "import")}
      <a href="/docs">API</a>
    </nav>
  </header>
  <main>{body}</main>
</body>
</html>""",
        status_code=status_code,
    )


def _css() -> str:
    return """
    :root { color-scheme: dark; --bg: #071b2e; --panel: #0f2c49; --panel-2: #12385c; --line: #235477; --text: #eef7ff; --muted: #a9c7dd; --blue: #40b6ec; --blue-dark: #0e83bd; --error: #f87171; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: var(--bg); color: var(--text); font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .topbar { min-height: 64px; display: flex; align-items: center; justify-content: space-between; gap: 24px; padding: 0 28px; background: #0b2a47; border-bottom: 1px solid var(--line); }
    .brand { color: var(--text); font-size: 18px; font-weight: 800; text-decoration: none; }
    nav { display: flex; flex-wrap: wrap; gap: 8px; }
    nav a, .button { display: inline-flex; align-items: center; justify-content: center; min-height: 38px; padding: 0 14px; border: 1px solid var(--line); border-radius: 6px; color: var(--text); background: #12365a; text-decoration: none; font-weight: 700; cursor: pointer; }
    nav a.active, .button.primary { color: #062038; background: var(--blue); border-color: var(--blue); }
    .button.disabled { opacity: .45; cursor: not-allowed; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    .training-band { display: flex; align-items: center; justify-content: space-between; gap: 24px; padding: 18px 0 22px; border-bottom: 1px solid var(--line); }
    h1, h2, p { margin: 0; }
    h1 { font-size: 28px; line-height: 1.15; }
    h2 { font-size: 17px; margin-bottom: 16px; }
    .eyebrow { color: var(--blue); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 4px; }
    .training-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 18px; }
    .training-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }
    .training-panel.wide { grid-column: 1 / -1; }
    .training-actions { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .panel-copy { color: var(--muted); margin-bottom: 12px; }
    .content-list { margin: 0 0 16px; padding-left: 20px; color: var(--muted); }
    .content-list li { margin: 7px 0; }
    .content-list strong { color: var(--text); }
    label { display: grid; gap: 8px; color: var(--muted); font-weight: 700; margin-bottom: 16px; }
    input { width: 100%; min-height: 40px; border-radius: 6px; border: 1px solid var(--line); background: #06192a; color: var(--text); padding: 8px 10px; }
    input[type=file] { padding: 8px; }
    button { font: inherit; }
    .status-list { display: grid; grid-template-columns: minmax(120px, .35fr) 1fr; gap: 8px 14px; margin: 0 0 16px; }
    .status-list.compact { grid-template-columns: minmax(76px, .2fr) 1fr; }
    .status-list dt { color: var(--muted); font-weight: 800; }
    .status-list dd { margin: 0; overflow-wrap: anywhere; }
    .flow-list { margin: 0; padding-left: 20px; color: var(--muted); }
    .flow-list li { margin: 7px 0; }
    .notice { margin-top: 18px; padding: 12px 14px; border-radius: 6px; border: 1px solid var(--line); background: var(--panel-2); }
    .notice.error { border-color: #7f1d1d; color: #fecaca; background: #3d1017; }
    @media (max-width: 760px) { .topbar, .training-band { align-items: stretch; flex-direction: column; } main { padding: 16px; } .training-grid { grid-template-columns: 1fr; } }
    """


def _nav_link(label: str, href: str, active: bool) -> str:
    class_name = ' class="active"' if active else ""
    return f'<a{class_name} href="{href}">{escape(label)}</a>'


def _status_rows(config: TrainingDataConfig) -> str:
    rows = {
        "Imagery": _path_status(config.imagery_source),
        "Labels": _path_status(config.label_source),
        "Chips": _path_status(config.output_dir),
        "Config": _path_status(config.path),
    }
    return "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>"
        for label, value in rows.items()
    )


def _path_status(path: Path) -> str:
    state = "ready" if path.exists() else "missing"
    return f"{path} ({state})"


def _file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{size} B"


def _summary_panel(summary: dict | None, title: str = "Exported") -> str:
    if not summary:
        return ""
    rows = "".join(
        f"<dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd>"
        for key, value in summary.items()
    )
    return f'<section class="notice"><h2>{escape(title)}</h2><dl class="status-list">{rows}</dl></section>'


def _notice(message: str | None, level: str = "") -> str:
    if not message:
        return ""
    class_name = f"notice {level}".strip()
    return f'<section class="{class_name}">{escape(message)}</section>'


async def _form_value(request: Request, name: str, default: str) -> str:
    form = await request.form()
    return str(form.get(name) or default)


def _load_config(config_path: str | Path) -> TrainingDataConfig:
    try:
        return load_training_data_config(config_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _label_frame_for_export(config: TrainingDataConfig) -> gpd.GeoDataFrame:
    if config.label_source.exists():
        return gpd.read_file(config.label_source, layer=config.label_layer)

    seed_path = _resolve_repo_path(DEFAULT_SEED_LABELS, config.path)
    if seed_path.exists():
        frame = gpd.read_file(seed_path)
        if not frame.empty:
            return frame

    crs = "EPSG:4326"
    if config.imagery_source.exists():
        with rasterio.open(config.imagery_source) as dataset:
            crs = dataset.crs or crs
    return gpd.GeoDataFrame({"class_name": []}, geometry=[], crs=crs)


def _write_label_frame(frame: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    frame.to_file(path, layer=layer, driver="GPKG")


def _label_package_readme(config: TrainingDataConfig) -> str:
    return (
        "GeoAI training label package\n\n"
        "This ZIP contains:\n"
        "- taos_building_labels.gpkg: editable building labels for QGIS.\n"
        f"- {config.path.name}: training configuration and expected source/output paths.\n"
        "- README.txt: this file.\n\n"
        "The imagery COG is not included in this ZIP. Download it separately from "
        "/training/export so the label package stays small.\n\n"
        "Open taos_building_labels.gpkg in QGIS, edit the buildings layer, then upload the "
        "corrected GeoPackage back to GeoAI at /training/import.\n\n"
        f"Imagery source expected by config: {config.imagery_source}\n"
        f"Upload target: {config.label_source}\n"
    )


def _store_upload(upload_dir: Path, filename: str, content: bytes) -> Path:
    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("Uploaded file must have a filename.")
    upload_path = upload_dir / safe_name
    upload_path.write_bytes(content)
    return upload_path


def _read_uploaded_labels(path: Path) -> gpd.GeoDataFrame:
    suffix = path.suffix.lower()
    if suffix in {".gpkg", ".geojson", ".json"}:
        return gpd.read_file(path)
    if suffix == ".zip":
        return _read_zipped_labels(path)
    raise ValueError("Upload a .gpkg, .geojson, .json, or .zip label file.")


def _read_zipped_labels(path: Path) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            suffix = Path(name).suffix.lower()
            if suffix not in {".gpkg", ".geojson", ".json"}:
                continue
            data = archive.read(name)
            temp_path = path.parent / f"extracted_{Path(name).name}"
            temp_path.write_bytes(data)
            try:
                return gpd.read_file(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
    raise ValueError("ZIP upload did not contain a .gpkg, .geojson, or .json label file.")


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def _timestamp_slug(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def _resolve_repo_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent.parent / path).resolve()


def _url_value(path: Path) -> str:
    return quote(str(path), safe="")
