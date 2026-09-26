"""Self-contained HTML run report generation."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drone3d.logging_utils import get_logger
from drone3d.types import PipelineResult

__all__ = ["generate_report", "make_contact_sheet"]

log = get_logger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_STATUS_COLORS = {"ok": "#1a7f37", "skipped": "#9a6700", "failed": "#cf222e", "pending": "#57606a"}


def _embed_image(path: Path, max_bytes: int = 8_000_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
    return f"data:{mime};base64,{encoded}"


class _Raw(str):
    """Marker for table cells that already contain trusted HTML."""


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{cell if isinstance(cell, _Raw) else html.escape(str(cell))}</td>" for cell in row
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _flatten(data: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in data.items():
        label = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(_flatten(value, prefix=f"{label}."))
        elif isinstance(value, list | tuple) and len(value) > 6:
            rows.append((label, f"[{len(value)} items]"))
        else:
            rows.append((label, value))
    return rows


def make_contact_sheet(
    image_paths: list[Path],
    out_path: str | Path,
    *,
    cols: int = 4,
    thumb_width: int = 320,
    max_images: int = 12,
) -> Path | None:
    """Tile up to ``max_images`` frames into a single labelled contact sheet."""
    selected = [Path(path) for path in image_paths[:max_images] if Path(path).is_file()]
    if not selected:
        return None

    tiles: list[np.ndarray] = []
    for path in selected:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        tile = cv2.resize(
            image,
            (thumb_width, max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 18), (0, 0, 0), thickness=-1)
        cv2.putText(
            tile,
            path.stem,
            (4, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return None
    tile_height = max(tile.shape[0] for tile in tiles)
    cols = max(1, min(cols, len(tiles)))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * tile_height, cols * thumb_width, 3), 32, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        sheet[
            row * tile_height : row * tile_height + tile.shape[0],
            col * thumb_width : col * thumb_width + tile.shape[1],
        ] = tile

    sheet_path = Path(out_path)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        log.warning("failed to write contact sheet: %s", sheet_path)
        return None
    return sheet_path


def generate_report(
    *,
    run_dir: str | Path,
    result: PipelineResult,
    context: dict[str, Any],
    out_path: str | Path | None = None,
) -> Path:
    """Render a standalone HTML report for a pipeline run.

    Args:
        run_dir: Root of the run directory.
        result: Aggregate pipeline result (stage statuses, metrics, artifacts).
        context: Extra sections, e.g. ``video``, ``telemetry``, ``metrics``,
            ``contact_sheet``, ``config``.
        out_path: Output HTML path; defaults to ``<run_dir>/report.html``.
    """
    run_path = Path(run_dir)
    report_path = Path(out_path) if out_path is not None else run_path / "report.html"

    status_rows = [
        [
            stage.name,
            _Raw(
                f'<span style="color:{_STATUS_COLORS.get(stage.status, "#57606a")};'
                f'font-weight:600">{html.escape(stage.status)}</span>'
            ),
            f"{stage.duration_s:.2f}s",
            # Left raw on purpose: `_table` escapes every non-`_Raw` cell, so
            # escaping here too would double-encode (F20) and print literal
            # `&amp;`/`&lt;` in the rendered report.
            stage.message or "-",
        ]
        for stage in result.stages
    ]
    stage_table = _table(["Stage", "Status", "Duration", "Message"], status_rows)

    artifact_rows = [
        [
            artifact.name,
            str(artifact.path.relative_to(run_path))
            if artifact.path.is_relative_to(run_path)
            else str(artifact.path),
            artifact.kind,
        ]
        for stage in result.stages
        for artifact in stage.artifacts
    ]
    artifact_table = _table(["Artifact", "Path", "Kind"], artifact_rows) if artifact_rows else ""

    metrics = dict(context.get("metrics") or {})
    metrics_rows = _flatten(metrics)
    metrics_table = (
        _table(["Metric", "Value"], [[key, value] for key, value in metrics_rows])
        if metrics_rows
        else "<p>No metrics available.</p>"
    )

    video = context.get("video") or {}
    video_rows = _flatten(video)
    video_table = (
        _table(["Property", "Value"], [[key, value] for key, value in video_rows])
        if video_rows
        else "<p>No video metadata (ingest stage skipped).</p>"
    )

    telemetry = context.get("telemetry") or {}
    telemetry_rows = _flatten(telemetry)
    telemetry_table = (
        _table(["Property", "Value"], [[key, value] for key, value in telemetry_rows])
        if telemetry_rows
        else "<p>No telemetry supplied.</p>"
    )

    contact_path = context.get("contact_sheet")
    sheet_html = "<p>No contact sheet.</p>"
    if contact_path:
        data_uri = _embed_image(Path(contact_path))
        if data_uri:
            sheet_html = (
                f'<img class="sheet" src="{data_uri}" alt="selected frames contact sheet"/>'
            )

    config_json = html.escape(json.dumps(context.get("config") or {}, indent=2))
    generated_at = html.escape(str(context.get("generated_at", "")))
    version = html.escape(str(context.get("version", "")))
    ok_text = "completed successfully" if result.ok else "completed with issues"
    ok_color = "#1a7f37" if result.ok else "#cf222e"

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>drone3d run report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0 auto; max-width: 1100px; padding: 2rem; line-height: 1.5; }}
  h1 {{ margin-bottom: 0.25rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #8884; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem; font-size: 0.92rem; }}
  th, td {{ border: 1px solid #8884; padding: 0.45rem 0.6rem; text-align: left; vertical-align: top; }}
  th {{ background: #8881; }}
  pre {{ background: #8881; padding: 1rem; overflow-x: auto; border-radius: 6px; font-size: 0.85rem; }}
  .sheet {{ width: 100%; border-radius: 8px; border: 1px solid #8884; }}
  .status {{ color: {ok_color}; font-weight: 600; }}
  footer {{ margin-top: 3rem; font-size: 0.85rem; opacity: 0.7; }}
</style>
</head>
<body>
<h1>Single-pass drone video &rarr; 3D reconstruction report</h1>
<p class="status">Run {ok_text} &middot; run directory <code>{html.escape(str(run_path))}</code>
   &middot; generated {generated_at} &middot; drone3d {version}</p>

<h2>Pipeline stages</h2>
{stage_table}

<h2>Key metrics</h2>
{metrics_table}

<h2>Input video</h2>
{video_table}

<h2>Telemetry</h2>
{telemetry_table}

<h2>Selected frames</h2>
{sheet_html}

<h2>Artifacts</h2>
{artifact_table}

<h2>Configuration</h2>
<pre>{config_json}</pre>

<footer>Generated by drone3d &mdash; SIH problem statement 26158 (NTRO),
single-pass drone video to accurate 3D model generation.</footer>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(document, encoding="utf-8")
    log.info("report written: %s", report_path)
    return report_path
