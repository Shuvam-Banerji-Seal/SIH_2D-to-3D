"""Tests for the HTML report content and the pipeline's run-directory bookkeeping.

The report was previously only asserted to *exist* (`.is_file()`), which would
pass even if it rendered empty -- criterion 9 needs its *content* verified.
"""

from __future__ import annotations

import json
from pathlib import Path

from drone3d.config import load_config
from drone3d.pipeline import Pipeline
from drone3d.report.html import generate_report, make_contact_sheet
from drone3d.types import Artifact, PipelineResult, StageReport


def _result() -> PipelineResult:
    result = PipelineResult(run_dir=Path("run"))
    result.stages = [
        StageReport("ingest", "ok", "20 frames sampled", duration_s=7.2),
        StageReport("sfm", "skipped", "no backend", duration_s=0.0),
        StageReport("mesh", "failed", "boom", duration_s=1.5),
    ]
    result.stages[0].artifacts = [Artifact("selected_frames", Path("run/frames"))]
    return result


# --- report content (criterion 9) ------------------------------------------


def test_report_contains_every_stage_name_and_status(tmp_path: Path) -> None:
    path = generate_report(
        run_dir=tmp_path, result=_result(), context={}, out_path=tmp_path / "report.html"
    )

    html = path.read_text(encoding="utf-8")

    for name in ("ingest", "sfm", "mesh"):
        assert name in html
    for status in ("ok", "skipped", "failed"):
        assert status in html


def test_report_carries_stage_messages_and_durations(tmp_path: Path) -> None:
    path = generate_report(
        run_dir=tmp_path, result=_result(), context={}, out_path=tmp_path / "report.html"
    )

    html = path.read_text(encoding="utf-8")

    assert "20 frames sampled" in html
    assert "7.20s" in html


def test_report_renders_metrics_values(tmp_path: Path) -> None:
    """A report that swallows its numbers would make criterion 9 a lie."""
    path = generate_report(
        run_dir=tmp_path,
        result=_result(),
        context={"metrics": {"frames": {"count": 20}, "reprojection_px": 0.3019}},
        out_path=tmp_path / "report.html",
    )

    html = path.read_text(encoding="utf-8")

    assert "count" in html
    assert "0.3019" in html


def test_report_lists_artifacts(tmp_path: Path) -> None:
    path = generate_report(
        run_dir=tmp_path, result=_result(), context={}, out_path=tmp_path / "report.html"
    )

    html = path.read_text(encoding="utf-8")

    assert "selected_frames" in html


def test_report_defaults_to_run_dir_report_html(tmp_path: Path) -> None:
    path = generate_report(run_dir=tmp_path, result=_result(), context={})

    assert path == tmp_path / "report.html"
    assert path.is_file()


def test_report_is_standalone_html(tmp_path: Path) -> None:
    path = generate_report(
        run_dir=tmp_path, result=_result(), context={}, out_path=tmp_path / "r.html"
    )

    html = path.read_text(encoding="utf-8")

    assert html.lstrip().lower().startswith("<!doctype html")
    assert "</html>" in html.lower()


def test_report_escapes_untrusted_messages(tmp_path: Path) -> None:
    result = _result()
    result.stages[0].message = "<script>alert(1)</script>"

    path = generate_report(
        run_dir=tmp_path, result=result, context={}, out_path=tmp_path / "r.html"
    )

    html = path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_escapes_exactly_once(tmp_path: Path) -> None:
    """F20: the message used to be escaped by the caller *and* by `_table`,
    so a benign filename rendered with visible entity artifacts."""
    result = _result()
    result.stages[0].message = "a & b.jpg"

    html = generate_report(
        run_dir=tmp_path, result=result, context={}, out_path=tmp_path / "r.html"
    ).read_text(encoding="utf-8")

    assert "a &amp; b.jpg" in html  # escaped once
    assert "a &amp;amp; b.jpg" not in html  # never double-encoded


def test_contact_sheet_of_missing_images_is_empty(tmp_path: Path) -> None:
    """Missing images yield None rather than raising."""
    assert make_contact_sheet([tmp_path / "nope.jpg"], tmp_path / "sheet.jpg") is None


# --- contact sheet ---------------------------------------------------------


def test_make_contact_sheet_builds_an_image(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    paths: list[Path] = []
    for i in range(3):
        path = tmp_path / f"f{i}.png"
        cv2.imwrite(str(path), np.full((24, 32, 3), 40 * i + 20, dtype=np.uint8))
        paths.append(path)

    sheet = make_contact_sheet(paths, out_path=tmp_path / "sheet.png")

    assert sheet is not None
    assert sheet.is_file()
    assert sheet.stat().st_size > 0


def test_make_contact_sheet_returns_none_without_frames(tmp_path: Path) -> None:
    """An empty run must yield None rather than crash."""
    assert make_contact_sheet([], out_path=tmp_path / "empty.png") is None


def test_report_honours_out_path(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "custom.html"

    path = generate_report(run_dir=tmp_path, result=_result(), context={}, out_path=target)

    assert path == target
    assert target.is_file()


def test_partial_rerun_keeps_earlier_stage_metrics(tmp_path: Path) -> None:
    """F21 regression: re-running one stage must not strip the others' metrics.

    `_collect_metrics` read only `self._result.stages`, so a later
    `--stages metrics` run rendered a report with the sfm/dense/mesh numbers
    missing even though their `result.json` files were still on disk.
    """
    run_dir = tmp_path / "run"
    for name, payload in (
        ("sfm", {"registered_images": 21, "points": 1286}),
        ("depth", {"images": 21}),
        ("mesh", {"meshes": [{"vertices": 31883}]}),
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
        (run_dir / name / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    pipeline = object.__new__(Pipeline)
    pipeline.run_dir = run_dir
    result = PipelineResult(run_dir=run_dir)
    result.stages = [StageReport("metrics", "ok", "4 cloud(s) summarised")]
    pipeline._result = result

    metrics = pipeline._collect_metrics()

    assert metrics["sfm"]["points"] == 1286
    assert metrics["depth"]["images"] == 21
    assert metrics["mesh"]["meshes"][0]["vertices"] == 31883


def test_metrics_are_not_stored_twice(tmp_path: Path) -> None:
    """F22: `metrics.json` is the same payload the `metrics` stage reports.

    Storing both under `metrics` and `summary` made the report render every
    number twice; it now appears once, under `summary`.
    """
    run_dir = tmp_path / "run"
    payload = {"frames": {"count": 20}}
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics" / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    pipeline = object.__new__(Pipeline)
    pipeline.run_dir = run_dir
    result = PipelineResult(run_dir=run_dir)
    result.stages = [StageReport("metrics", "ok", "summarised", metrics=payload)]
    pipeline._result = result

    metrics = pipeline._collect_metrics()

    assert metrics["summary"]["frames"]["count"] == 20
    assert "metrics" not in metrics


def test_manifest_lists_stages_from_earlier_runs(tmp_path: Path) -> None:
    """F23: manifest.json must describe the run directory, not just this invocation.

    `_write_manifest` serialised only `self._result`, so a `--stages metrics`
    re-run wrote a manifest claiming 2 stages and 3 artifacts while six
    result.json files sat on disk.
    """
    run_dir = tmp_path / "run"
    for name in ("ingest", "sfm", "depth"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
        (run_dir / name / "result.json").write_text(json.dumps({"x": 1}), encoding="utf-8")

    pipeline = object.__new__(Pipeline)
    pipeline.run_dir = run_dir
    pipeline.config = load_config(None, [])
    pipeline._result = PipelineResult(run_dir=run_dir)
    pipeline._result.stages = [StageReport("metrics", "ok", "summarised")]

    pipeline._write_manifest()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    names = [s["name"] for s in manifest["result"]["stages"]]
    assert set(names) >= {"metrics", "ingest", "sfm", "depth"}
    assert manifest["result"]["stages"][0]["name"] == "metrics"  # this run first
