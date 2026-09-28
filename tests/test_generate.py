"""The generated object: which keyframe it starts from, the subprocess around TRELLIS.2, the console's view of it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from drone3d import generate
from drone3d.export.stage import _subject_view


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    d = target - eye
    return d / np.linalg.norm(d)


def _orbit(n: int, radius: float, height: float) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(n, height)], 1)


def test_the_farthest_keyframe_looking_at_the_subject_is_chosen() -> None:
    near, far = _orbit(10, 5.0, 3.0), _orbit(4, 12.0, 6.0)
    away = np.array([[30.0, 0.0, 6.0]])  # farthest of all, but looking out to the skyline
    cams = np.concatenate([near, far, away])
    axes = np.array([_look_at(c, np.zeros(3)) for c in cams[:-1]] + [_look_at(away[0], np.array([60.0, 0.0, 0.0]))])
    k = generate.subject_keyframe(cams, axes)
    assert 10 <= k < 14  # one of the far orbit's views: the subject whole, not one arch, not the skyline


def test_a_survey_whose_axes_do_not_converge_keeps_the_middle_keyframe() -> None:
    xs = np.linspace(-50, 50, 9)
    cams = np.stack([xs, np.zeros(9), np.full(9, 40.0)], 1)
    axes = np.tile([0.0, 0.0, -1.0], (9, 1))
    assert generate.subject_keyframe(cams, axes) == 4


def test_the_viewer_starts_at_a_keyframe_looking_at_the_subject() -> None:
    cams = _orbit(9, 10.0, 4.0)
    axes = np.array([_look_at(c, np.zeros(3)) for c in cams])
    axes[4] = _look_at(cams[4], cams[4] + np.array([0.0, 0.0, 1.0]) + 3 * (cams[4] / 10))  # the middle one: skyward
    rng = np.random.default_rng(0)
    points = rng.normal(0, 0.5, (2000, 3))
    eye, target = _subject_view(points, cams, axes)
    assert not np.allclose(eye, cams[4])
    assert np.linalg.norm(target) < 1.0  # at the subject, not at the sky


def _fake_trellis(tmp_path: Path, ok: bool = True) -> Path:
    """A stand-in for tools/trellis2_generate.py: OUT.glb IMAGE -> writes OUT and OUT.input.png."""
    script = tmp_path / "fake_trellis.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\nout = Path(sys.argv[1])\n"
        + ("out.write_bytes(b'glTF'); out.with_suffix('.input.png').write_bytes(b'png')\nprint('generated')\n"
           if ok else "print('CUDA error: out of memory', file=sys.stderr); sys.exit(3)\n")
    )  # fmt: skip
    return script


def test_generate_object_records_its_files_and_what_it_was_made_from(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    run = tmp_path / "run"
    (run / "dataset" / "images").mkdir(parents=True)
    image = run / "dataset" / "images" / "k.jpg"
    image.write_bytes(b"jpg")
    monkeypatch.setattr(generate, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(generate, "SCRIPT", _fake_trellis(tmp_path))
    rec = generate.generate_object(run, image=image)
    assert rec["status"] == "ok" and rec["glb"] == "object.glb" and rec["input"] == "object.input.png"
    assert (run / "export" / "generated" / "object.glb").read_bytes() == b"glTF"
    on_disk = generate.status(run)
    assert on_disk["status"] == "ok" and "not measured" in on_disk["note"] and on_disk["licenses"]


def test_a_failed_generation_keeps_the_reason(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(generate, "PYTHON", Path(sys.executable))
    monkeypatch.setattr(generate, "SCRIPT", _fake_trellis(tmp_path, ok=False))
    rec = generate.generate_object(run, image=tmp_path / "missing.jpg")
    assert rec["status"] == "failed" and rec["glb"] is None
    assert any("out of memory" in line for line in rec["log"])


pytest.importorskip("fastapi")


def _console(tmp_path: Path):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from drone3d.app.server import create_app

    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "fast.yaml").write_text("stages: [ingest]\n")
    run = repo / "outputs" / "demo"
    (run / "sfm").mkdir(parents=True)
    (run / "sfm" / "result.json").write_text(json.dumps({"models": [{"path": str(run / "dataset" / "sparse" / "0"), "images": 12}]}))
    return TestClient(create_app(repo, repo / "outputs", engine_port=1)), run


def test_the_catalog_lists_the_generated_object_and_notices_a_dead_generator(tmp_path: Path) -> None:
    client, run = _console(tmp_path)
    g = client.get("/api/runs/demo/models").json()["generated"]
    assert g["status"] == "none" and g["glb"] is None and "not measured" in g["note"]
    out = run / "export" / "generated"
    out.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (out / "result.json").write_text(json.dumps({"status": "running", "pid": dead.pid, "keyframe": "pass_09/f_1.jpg"}))
    assert client.get("/api/runs/demo/models").json()["generated"]["status"] == "interrupted"
    (out / "result.json").write_text(json.dumps({"status": "ok", "glb": "object.glb", "input": "object.input.png",
                                                 "keyframe": "pass_09/f_1.jpg", "seconds": 310.0}))  # fmt: skip
    g = client.get("/api/runs/demo/models").json()["generated"]
    assert g["glb"] == "generated/object.glb" and g["input"] == "/runs/demo/export/generated/object.input.png"


def test_generate_endpoint_refuses_unknown_runs_and_missing_trellis(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client, _ = _console(tmp_path)
    assert client.post("/api/runs/nope/generate").status_code == 404
    monkeypatch.setattr(generate, "available", lambda: False)
    r = client.post("/api/runs/demo/generate")
    assert r.status_code == 409 and "TRELLIS.2" in r.json()["detail"]


def test_the_focus_box_is_centred_on_what_the_orbit_circles() -> None:
    from drone3d.export.stage import _subject

    centre = np.array([5.0, -2.0, 0.0])
    cams = centre + _orbit(12, 10.0, 4.0)
    axes = np.array([_look_at(c, centre) for c in cams])
    point, radius = _subject(cams, axes)
    assert np.allclose(point, centre, atol=1e-6)
    assert abs(radius - np.hypot(10.0, 4.0)) < 1e-6
    survey = np.stack([np.linspace(-50, 50, 9), np.zeros(9), np.full(9, 40.0)], 1)
    assert _subject(survey, np.tile([0.0, 0.0, -1.0], (9, 1))) is None


def test_a_point_no_keyframe_looks_at_is_not_a_subject() -> None:
    """A turning fly-by: the axes' least-squares point lies behind the cameras, so there is no focus."""
    from drone3d.export.stage import _subject

    cams = _orbit(9, 10.0, 4.0)[:5]
    axes = np.array([-_look_at(c, np.zeros(3)) for c in cams])  # every camera looks away from the centre
    assert _subject(cams, axes) is None
