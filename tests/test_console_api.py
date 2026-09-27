"""The console's HTTP API: schema, profiles, runs, scenes, filmstrips, thumbnails, live and engine-offline paths."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from drone3d.app.server import create_app  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "fast.yaml").write_text(
        "# Fast profile: test.\nstages: [ingest, keyframes]\n"
    )
    outputs = repo / "outputs"
    run = outputs / "demo"
    (run / "export" / "model_0").mkdir(parents=True)
    (run / "export" / "scene.json").write_text(json.dumps({"title": "demo", "models": [
        {"name": "model 0", "dir": "model_0", "mesh": "model_0/mesh.glb", "points": "model_0/points.ply", "splat": None,
         "files": [{"label": "mesh.glb", "path": "model_0/mesh.glb"}]}]}))  # fmt: skip
    (run / "export" / "model_0" / "frame.json").write_text(
        json.dumps({"scale": 1.0, "frame": "ENU"})
    )
    images = run / "dataset" / "images" / "pass_00"
    images.mkdir(parents=True)
    from PIL import Image

    for k in (0, 8):
        Image.new("RGB", (640, 360), (40 * k, 90, 160)).save(images / f"f_{k:06d}.jpg")
    (run / "dense" / "model_0" / "depth").mkdir(parents=True)
    Image.new("RGB", (160, 90)).save(run / "dense" / "model_0" / "depth" / "f_000008.jpg")
    (run / "logs").mkdir()
    (run / "logs" / "run.log").write_text("00:00:01 | INFO    | drone3d.pipeline | === stage: ingest ===\n"
                                          "00:00:02 | INFO    | drone3d.pipeline | stage ingest: ok (0.5s) probed\n")  # fmt: skip
    app = create_app(repo, outputs, engine_port=_free_port())
    return TestClient(app)


def test_schema_and_profiles(client: TestClient) -> None:
    schema = client.get("/api/schema").json()
    assert {s["key"] for s in schema["sections"]} >= {
        "keyframes",
        "sfm",
        "dense",
        "export",
        "metrics",
    }
    assert all(f["help"] for s in schema["sections"] for f in s["fields"])  # every option explained
    (prof,) = client.get("/api/profiles").json()
    assert prof["name"] == "fast" and prof["description"] == "Fast profile: test."


def test_runs_scene_and_frames(client: TestClient) -> None:
    runs = client.get("/api/runs").json()
    assert [r["name"] for r in runs] == ["demo"] and runs[0]["stages"]["ingest"]["status"] == "ok"
    scene = client.get("/api/runs/demo/scene").json()
    (m,) = scene["models"]
    assert m["mesh"] == "/runs/demo/export/model_0/mesh.glb" and m["frame"]["frame"] == "ENU"
    assert m["files"][0]["path"] == "/runs/demo/export/model_0/mesh.glb"
    frames = client.get("/api/runs/demo/frames").json()
    assert frames["total"] == 2
    assert frames["frames"][0]["depth"] is None
    assert frames["frames"][1]["depth"] == "/runs/demo/dense/model_0/depth/f_000008.jpg"
    thumb = client.get(frames["frames"][0]["image"] + "?w=160")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert client.get("/api/runs/nope/scene").status_code == 404


def test_thumbnails_stay_inside_outputs(client: TestClient) -> None:
    assert client.get("/api/thumb/../configs/fast.yaml").status_code == 404
    assert client.get("/api/thumb/demo/export/scene.json").status_code == 404  # only JPEGs


def test_live_rejects_bad_sources_and_names(client: TestClient) -> None:
    r = client.post(
        "/api/live", json={"name": "x", "profile": "fast", "source": "file:///etc/passwd"}
    )
    assert r.status_code == 400 and "rtsp" in r.json()["detail"]
    r = client.post(
        "/api/live", json={"name": "../bad", "profile": "fast", "source": "rtsp://cam/live"}
    )
    assert r.status_code == 400


def test_engine_offline_status_and_cold_fallback(client: TestClient) -> None:
    e = client.get("/api/engine").json()
    assert e["online"] is False and e["managed"] is False
    assert client.get("/api/system").json()["engine_online"] is False
    r = client.post(
        "/api/runs",
        json={"name": "cold", "profile": "fast", "video": "datasets/none.mp4", "engine": True},
    )
    assert (
        r.status_code == 200 and r.json()["via"] == "subprocess"
    )  # no engine: queued as a subprocess
    r = client.post("/api/runs/cold/stop")
    assert r.status_code == 200


def test_gpu_endpoint_shape(client: TestClient) -> None:
    g = client.get("/api/gpu?since=0").json()
    assert set(g) == {"now", "latest", "series"}
