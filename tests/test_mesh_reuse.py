"""A finished mesh is reused only if it was made from the current splat."""

from __future__ import annotations

import os
from pathlib import Path

from drone3d.splat.spirula import _finished_mesh

DONE = "[meshing] wrote: x\n[meshing] done: vertices: 10, faces: 12 (total 42.50s)\n"


def _run(tmp_path: Path, log_text: str | None, names: list[str]) -> tuple[Path, Path]:
    run = tmp_path / "model_0"
    ckpt = run / "step-000030000.ckpt"
    ckpt.mkdir(parents=True)
    splat = ckpt / "splat.ply"
    splat.write_bytes(b"ply")
    os.utime(splat, (1_000, 1_000))
    for name in names:
        (ckpt / name).write_bytes(b"m")
    log = tmp_path / "model_0_mesh.log"
    if log_text is not None:
        log.write_text(log_text)
    return run, log


def test_reuses_complete_mesh(tmp_path: Path) -> None:
    run, log = _run(tmp_path, DONE, ["mesh_vertexcolor.ply", "mesh_textured.glb", "mesh_textured.obj"])
    found = _finished_mesh(run, log, ("ply", "glb", "obj"))
    assert found is not None and found.seconds == 42.5 and len(found.files) == 3


def test_rejects_unfinished_log_missing_format_or_stale_mesh(tmp_path: Path) -> None:
    run, log = _run(tmp_path / "a", "[meshing] UV: charts\n", ["mesh_vertexcolor.ply"])
    assert _finished_mesh(run, log, ("ply",)) is None  # log never reached "done"
    run, log = _run(tmp_path / "b", DONE, ["mesh_vertexcolor.ply"])
    assert _finished_mesh(run, log, ("ply", "glb")) is None  # glb was not produced
    run, log = _run(tmp_path / "c", DONE, ["mesh_vertexcolor.ply"])
    splat = run / "step-000030000.ckpt" / "splat.ply"
    os.utime(splat, None)  # retrained after meshing: the mesh is stale
    os.utime(run / "step-000030000.ckpt" / "mesh_vertexcolor.ply", (2_000, 2_000))
    assert _finished_mesh(run, log, ("ply",)) is None
    assert _finished_mesh(tmp_path / "missing", tmp_path / "none.log", ("ply",)) is None
