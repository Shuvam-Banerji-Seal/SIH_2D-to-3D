"""The warm engine: model cache safety, GPU fitting, job execution and live segmenting."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from drone3d.engine import models
from drone3d.engine.live import LiveSession, recorder_command
from drone3d.engine.service import fit_to_gpu, models_for

GiB = 2**30


@pytest.fixture(autouse=True)
def _no_active_cache():
    yield
    models.activate(None)


class _Net:
    def __init__(self, key: str) -> None:
        self.key = key


def _fake_cache(monkeypatch: pytest.MonkeyPatch, free_gb: float) -> models.ModelCache:
    monkeypatch.setattr(models, "free_bytes", lambda: int(free_gb * GiB))
    cache = models.ModelCache(device="cpu", reserve_gb=2.0)
    cache._loaders = {k: (lambda k=k: _Net(k)) for k in models.SPECS}
    return cache


def test_profile_models() -> None:
    fast = {"stages": ["ingest", "keyframes", "sfm", "dense", "export"], "sfm": {"backend": "flow"},
            "dense": {"backend": "flow"}}  # fmt: skip
    assert models_for(fast) == ["raft_large", "depth_anything_v2_large"]
    accurate = {"stages": ["keyframes", "sfm", "depth", "splat"], "sfm": {"backend": "spirula"},
                "keyframes": {"flow_model": "raft_small"}}  # fmt: skip
    assert models_for(accurate) == ["raft_small", "marigold_v2"]
    assert models_for({"stages": ["ingest"]}) == []


def test_fit_to_gpu_scales_to_free_memory() -> None:
    cfg = {
        "dense": {"tsdf_memory_gb": 24.0},
        "keyframes": {"flow_batch": 32},
        "export": {"texture_size": 8192},
    }
    changes = fit_to_gpu(cfg, free_gb=10.0)
    assert cfg["dense"]["tsdf_memory_gb"] == 4.5
    assert cfg["keyframes"]["flow_batch"] == 16
    assert cfg["export"]["texture_size"] == 4096
    assert len(changes) == 3
    roomy = {"dense": {"tsdf_memory_gb": 8.0}, "keyframes": {"flow_batch": 32}}
    assert fit_to_gpu(roomy, free_gb=70.0) == []
    assert fit_to_gpu(roomy, free_gb=None) == []


def test_cache_refuses_a_model_that_does_not_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _fake_cache(monkeypatch, free_gb=5.0)  # RAFT large needs 4 + 2 reserve
    with pytest.raises(models.InsufficientMemory):
        cache.load("raft_large")
    row = next(m for m in cache.status() if m["key"] == "raft_large")
    assert row["status"] == "unloaded" and "needs 6.0 GiB" in row["error"]
    cache.load("raft_small")  # 2 + 2 fits
    assert next(m for m in cache.status() if m["key"] == "raft_small")["status"] == "loaded"


def test_unload_waits_for_the_running_job(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _fake_cache(monkeypatch, free_gb=40.0)
    cache.load("depth_anything_v2_large")
    with cache.job():
        assert cache.unload("depth_anything_v2_large") == "pending"
        assert (
            next(m for m in cache.status() if m["key"] == "depth_anything_v2_large")["status"]
            == "loaded"
        )
    assert (
        next(m for m in cache.status() if m["key"] == "depth_anything_v2_large")["status"]
        == "unloaded"
    )
    assert cache.unload("depth_anything_v2_large") == "not-loaded"


def test_accessors_build_fresh_models_without_an_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    built = []

    class FakeRaft:
        def __init__(self, model: str, **kw: object) -> None:
            built.append((model, kw))

    monkeypatch.setattr("drone3d.keyframes.flow.RaftFlow", FakeRaft)
    models.raft("raft_small", batch=4, iters=6)
    models.raft("raft_small", batch=4, iters=6)
    assert len(built) == 2  # no cache: one model per call, as before the engine existed


def test_engine_cache_shares_one_network_across_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _fake_cache(monkeypatch, free_gb=40.0)
    made = []

    class FakeRaft:
        def __init__(self, model: str, *, batch: int, iters: int, device: str, net: object) -> None:
            self.net, self.batch = net, batch
            made.append(batch)

    monkeypatch.setattr("drone3d.keyframes.flow.RaftFlow", FakeRaft)
    models.activate(cache)
    a = models.raft("raft_large", batch=8, iters=12)
    b = models.raft("raft_large", batch=16, iters=12)
    c = models.raft("raft_large", batch=8, iters=12)
    assert a is c and a is not b and a.net is b.net
    assert made == [8, 16]
    row = next(m for m in cache.status() if m["key"] == "raft_large")
    assert row["uses"] == 3 and row["status"] == "loaded"


def test_recorder_commands() -> None:
    seg = Path("/tmp/seg")
    rtsp = recorder_command("ffmpeg", "rtsp://drone/live", seg, 30)
    assert (
        rtsp[rtsp.index("-rtsp_transport") + 1] == "tcp"
        and "-rw_timeout" in rtsp
        and "copy" in rtsp
    )
    cam = recorder_command("ffmpeg", "/dev/video0", seg, 20)
    assert cam[cam.index("-f") + 1] == "v4l2" and "h264_nvenc" in cam
    cpu = recorder_command("ffmpeg", "/dev/video0", seg, 20, encoder="libx264")
    assert "libx264" in cpu and "h264_nvenc" not in cpu
    replay = recorder_command("ffmpeg", "flight.mp4", seg, 10, simulate=True)
    assert replay[replay.index("-re") + 1] == "-i" and "-rw_timeout" not in replay
    assert replay[-1].endswith("seg_%04d.mkv") and replay[replay.index("-segment_time") + 1] == "10"


def _test_video(path: Path, seconds: int = 12) -> Path:
    from drone3d.io.nvdec import ffmpeg_bin

    subprocess.run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=size=320x180:rate=30:duration={seconds}", "-c:v", "libx264", "-g", "30",
                    "-pix_fmt", "yuv420p", str(path)], check=True)  # fmt: skip
    return path


class _Recorder:
    """Stands in for the engine: records what a live session queues."""

    def __init__(self) -> None:
        self.jobs: list[tuple[str, dict, dict]] = []

    def submit(self, name: str, config: dict, **kw: object) -> None:
        self.jobs.append((name, config, kw))


def test_live_session_queues_every_segment(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    video = _test_video(tmp_path / "flight.mp4")
    engine = _Recorder()
    cfg = {"stages": ["ingest", "keyframes", "report"], "ingest": {}}
    session = LiveSession(engine, "live1", str(video), tmp_path / "live1", cfg, segment_s=5)  # type: ignore[arg-type]
    session.start()
    deadline = time.time() + 30
    while (session.running or session.stopped is None) and time.time() < deadline:
        time.sleep(0.2)
    assert session.error is None
    names = [n for n, _, _ in engine.jobs]
    assert names[:2] == ["live1__seg_0000", "live1__seg_0001"] and len(names) >= 2
    total = sum(s["duration_s"] for s in session.segments)
    assert abs(total - 12) < 0.2  # nothing lost between segments
    _, seg_cfg, kw = engine.jobs[1]
    assert seg_cfg["ingest"]["video"].endswith("seg_0001.mkv") and "report" not in seg_cfg["stages"]
    assert kw["front"] is True and kw["kind"] == "segment" and kw["live"] == "live1"
    state = json.loads((tmp_path / "live1" / "live.json").read_text())
    assert len(state["segments"]) == len(names)


def test_engine_runs_a_job_and_writes_the_console_files(tmp_path: Path) -> None:
    from drone3d.engine.service import Engine

    video = _test_video(tmp_path / "clip.mp4", seconds=2)
    engine = Engine(tmp_path, tmp_path / "outputs", device="cpu")
    try:
        cfg = {
            "stages": ["ingest"],
            "ingest": {"video": str(video)},
            "metrics": {"gpu_telemetry": False},
        }
        engine.submit("clip", cfg)
        deadline = time.time() + 60
        while time.time() < deadline and (engine.queue or engine.current):
            time.sleep(0.1)
        job = json.loads((tmp_path / "outputs" / "clip" / "ui_job.json").read_text())
        assert job["status"] == "done" and job["engine"] is True
        log = (tmp_path / "outputs" / "clip" / "logs" / "run.log").read_text()
        assert "=== stage: ingest ===" in log and "stage ingest: ok" in log
        assert engine.status()["history"][0]["name"] == "clip"
        from drone3d.exceptions import ConfigError

        with pytest.raises(ConfigError):
            engine.submit("bad", {"stages": ["nope"]})
    finally:
        engine.shutdown()


def test_two_slots_run_together_and_keep_their_logs_apart(tmp_path: Path) -> None:
    from drone3d.engine.service import Engine

    a, b = _test_video(tmp_path / "a.mp4", seconds=2), _test_video(tmp_path / "b.mp4", seconds=3)
    engine = Engine(tmp_path, tmp_path / "outputs", device="cpu", slots=2)
    try:
        for name, video in (("ja", a), ("jb", b)):
            engine.submit(
                name,
                {
                    "stages": ["ingest"],
                    "ingest": {"video": str(video)},
                    "metrics": {"gpu_telemetry": False},
                },
            )
        deadline = time.time() + 60
        while time.time() < deadline and (engine.queue or engine.running):
            time.sleep(0.1)
        for name, other in (("ja", "b.mp4"), ("jb", "a.mp4")):
            assert (
                json.loads((tmp_path / "outputs" / name / "ui_job.json").read_text())["status"]
                == "done"
            )
            log = (tmp_path / "outputs" / name / "logs" / "run.log").read_text()
            assert "stage ingest: ok" in log and other not in log
        assert engine.status()["slots"] == 2
    finally:
        engine.shutdown()


def test_marigold_runs_take_the_engine_to_themselves(tmp_path: Path) -> None:
    from drone3d.engine.service import Engine, Job

    engine = Engine(tmp_path, tmp_path / "outputs", device="cpu", slots=2)
    engine._stop.set()  # drive admission by hand
    engine._wake.set()
    for t in engine._threads:
        t.join(timeout=5)

    def plain(n: str) -> Job:
        return Job(name=n, run_dir=str(tmp_path / n), config={"stages": ["ingest", "keyframes"]})

    marigold = Job(
        name="m", run_dir=str(tmp_path / "m"), config={"stages": ["ingest", "depth", "splat"]}
    )
    engine.queue.extend([plain("p1"), marigold, plain("p2")])
    assert engine._next().name == "p1"
    assert engine._next() is None  # the Marigold run waits for p1, and p2 waits behind it
    engine.running.clear()
    assert engine._next().name == "m"
    assert engine._next() is None  # nothing joins a Marigold run
    engine.running.clear()
    assert engine._next().name == "p2"


def test_raft_wrappers_are_per_engine_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    cache = _fake_cache(monkeypatch, free_gb=40.0)

    class FakeRaft:
        def __init__(self, model: str, *, batch: int, iters: int, device: str, net: object) -> None:
            self.net = net

    monkeypatch.setattr("drone3d.keyframes.flow.RaftFlow", FakeRaft)
    models.activate(cache)
    here = models.raft("raft_large", batch=8, iters=12)
    other = {}
    t = threading.Thread(
        target=lambda: other.update(w=models.raft("raft_large", batch=8, iters=12))
    )
    t.start()
    t.join()
    assert (
        other["w"] is not here and other["w"].net is here.net
    )  # own buffers and graphs, one network


def test_restart_drains_and_keeps_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drone3d.engine import service
    from drone3d.engine.service import Engine, Job

    exits = []
    monkeypatch.setattr(service.os, "_exit", lambda code: exits.append(code))
    engine = Engine(tmp_path, tmp_path / "outputs", device="cpu", slots=1)
    engine._stop.set()
    engine._wake.set()
    for t in engine._threads:
        t.join(timeout=5)
    busy = Job(
        name="busy", run_dir=str(tmp_path / "busy"), config={"stages": ["ingest"]}, status="running"
    )
    engine.running["busy"] = busy
    engine.queue.append(
        Job(name="next", run_dir=str(tmp_path / "next"), config={"stages": ["ingest"]})
    )
    engine.restart(slots=2)
    assert engine._next() is None  # draining: nothing new starts
    time.sleep(1.0)
    assert exits == []  # still waiting for the running job
    engine.running.clear()
    deadline = time.time() + 5
    while not exits and time.time() < deadline:
        time.sleep(0.1)
    assert exits == [0]
    assert json.loads((tmp_path / "outputs" / ".engine_restart").read_text()) == {"slots": 2}
    assert [
        r["name"] for r in json.loads((tmp_path / "outputs" / ".engine_queue.json").read_text())
    ] == ["next"]


def test_status_is_not_blocked_by_a_slow_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    cache = _fake_cache(monkeypatch, free_gb=40.0)
    gate = threading.Event()

    def slow() -> object:
        gate.wait(5)
        return _Net("depth")

    cache._loaders["depth_anything_v2_large"] = slow
    loads = [
        threading.Thread(target=cache.load, args=("depth_anything_v2_large",)) for _ in range(2)
    ]
    for t in loads:
        t.start()
    time.sleep(0.2)
    t0 = time.time()
    row = next(m for m in cache.status() if m["key"] == "depth_anything_v2_large")
    assert time.time() - t0 < 0.5 and row["status"] == "loading"  # answered while the load runs
    gate.set()
    for t in loads:
        t.join(5)
    row = next(m for m in cache.status() if m["key"] == "depth_anything_v2_large")
    assert row["status"] == "loaded"
    assert (
        cache._entries["depth_anything_v2_large"].obj.key == "depth"
    )  # one load, shared by both requests
