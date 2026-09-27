"""The console's side of the engine: find it, start and stop it, supervise it, forward requests.

The engine is a separate process (``drone3d engine``) so the web server never
imports torch and a CUDA fault cannot take the console down. The console
starts one on request, restarts it when it exits after a sticky CUDA error
(exit code 3; it saved its queue and warm list first), and otherwise only
talks to it over HTTP -- an engine started by hand is used the same way.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

__all__ = ["EngineClient", "EngineError"]

POISONED_EXIT = 3  # the engine's exit code after a sticky CUDA fault


class EngineError(RuntimeError):
    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


class EngineClient:
    def __init__(self, repo: Path, outputs: Path, *, port: int = 8770, slots: int = 1) -> None:
        self.repo, self.outputs, self.default_port, self.slots = repo, outputs, port, slots
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.last_exit: int | None = None
        self._lock = threading.Lock()
        threading.Thread(target=self._supervise, name="engine-supervisor", daemon=True).start()

    # ---------------------------------------------------------- discovery
    @property
    def url(self) -> str:
        try:
            info = json.loads((self.outputs / ".engine.json").read_text())
            return f"http://{info.get('host', '127.0.0.1')}:{int(info['port'])}"
        except (OSError, ValueError, KeyError):
            return f"http://127.0.0.1:{self.default_port}"

    def _call(self, method: str, path: str, body: Any = None, timeout: float = 5.0) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail", str(exc))
            except ValueError:
                detail = str(exc)
            raise EngineError(str(detail), exc.code) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise EngineError(f"engine unreachable at {self.url}: {exc}", 503) from exc

    def online(self) -> bool:
        try:
            self._call("GET", "/health", timeout=0.6)
            return True
        except EngineError:
            return False

    def status(self) -> dict[str, Any]:
        base = {"url": self.url, "managed": self.proc is not None and self.proc.poll() is None,
                "restarts": self.restarts, "last_exit": self.last_exit}  # fmt: skip
        try:
            s = self._call("GET", "/status", timeout=2.0)
        except EngineError as exc:
            return {**base, "online": False, "error": str(exc)}
        s.pop("gpu", None)  # the console samples the GPU itself, every second
        return {**base, "online": True, **s}

    # ---------------------------------------------------------- lifecycle
    def start(self, warm: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.online():
                return {"started": False, "reason": "already online"}
            cmd = [str(Path(sys.executable).with_name("drone3d")), "engine", "--port", str(self.default_port),
                   "--outputs", str(self.outputs), "--slots", str(self.slots)]  # fmt: skip
            if warm is not None:
                cmd += ["--warm", ",".join(warm)]
            env = {**os.environ}
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if Path("/usr/local/cuda-13.3").is_dir():
                env.setdefault("CUDA_HOME", "/usr/local/cuda-13.3")
            log = self.repo / "logs" / "engine.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as out:
                self.proc = subprocess.Popen(cmd, cwd=self.repo, stdout=out, stderr=subprocess.STDOUT, env=env,
                                             start_new_session=True)  # fmt: skip
        for _ in range(100):  # torch import + CUDA init: a few seconds
            if self.online():
                return {"started": True, "pid": self.proc.pid}
            if self.proc.poll() is not None:
                raise EngineError(
                    f"engine exited during start (code {self.proc.returncode}); see logs/engine.log"
                )
            time.sleep(0.2)
        raise EngineError("engine did not come up within 20 s; see logs/engine.log", 504)

    def stop(self, force: bool = False) -> dict[str, Any]:
        result = self._call("POST", "/shutdown", {"force": force})
        if self.proc is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=15)
        return result

    def _supervise(self) -> None:
        """Restart the engine after a sticky CUDA fault, whether we started it or found it running.

        An engine that exits for a restart leaves ``.engine_poisoned`` (and its
        queue and warm list); one we started also reports exit code 3.
        """
        marker = self.outputs / ".engine_poisoned"
        while True:
            time.sleep(1.0)
            proc = self.proc
            if proc is not None and proc.poll() is not None:
                self.last_exit = proc.returncode
                self.proc = None
            if marker.is_file() and not self.online():
                marker.unlink(missing_ok=True)
                self.restarts += 1
                with contextlib.suppress(EngineError):
                    self.start(warm=None)  # the engine re-warms what it held from .engine_warm.json

    # ------------------------------------------------------------ forwarding
    def load(self, key: str) -> Any:
        return self._call("POST", f"/models/{key}/load", {}, timeout=120)

    def unload(self, key: str) -> Any:
        return self._call("POST", f"/models/{key}/unload", {}, timeout=30)

    def warm(self, keys: list[str] | None = None, config: dict | None = None) -> Any:
        return self._call("POST", "/models/warm", {"keys": keys, "config": config}, timeout=300)

    def unload_all(self) -> Any:
        return self._call("POST", "/models/unload", {}, timeout=60)

    def submit(self, name: str, config: dict) -> Any:
        return self._call("POST", "/jobs", {"name": name, "config": config}, timeout=15)

    def cancel(self, name: str) -> Any:
        return self._call("POST", f"/jobs/{name}/cancel", {}, timeout=10)

    def live_start(self, body: dict) -> Any:
        return self._call("POST", "/live", body, timeout=30)

    def live_stop(self, name: str) -> Any:
        return self._call("POST", f"/live/{name}/stop", {}, timeout=10)

    def capabilities(self) -> Any:
        return self._call("GET", "/capabilities", timeout=30)
