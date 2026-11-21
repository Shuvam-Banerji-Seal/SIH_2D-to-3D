"""The warm engine's HTTP API and run naming, shared by the scripts that submit jobs to it
(``benchmark.py``, ``engine_throughput.py``)."""

from __future__ import annotations

import json
import re
import urllib.request

ENGINE = "http://127.0.0.1:8770"


def call(path: str, body: dict | None = None) -> dict:
    """GET ``path`` on the engine, or POST ``body`` to it as JSON."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        ENGINE + path, data=data, headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST"
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read())


def slug(name: str, n: int = 40) -> str:
    """``'Qutub Minar, New Delhi [47YrmPhMeho].webm'`` -> ``'qutub_minar_new_delhi'`` (at most ``n`` characters)."""
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"\s*\[[^\]]+\]\.\w+$", "", name).lower()).strip("_")[:n].strip("_")
