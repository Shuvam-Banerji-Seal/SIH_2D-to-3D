# Test suite

Run with `uv run pytest`. 42 tests, all fast and deterministic — no network
access, no GPU, no external binaries.

## Layout

| File | Tests | Covers |
| --- | --- | --- |
| `test_types.py` | 4 | `TelemetrySample.to_dict` (slots), `has_position` |
| `test_io_video.py` | 6 | frame sampling, `max_frames` budget on unknown-length video |
| `test_sfm_colmap_backend.py` | 8 | COLMAP argv construction, mask wiring, model selection |
| `test_mesh_colmap_mesher.py` | 4 | Poisson→Delaunay crash fallback, argv correctness |
| `test_metrics_quality.py` | 5 | Chamfer against a brute-force reference, memory clamping |
| `test_geo.py` | 15 | ECEF/ENU round-trips, Umeyama similarity fit, GPS RMSE |

## Conventions

- Keep tests fast (whole suite < 1 s) and deterministic; no network access.
- Heavy backends (COLMAP, Open3D, torch) are **not** required: tests stub
  `run_command`/`which` and assert on the argv the backend would execute.
- External tooling lives outside the package (`.tools/`, git-ignored) and is
  never imported by tests.
- Optional markers `slow` and `gpu` are declared in `pyproject.toml` for tests
  that do need a real backend or device.

## Not yet covered

- `preprocess/` (quality metrics, keyframe selection, deblur)
- `io/telemetry.py` (CSV/SRT/GPX/JSON parsing + interpolation)
- `geo/georef.py` end-to-end through `Pipeline._stage_georef` — blocked on F7:
  the shipped sample video carries no GPS, so there is nothing to align.
  See `docs/roadmap.md` M3.
- `utils/ply.py` round-trips
