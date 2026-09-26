# Test suite

Run with `uv run pytest`. **216 tests** across 15 files, all fast and
deterministic — no network access and no external binaries required.
Whole suite runs in well under a minute; coverage is **77%** (`--cov=drone3d`).

## Layout

| File | Tests | Covers |
| --- | --- | --- |
| `test_telemetry.py` | 33 | CSV / DJI SRT / GPX / JSON parsing, interpolation, summary |
| `test_config.py` | 20 | YAML loading, `--set` override coercion, validation |
| `test_preprocess.py` | 17 | quality scoring, keyframe selection, masks, deblur |
| `test_projection.py` | 17 | intrinsics from FoV, GSD, `LocalTangentPlane` |
| `test_features.py` | 16 | SIFT/ORB detection, ratio matching, frame overlap graph |
| `test_geo.py` | 15 | ECEF/ENU round-trips, Umeyama similarity, GPS RMSE |
| `test_metrics_quality.py` | 15 | Chamfer vs brute force, bounds, voxel coverage, completeness |
| `test_preprocess_filters.py` | 14 | stabilization, Wiener deblur, dynamic masking |
| `test_colmap_model.py` | 14 | COLMAP `images.txt` / `points3D.txt` parsing |
| `test_dense_mesh.py` | 11 | dense MVS stage argv, Open3D/Trimesh meshing |
| `test_cli.py` | 10 | CLI entry points (`doctor`, `init-config`, `run`) |
| `test_ply.py` | 8 | PLY binary round-trip (interleaved vertex records) |
| `test_sfm_colmap_backend.py` | 8 | COLMAP argv construction, mask wiring |
| `test_io_video.py` | 6 | frame sampling, `max_frames` budget |
| `test_mesh_colmap_mesher.py` | 4 | Poisson→Delaunay fallback |
| `test_types.py` | 4 | `slots` dataclass serialisation |
| `test_pipeline_smoke.py` | 4 | end-to-end run, graceful degradation |

## Conventions

- Keep tests fast and deterministic; **no network access**.
- Heavy backends (COLMAP, Open3D, torch) are **not** required: tests stub
  `run_command`/`which` and assert on the argv the backend would execute.
- External tooling lives outside the package (`.tools/`, git-ignored) and is
  never imported by tests.
- Optional markers `slow` and `gpu` are declared in `pyproject.toml` for tests
  that do need a real backend or device.

## Writing a new test

If you are testing a wrapper around an external binary, stub `run_command` and
`which`, then assert on the **argv** — that keeps the suite hermetic while still
locking the command contract. See `test_sfm_colmap_backend.py` or
`test_dense_mesh.py`.

Two traps worth knowing, both hit while writing this suite:

- `cv2.VideoWriter_fourcc` / `cv2.SIFT_create` / `cv2.ORB_create` are real
  runtime attributes that the OpenCV 5.x type stubs do not model. LSP errors
  about them are false positives.
- `zip(a, a[1:], strict=True)` raises — the slices are never equal length.

## Not yet covered

- `dense/mono_depth.py` — needs torch + a model download (see the `ai` extra).
- `mesh/texturing.py` Poisson path end-to-end — needs Open3D at runtime and is
  slow; only its guard paths are covered.
- `georef` end-to-end through `Pipeline._stage_georef` with **real** GPS —
  blocked on data (F7): the bundled sample video carries no GPS. The maths
  underneath is covered by `test_geo.py`.
