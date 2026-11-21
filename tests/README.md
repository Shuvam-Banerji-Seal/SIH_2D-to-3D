# Test suite

Run with `uv sync --all-extras && uv run pytest`. The tests are fast and deterministic: no
network access, no model downloads, no external binaries. Tests that need an optional library
or a GPU skip themselves (`pytest.importorskip`, `skipif(not torch.cuda.is_available())`), so the
suite also runs on a CPU-only machine with only part of the extras installed.

```bash
uv run pytest                          # everything this machine can run
uv run pytest tests/test_geo.py -k enu # one file, one test
uv run pytest --cov=drone3d            # with coverage
```

## Layout

| Area | Files |
| --- | --- |
| Configuration, CLI, run bookkeeping | `test_config`, `test_cli`, `test_types`, `test_report`, `test_shell`, `test_thread_binding` |
| Dependencies and model paths | `test_dependencies` (everything `src/` imports is declared in `pyproject.toml`), `test_model_paths` |
| Telemetry, geodesy, georeferencing | `test_telemetry`, `test_projection`, `test_geo`, `test_georef_straight_track` |
| Ingest and keyframes | `test_nv12`, `test_sidecar_decode`, `test_nvjpeg_threads`, `test_raft_graph_cache`, `test_keyframe_geometry`, `test_overlay` |
| Fast profile: tracks, dense depth, fusion | `test_fastsfm_tracks`, `test_fastsfm_components`, `test_fastsfm_dense`, `test_fastsfm_mono`, `test_fusion_worker`, `test_merge_passes`, `test_depth_refine` |
| Accurate profile: depth prior, splats | `test_depth_align`, `test_depth_calibration_and_crop`, `test_splats`, `test_spirula_watchdog`, `test_mesh_reuse` |
| Export and metrics | `test_ply`, `test_export_formats`, `test_render_mesh`, `test_texture_gpu`, `test_terrain`, `test_metrics_quality` |
| Complete and generated models | `test_complete`, `test_generate` |
| Engine and web console | `test_engine`, `test_console_api` |

## Conventions

- Test the maths against known answers (exact projections, recovered depths, round trips), not
  against a snapshot of the current output. Round-trip tests compare **values**, never only shape
  and dtype.
- Anything that needs a GPU or an optional library guards itself at module level so that collection
  never fails on a smaller install.
- Wrappers around external programs (`utils/shell.py`, the spirula and ffmpeg launchers) are tested
  by stubbing the subprocess and asserting on the argv.
- Add a regression test with every bug fix.

## Not covered here

- A full run on real footage: it needs a GPU, the spirula build, the model weights and the sample
  videos. `experiments/` holds the scripts that exercise those and write the numbers the paper uses.
- Georeferencing against a real flight log: none of the public sample videos carries GPS. The maths
  underneath is covered by `test_geo.py` and `test_georef_straight_track.py`.
