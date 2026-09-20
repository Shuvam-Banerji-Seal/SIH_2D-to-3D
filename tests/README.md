# Test suite

Tests are added incrementally (see `docs/roadmap.md`). This directory is part
of the repository layout from day one so CI, coverage and conventions are
already wired up.

Planned layout:

```
tests/
├── conftest.py                 # shared fixtures (synthetic video, configs)
├── test_config.py              # config loading / --set overrides
├── test_geo.py                 # ENU round-trips, Umeyama similarity fit
├── test_telemetry.py           # CSV / SRT / GPX / JSON parsing + interpolation
├── test_preprocess.py          # quality metrics, selection, deblur
├── test_ply.py                 # PLY read/write round-trips
└── test_pipeline_smoke.py      # end-to-end run on a synthetic clip
```

Rules:

- Keep tests fast (< a few seconds) and deterministic; no network access.
- Heavy backends (COLMAP, Open3D, torch) are marked `slow` and skipped when the
  optional extra is not installed.
- Run with `uv run pytest`; coverage is reported in CI.
