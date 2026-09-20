# Contributing

Thanks for improving the single-pass drone → 3D pipeline. This is a hackathon
project, so keep changes focused, documented and reproducible.

## Setup

```bash
uv sync          # runtime + dev dependency group from uv.lock
make dev         # same, plus pre-commit hooks
uv run drone3d doctor
```

## Workflow

1. Create a branch: `feat/<scope>`, `fix/<scope>`, `docs/<scope>`.
2. Make the change; keep commits small and imperative
   (e.g. `preprocess: add Tenengrad fallback for low-texture frames`).
3. Run the checks:
   ```bash
   make lint && make format && make test
   uv run drone3d doctor
   ```
4. Open a PR using the template and describe how you tested it.

## Conventions

- Python ≥ 3.10, type hints on public functions, `from __future__ import annotations`.
- Ruff for lint + format (line length 100); no comments that restate the code.
- Stages read/write artifacts inside the run directory — do not introduce
  hidden state or cross-stage imports beyond the documented contracts.
- Optional heavy dependencies go behind extras (`ai`, `sfm`, `mesh`, `geo`,
  `api`) and must degrade with a clear `BackendUnavailable` message.
- New config keys: add a dataclass field in `src/drone3d/config.py`, document it
  in `configs/default.yaml`, and validate it in `PipelineConfig.validate()`.
- New backends: implement the relevant `is_available()` / stage interface and
  register it in the factory function.

## Data hygiene

Never commit drone footage, datasets, model weights or run outputs. They are
git-ignored (`data/`, `outputs/`, `*.ply`, `*.mp4`, ...). Use small synthetic
clips in tests instead.

## Testing

- Fast, deterministic, offline unit tests by default.
- Mark tests needing COLMAP/Open3D/torch with `@pytest.mark.slow`.
- Add a regression test with any bug fix.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/` and include your
`drone3d doctor` output, config overrides and the failing stage message.
