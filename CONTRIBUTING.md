# Contributing

Thanks for improving the single-pass drone → 3D pipeline. This is a hackathon
project, so keep changes focused, documented and reproducible.

## Setup

```bash
uv sync                  # core + dev dependencies from uv.lock
uv sync --all-extras     # plus torch/CUDA, pycolmap, Open3D, the web app ... (needed by most tests)
make dev                 # `uv sync` plus the pre-commit hooks
uv run drone3d doctor    # what is installed, what is missing
```

## Workflow

1. Create a branch: `feat/<scope>`, `fix/<scope>`, `docs/<scope>`.
2. Make the change; keep commits small and imperative
   (e.g. `keyframes: pick the sharpest frame inside the overlap band`).
3. Run the checks:
   ```bash
   make lint && make format && make test
   uv run drone3d doctor
   ```
4. Open a PR using the template and describe how you tested it.

## Conventions

- Python ≥ 3.12, type hints on public functions, `from __future__ import annotations`.
- Ruff for lint + format (line length 100); no comments that restate the code.
- Stages read/write artifacts inside the run directory — do not introduce
  hidden state or cross-stage imports beyond the documented contracts.
- Heavy dependencies go behind an extra (`gpu`, `depth`, `sfm`, `render`, `mesh`, `geo`, `api`), are
  imported inside the function that needs them, and fail with a clear `BackendUnavailable`
  message. Whatever the code imports must be declared in `pyproject.toml` (and listed in
  `drone3d doctor`); do not rely on another package pulling it in.
- New config keys: add a dataclass field in `src/drone3d/config.py`, document it
  in `configs/default.yaml`, and validate it in `PipelineConfig.validate()`.
- New stages: add the name to `ALL_STAGES` in `src/drone3d/config.py` and a `_stage_<name>`
  method to `Pipeline` in `src/drone3d/pipeline.py` that reads and writes its run-directory files.

## Data hygiene

Never commit drone footage, datasets, model weights or run outputs. They are
git-ignored (`data/`, `outputs/`, `*.ply`, `*.mp4`, ...). Use small synthetic
clips in tests instead.

## Testing

- Fast, deterministic, offline unit tests; see `tests/README.md`.
- A test that needs torch/CUDA, Open3D, ... guards itself with `pytest.importorskip` or
  `skipif(not torch.cuda.is_available())` so collection never fails on a smaller install.
- Add a regression test with any bug fix.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/` and include your
`drone3d doctor` output, config overrides and the failing stage message.
