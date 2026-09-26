# 01 — Research archive

Started 2026-09-26 (cycle 2). Every claim below carries a source fetched **this session**.

---

## R1: Is `harry7557558/spirula-studio` a good addition?

**Queries**
1. `https://github.com/harry7557558/spirula-studio` → README (fetched)
2. `harry7557558 spirula-studio github` → websearch 8 results
3. `raw.githubusercontent.com/.../master/LICENSE` → **GPL v3 text** (fetched)
4. `api.github.com/repos/.../git/trees/master?recursive=1` → full file tree (fetched)
5. `api.github.com/repos/.../contents/docs` → docs index (fetched)
6. `raw.../docs/README.md` + `raw.../docs/architecture.md` → doc index + layer diagram (fetched)
7. `raw.../src/mesh/MeshExport.h` → **export format surface** (fetched)
8. `Spirula Studio CLI "spirula train" export mesh PLY telemetry` → websearch (weak results)

**Findings** (Source = URL fetched)

| # | Finding | Source | Confidence |
| --- | --- | --- | --- |
| 1 | C++ 3D **Gaussian Splatting** trainer, one self-contained binary; video → splat → textured mesh | README | [VERIFIED] |
| 2 | "No Python/PyTorch, no separate COLMAP install" — has **built-in SfM, AI masking (SAM), frame extraction** | README | [VERIFIED] |
| 3 | **License = GNU GPL v3** ("Version 3, 29 June 2007"); AUR + radiancefields independently say GPL-3.0(-or-later) | LICENSE raw + AUR + radiancefields | [VERIFIED] |
| 4 | CLI exists: `spirula --help`, `spirula train`; TrainerCore shared by CLI and GUI | README + architecture.md | [VERIFIED] |
| 5 | Mesh writers: **PLY, OBJ(+mtl), glTF, GLB, STL**; colour modes vertex/texture | `src/mesh/MeshExport.h` | [VERIFIED] |
| 6 | **Reads** COLMAP / Nerfstudio / Metashape dataset layouts | architecture.md table | [VERIFIED] |
| 7 | Nothing left in Python — "the Python halves were deleted"; a note exists on "retiring the Python client" | architecture.md + docs index | [VERIFIED] |
| 8 | 2026-09-10: dataset module uses **telemetry metadata for metric scale + orientation** | README News | [VERIFIED] |
| 9 | 2026-09-23: editing/rendering export added; 2026-08-08: end-to-end video→splat→mesh in GUI **and CLI** | README News | [VERIFIED] |
| 10 | Maintained "almost entirely by one person"; issues reviewed but replies late | README | [VERIFIED] |
| 11 | Trains 10M SH3 Gaussians in 8 GB VRAM; Vulkan runs NVIDIA/AMD/Intel/Apple | README | [VERIFIED] |

**Conflicts**
- None material. GPL-3.0 (LICENSE file) vs "GPL-3.0-or-later" (AUR packaging) is a *packaging* description, not a conflict with the primary source.

**Conclusion** [VERIFIED]
Technically strong, actively developed, CLI-able. But it is a **C++ binary, not a Python package** — `uv` cannot install it — and its licence (GPL-3.0) is incompatible with this repo's `GPL-2.0-only` for anything other than arm's-length subprocess invocation.

**Open**
- Does it **write** a COLMAP-format `sparse/` model, or only read one? [UNKNOWN] — decisive for our `georef` stage.
- Does its dataset creation emit GeoJSON/EPSG? [UNKNOWN].
- Actual `spirula train --help` flag list not yet retrieved (repo docs have no `cli.md`; authoritative list is `src/config/TrainConfig.h`).

---

## R2: Environment after `uv sync`

**Queries**
1. `uv init .` in a directory with `pyproject.toml` (in `/tmp/opencode/uvinit-probe`, isolated) → **`error: Project is already initialized`**, `README.md` untouched
2. `uv sync` in the repo → exit 0
3. `uv run drone3d --help` / `doctor` / `ruff check .` / `ruff format --check .` / `pytest`
4. GPU/Vulkan/colmap probing commands

**Findings**

| # | Finding | Confidence |
| --- | --- | --- |
| 1 | `uv init` **refuses** on an initialized project and does **not** clobber files → the `uv init` half of the request is a no-op by design | [VERIFIED: executed in /tmp] |
| 2 | `uv sync` created `.venv` on **CPython 3.12.8** (`/store/miniforge3/bin/python3`), installed **22 packages**, built + installed `drone3d 0.1.0` editable | [VERIFIED: executed] |
| 3 | `ruff check .` → *All checks passed!*; `ruff format --check .` → *54 files already formatted* | [VERIFIED: executed] |
| 4 | `pytest` → *no tests ran in 0.00s*, exit 0 → confirms **zero tests exist** (Findings F-table in 00-understanding.md) | [VERIFIED: executed] |
| 5 | `doctor`: core deps OK (numpy/cv2/yaml/tqdm), all 9 optional extras missing, **colmap not on PATH**, ffmpeg/ffprobe OK, exiftool missing | [VERIFIED: executed] |
| 6 | Cosmetic but noisy: env var `VIRTUAL_ENV=/store/shuvam/.venv` ≠ project `.venv` → warning on **every** uv command | [VERIFIED: executed] |
| 7 | GPU = **NVIDIA A100 80GB PCIe**, driver **610.43.02** | [VERIFIED: nvidia-smi] |
| 8 | Vulkan loader `libvulkan.so.1` present; ICD manifests include **`nvidia_icd.json`** | [VERIFIED: /usr/share/vulkan/icd.d/] |
| 9 | COLMAP installable: apt candidate **3.7-2**; conda-forge **4.2.0** (cpu/cuda_129/cuda_130) | [VERIFIED: apt-cache + conda search] |

**REVERSALS recorded** (see 04-decisions.md)
- `R2-a`: I first inferred "no NVIDIA Vulkan" from `ldconfig -p` showing only mesa/radeon/lvp ICDs. **Wrong.** The Vulkan loader resolves ICDs from JSON manifests in `/usr/share/vulkan/icd.d/`, where `nvidia_icd.json` exists. Hypothesis retracted → NVIDIA Vulkan **is** available.
