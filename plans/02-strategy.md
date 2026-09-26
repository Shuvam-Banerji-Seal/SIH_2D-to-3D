# 02 — Strategy

Written 2026-09-26 (cycle 2). Decision record for the two asks:
**(a)** evaluate `spirula-studio` as an addition · **(b)** "setup uv init and uv sync".

---

## The user's strategy, restated (§9.1)

> Add `harry7557558/spirula-studio` to this project, and set up the uv environment.

Two independent sub-goals. Treated as hypotheses and validated separately.

---

## (b) uv init + uv sync — **VALID, with one correction**

| Sub-ask | Verdict | Evidence |
| --- | --- | --- |
| `uv init` | **No-op** — already initialized; `uv` errors out by design | `/tmp/opencode/uvinit-probe`: `error: Project is already initialized (pyproject.toml file exists)`; `README.md` byte-identical before/after |
| `uv sync` | **Done** | `.venv` on CPython 3.12.8, 22 pkgs, `drone3d 0.1.0` installed |

Post-sync verification (all executed):

| Check | Result |
| --- | --- |
| `uv run drone3d --help` | ✅ CLI dispatches |
| `uv run drone3d doctor` | ✅ exit 0; core deps OK, colmap ✗, ffmpeg ✗→OK |
| `uv run ruff check .` | ✅ *All checks passed!* |
| `uv run ruff format --check .` | ✅ *54 files already formatted* |
| `uv run pytest` | ⚠️ *no tests ran* — **zero tests exist** |

Carry-forward action: `VIRTUAL_ENV=/store/shuvam/.venv` is set in the shell and
mismatches the project `.venv`, producing a warning on every `uv` invocation.
Fix by unsetting it or exporting the project path — **do not** edit the repo for this.

---

## (a) spirula-studio — **PARTIALLY VALID → recommend DO NOT ADD NOW**

### Classification: **FLAWED as stated, RECOVERABLE as a narrower thing.**

**Where it is right**
- It genuinely addresses a real blocker: **F8 (COLMAP absent)** — it ships built-in
  SfM with no COLMAP install, and its mesh writers emit **PLY**, which our
  `mesh/` and `dense/` stages already consume (`MeshExport.h`).
- The hardware does support it: A100 80 GB + driver 610.43.02 + `nvidia_icd.json`
  ⇒ the Vulkan backend would run (GPU, not lavapipe).
- It is actively developed (commits through 2026-09-23) with a real CLI.

**Where it is flawed**

| # | Flaw | Evidence | Impact |
| --- | --- | --- | --- |
| S1 | **Not installable by `uv`** — it is a C++/Vulkan binary, not a Python distribution | README build section (CMake/Ninja/Vulkan SDK); `docs/architecture.md`: "Nothing is left in Python" | The two asks are unrelated; `uv sync` cannot pull it in |
| S2 | **Licence incompatible**: theirs **GPL-3.0**, ours **GPL-2.0-only** | `LICENSE` raw → "Version 3"; `pyproject.toml` `license = { text = "GPL-2.0-only" }`; `LICENSE` → "Version 2, June 1991" | Cannot vendor, link, or bundle. **Arm's-length subprocess only** (same category as shelling out to `ffmpeg`). Bundling it in a release would force a relicence |
| S3 | **Out of scope for the PS**: zero mention of splatting/NeRF anywhere in our problem statement or code | `grep -rin "splat\|gaussian\|nerf\|neural radiance" docs/ src/ README.md configs/` → only `GaussianBlur` PSF + mixture-of-Gaussians background model | PS criteria 1–8 are mesh accuracy, scale, completeness, georeferencing — **none scored by 3DGS** |
| S4 | **It re-implements 4 of our 8 stages** (ingest, preprocess/dynamic-masking, sfm, mesh) | README feature list | Adding it means writing **adapters**, not deleting code — net *more* work |
| S5 | **It would break our `georef` contract**: `georef` reads `sparse_txt/{cameras,images}.txt` + `ingest/frames.csv`. Spirula *parses* COLMAP; no evidence it **writes** it | architecture.md parser table = read-only | Our scored criteria 2 & 6 (scale ≤2 %, GPS RMSE ≤3 m) run through our own Umeyama stage |
| S6 | Single maintainer, bus factor 1 | README: "almost entirely by one person" | Supply-chain/continuity risk for a competition deliverable |
| S7 | COLMAP is the **strictly cheaper** fix for F8 | `apt-cache policy colmap` → 3.7-2; `conda search colmap` → 4.2.0 (cpu/cuda_129/cuda_130) | Installable today, outputs *exactly* the layout our code already parses, **BSD-3-Clause** (no licence issue), zero adapter |

**Rejected alternative:** "Install Spirula *instead* of COLMAP."
Rejected because S5 — it silently removes the inputs our georef/metrics stages
depend on, i.e. it would regress criteria 2 and 6 while appearing to fix F8.

### Decision

**D1 — Do not add Spirula Studio now.** Rationale: S1 (not a `uv` dep) + S3 (out of
scope) + S4/S5 (adapter cost > benefit) + S7 (COLMAP is cheaper and licence-clean).
Confidence: **[DERIVED]** from verified findings.

**D2 — Fix F8 via COLMAP first** (`apt install colmap` = 3.7-2, or conda-forge 4.2.0),
because it unblocks stages 3–5 with zero code changes. Confidence: **[DERIVED]**.

**D3 — Re-open Spirula later as an *optional* `splat` stage / alternative mesh
backend**, under these conditions (any one):
- COLMAP install fails or is unavailable to judges; **or**
- we deliberately choose to ship a bonus 3DGS deliverable for the demo (A100 is
  present and otherwise unused by us); **or**
- it is confirmed to **write** a COLMAP-format sparse model (closes UNKNOWN R1).
Integration rule if pursued: **subprocess at arm's length, never bundled** —
documented as an external tool exactly like `ffmpeg`/`colmap`, so the GPL-2.0-only
licence of this repo is untouched. Confidence: **[HYPOTHESIS]** pending the
COLMAP-write UNKNOWN.

**D4 — Not in scope this cycle: relicensing.** If deeper Spirula integration is ever
wanted, the cheapest fix is `GPL-2.0-or-later` → then GPL-3.0 code may be combined.
Requires owner approval; logged, not acted on. Confidence: [HYPOTHESIS] on the
legal reading — arm's-length invocation is the conservative position regardless.

### Next actions this cycle (ordered)

1. ✅ `uv sync` + verify (done, see (b)).
2. ⬜ Record D1–D4 + the `R2-a` reversal in `plans/04-decisions.md`.
3. ⬜ Install COLMAP (D2) → unblocks `sfm`/`dense`/`mesh`.
4. ⬜ Fix confirmed defects F1, F3, F4 with regression tests (F2 next).
