# 06 — Evolution log

## Evolution Log

| Cycle | What I tried | What broke | What I learned | Spec change needed |
|-------|-------------|-----------|---------------|-------------------|
| 1 | Read every file; guess defects from source | — | Reading found only 2 of 14 real defects (F1, F3) | §5 should weight *execution* far higher than inspection |
| 2 | Validate Spirula as a dependency | Would have been illegal (GPL-3.0) + useless (not a `uv` dep) | License + packaging checks must come before feature appeal | none |
| 2 | `uv init` + `uv sync` blindly | — | Probed `uv init` in `/tmp` first; it refuses safely | A9/§3 quarantine discipline earned its keep |
| 3 | Installed COLMAP via `nohup … &` | Background job killed by tool timeout → install silently half-done | Never background long installs inside a bounded tool call | §6 should require foreground + explicit timeout for installs |
| 3 | Diagnosed F13 as "COLMAP 4 rig/frames.bin incompatibility" | **Wrong** — real cause was a missing `mkdir` | Read the *first* error line, not the stack tail | none |
| 3 | F6 depth polarity via synthetic 2-tone image | **Wrong verdict** (out-of-distribution input) | Verify with in-distribution data | §5: flag OOD fixtures explicitly |
| 3 | Ran the real pipeline | Uncovered F13/F14/F15/F16 | **6 of 14 defects only surfaced at runtime** | §8: make a real end-to-end run mandatory |
| 4 | Fixed `write_ply` colour block | Found silent data corruption | Round-trip tests must compare *values*, not just shape/dtype | §8 |
| 5 | Targeted tests by coverage % | Found F18, F19 | Coverage-driven targeting finds defects reading misses | none |
| 5 | Wrote 10 tests from guessed APIs | 10 red failures, all mine | Read the signature before writing the test | §6 |
| 6 | Re-audit after big deltas | CI's `--cov` arg + stale README drifted | Docs about tests rot fastest | §3: docs must be updated in the same commit as tests |

## Mutation History

| From strategy | To strategy | Trigger | Outcome |
|--------------|------------|---------|---------|
| Static code reading | Execute the real pipeline | 6 defects invisible to reading | 4 new defects found immediately |
| Blind `uv init` | Isolated `/tmp` probe first | A9 (never destroy) | Correctly skipped a destructive no-op |
| `nohup` background install | Foreground + long timeout | Job killed by tool timeout | Install completed |
| Synthetic test image | Real photograph | OOD verdict contradiction | Correct verdict |
| Import-everything smoke test | Per-module isolated imports | 300s hang with no attribution | Identified slow-vs-broken correctly |
| Trust `doctor`'s "OK" | Actually execute the binary | `colmap` found but `libOpenImageIO` missing | Caught a real blocker |
| Guess test APIs | Read signatures first | 10 consecutive red tests | Zero red after switching |

## Anti-patterns discovered

- **Trusting `find_spec`/`which` over execution**: `doctor` said colmap was "OK"
  while the binary could not load a shared library. → Always run the binary.
- **Backgrounding long installs in a bounded tool call**: the job died with the
  call and the log looked merely "slow". → Foreground with explicit timeout.
- **Reading the stack tail instead of the first error line**: blamed COLMAP's rig
  format for what was a missing `mkdir`. → Read the *first* diagnostic line.
- **Out-of-distribution test fixtures**: a 2-tone image "tested" depth polarity
  and gave the opposite answer to a real photo. → Use in-distribution inputs.
- **Eager maps of optional backends** (`detectors = {...}`): one missing backend
  broke *all* of them (F19). → Construct lazily, fail with a named error.
- **Docs-about-tests rot first**: `tests/README.md` was wrong within two cycles.
  → Update docs in the same commit as the tests they describe.

## Strengths confirmed

- **Running the real pipeline** is the single highest-yield verification: 6 of
  14 defects, including silent data corruption, were only visible at runtime.
- **Mutation testing** proves a test can fail; without it a green suite proves
  little. Used on F4 with a deliberate reintroduction.
- **Live-docs over memory**: fetching COLMAP's mask FAQ caught an *inverted
  polarity* that would have silently kept the vehicles and dropped the scene.
- **Hermetic backend tests** (stub `run_command`, assert argv) let the suite
  cover COLMAP/Open3D wrappers without the binaries — 216 tests in ~20 s.
- **Probing destructive-looking commands in `/tmp` first** prevented a possible
  `README.md` clobber from `uv init`.
