# 07 — Spec mutations

Proposed changes to `rigor-infinity.md`, derived from `06-evolution-log.md`.
Only mutations with confidence ≥ `[DERIVED]` are marked for application.

## Proposed Spec Mutations

| Section | Current text | Proposed change | Rationale | Confidence |
|---------|-------------|----------------|-----------|------------|
| §5 Research protocol | "Stop iterating, start reading" | Add: **"Stop *reading*, start *running*."** Make one real end-to-end execution mandatory before any defect list is declared complete. | Static reading found 2 of 14 defects; executing the pipeline found 6 more, including silent data corruption (F16) that no amount of reading would reveal. | **[DERIVED]** |
| §8 Testing | "Run the full relevant suite after every change" | Add: *"A green suite is not evidence unless you have shown it can go red. Mutation-check at least one test per defect fixed."* | Without the F4 mutation check, a vacuous test would have looked identical to a real one. | **[DERIVED]** |
| §8 Testing | Categories required | Add category: **round-trip / value-equality** tests for any binary or serialisation format. Shape-and-dtype checks pass on corrupt data. | `write_ply` corrupted every vertex past the first while shape/dtype assertions held (F16). | **[DERIVED]** |
| §5 Research protocol | Confidence tags table | Add tag **`[OOD]`** — evidence gathered on out-of-distribution input. | A synthetic 2-tone image gave the *opposite* depth-polarity verdict to a real photograph; OOD evidence must be labelled as such. | **[DERIVED]** |
| §3 Workspace discipline | "MCP-first tooling" | Add: *"Never background a long install inside a bounded tool call — run it foreground with an explicit timeout, and verify the binary executes (not merely `which`-resolvable) afterwards."* | `nohup … &` install died with its tool call and looked merely slow; `doctor` reported COLMAP "OK" via `which` while the binary could not load a shared library. | **[DERIVED]** |
| §10 Anti-hallucination | H1–H10 | Add **H11**: *"A type-checker error about an API is not proof the API is missing — stubs lag runtimes. Verify by execution before 'fixing' working code."* | OpenCV 5.x stubs omit `SIFT_create`/`VideoWriter_fourcc`, which work fine at runtime; three false "defects" nearly got "fixed". | **[DERIVED]** |
| §6 Planning | Step spec format | Add a `Docs:` field: every step that changes behaviour must name the doc it updates, in the same commit. | `tests/README.md` went stale within two cycles and described a suite that no longer existed. | **[DERIVED]** |
| §12b Meta-loop | "Never repeat a move" | Add: *"Guessing an API signature counts as the same move as the last guess. Read the signature first."* | 10 consecutive red tests were all guessed-API failures; switching to reading signatures produced zero red afterwards. | **[DERIVED]** |

## Not applied (below threshold, or axioms)

| Proposal | Why not applied |
|---|---|
| Require double-audit before any commit | `[HYPOTHESIS]` — would have blocked 9 useful incremental commits; batching raises risk |
| Drop the infinite loop when blocked on external data | **Axiom-level (A11/A12)** — needs user approval to touch §0 |
| Add coverage thresholds to CI | `[HYPOTHESIS]` — ratcheting thresholds without a baseline invites gaming |

## Application note

Per §17 these mutations edit `rigor-infinity.md` itself, which lives outside this
workspace (`~/.config/opencode/...`). **They are recorded here rather than
applied**, because mutating the shared spec without the owner seeing the diff
would be a silent change to another agent's operating rules. The table above is
the exact diff to apply on approval.
