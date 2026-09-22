# Upstream audit

Updated: 2026-08-27 UTC. All refs were re-fetched after relocation; discovery pins are not dependency pins.

## Foundation decision

Keep `natpate/ninfer-windows` as the Windows foundation. Current local `master` is a maintained fork with native MSVC/vcpkg, FFmpeg/libcurl, vision, serving, and Blackwell paths. `Neroued/ninfer` remains the authority for model semantics and the context/resource architecture. Do not replace the base with q27, vLLM, or a snapshot overlay.

## Current refs

| Remote/ref | Fetched head | Disposition |
| --- | --- | --- |
| `upstream/master`, `upstream/dev` | `b686696eebd4` | Windows base; current branches coincide |
| `canonical/master` | `9dbc074005a1` | Canonical authority; now includes the former dev resource cutover and later hardening |
| `canonical/dev` | `9dbc074005a1` | Refreshed head; current master/dev coincide; local staged tree is qualified only through `59febed27ca2` |
| `headpiece-reference/main` | `c1b0ad34d438` | Patch donor only; do not adopt snapshot history |

## Canonical dev map

The dev series is a coupled runtime cutover, not a safe list of independent cherry-picks:

- `d6af046a`: closes Engine execution ownership boundaries and introduces the resource-manager/scheduler contract.
- `c648a132`: adds StateImage physical containers.
- `9d4cc6f9`: adds paged-KV physical containers and host-KV foundations.
- `f7bcd2ba`: implements prefix-caching resource scheduling; depends on the above and changes Engine/Program APIs broadly.
- `3a011c70`: provisions the default host context cache.
- `ab6ab5a8`: restores agent-prefix cache reuse.
- `dda31c75`: adds measured context-cost scheduling.
- `0e6d4f45`: indexes prefix checkpoints by content.
- `8554dfa0`: partitions host KV extents by run.
- `e0829866`: restores anonymous prefix reuse.
- `fc5c4834`: reclaims transient vision workspace.
- `e4e6f897` through `59febed2`: harden response ownership, media reuse, prefill boundaries, host timing, and warmup/cache behavior.

The 2026-08-27 refresh adds the following post-`59febed2` work: `3e903b704c` pressure-aware materialization planning and the coupled `PressurePlanningSession` cutover; `55222c546a` architecture refresh; `081c6039f9` device binding; `fbd04729c8` sparse-MoE prefill synchronization; `ab82f88603` readiness gating on successful warmup; `79c292bc07` typed tool arguments; `0e4cdf84f0` schema-aware tool arguments; `780d576791` literal control-token provenance; and `9dbc074005` valid long-anchor publication. These commits are not yet integrated into the staged Windows tree.

Canonical dev is now at the same head as canonical master. Its first Engine/resource commit and the later pressure planner replace core lifecycle and public/internal runtime assumptions; partial cherry-picks can produce stale mappings or invalid ownership. The current staged tree remains the last locally qualified slice through `59febed2`; the next refresh must use a dedicated worktree, reapply Windows portability, and rerun the full build/CTest plus Qwen3.8 artifact gates before promotion.

## Windows preservation ledger

Local Windows work that must survive integration includes the MSVC/TMA fixes, aligned allocation portability in tests, vcpkg/CUDA target selection, media/build handling, README/Windows docs, and integrity CI commits `4acd1611`, `10173f32`, and `3e2a28be`. Canonical dev has no identified Windows-specific integration commit. Never merge dev wholesale without a clean Windows reapplication and CTest run.

## Refresh status

The refs were refreshed on 2026-08-27. The new canonical head is recorded above, but no post-`59febed2` code has been promoted to the working tree because the coupled pressure-planning API conflicts with the staged Windows slice. Keep the refresh as the next isolated integration experiment rather than silently claiming the old qualification covers it.

## Qualification gates

Before integration: exact baseline manifest, Qwen3.8 route test, serving smoke, state/page tests, and a rollback tag/worktree. After each coupled slice: configure/build, focused unit tests, full CTest, identity/ownership checks, and no Windows path regressions. For the complete stack: private/shared restore, partial-page COW, cancellation/eviction, host enabled/disabled, first-repeat and second-repeat with MTP, media reuse, C1-C8, and soak.

No upstream code has been integrated from this audit. Full source-level findings from the read-only audit are recorded in the session evidence and summarized here to keep provenance reviewable.

## Publication provenance note (2026-09-22)

The publication snapshot consolidates later local Windows, Engine, EXL3, DFlash2, OSCAR, and test
work on top of the existing public Windows history. It was constructed from the public `master`
baseline rather than publishing local recovery history that contained generated receipts and large
artifacts. The source snapshot before publication-only edits contained 1,283 paths with aggregate
SHA-256 `642169e8ccc96cfe4115868a399a3af0f6e22ffccd90667e1aea6a540b348425`.

`src/exl3/mia_exllamav3` contains adapted utilities from MiaAI-Lab/exllamav3. Its local `LICENSE`
file is retained adjacent to the adapted source. Those utilities are not evidence that the full Mia
engine is embedded or that VoidInfer matches Mia performance. Algorithm papers and upstream projects
are research references unless code is explicitly present. The surrounding Engine integration,
Windows portability, ownership contracts, qualification harnesses, and publication evidence are
project integration work; commit authorship alone is not used to infer who manually wrote each line.

The repository remains under its existing Apache-2.0 license. This audit does not replace or
relicense third-party notices, model-weight licenses, or dependency terms.
