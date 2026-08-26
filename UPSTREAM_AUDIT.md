# Upstream audit

Updated: 2026-08-26 UTC. All refs were fetched after relocation; discovery pins are not dependency pins.

## Foundation decision

Keep `natpate/ninfer-windows` as the Windows foundation. Current local `master` is a maintained fork with native MSVC/vcpkg, FFmpeg/libcurl, vision, serving, and Blackwell paths. `Neroued/ninfer` remains the authority for model semantics and the context/resource architecture. Do not replace the base with q27, vLLM, or a snapshot overlay.

## Current refs

| Remote/ref | Fetched head | Disposition |
| --- | --- | --- |
| `upstream/master`, `upstream/dev` | `b686696eebd4` | Windows base; current branches coincide |
| `canonical/master` | `feaf4dd0983f` | Canonical authority; six commits ahead of local merge base |
| `canonical/dev` | `59febed27ca2` | 58 commits ahead of canonical master; first architecture integration target |
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

Canonical dev is 58 commits ahead of canonical master at the fetched head. Its first Engine/resource commit replaces core lifecycle and public/internal runtime assumptions; partial cherry-picks can produce stale mappings or invalid ownership. The safe first experiment is a branch-level cutover of StateImage + paged-KV + Engine/resource foundations, followed by a build and focused state/KV/runtime tests before prefix scheduling. Reapply local Windows portability as soon as the first compile boundary is reached.

## Windows preservation ledger

Local Windows work that must survive integration includes the MSVC/TMA fixes, aligned allocation portability in tests, vcpkg/CUDA target selection, media/build handling, README/Windows docs, and integrity CI commits `4acd1611`, `10173f32`, and `3e2a28be`. Canonical dev has no identified Windows-specific integration commit. Never merge dev wholesale without a clean Windows reapplication and CTest run.

## Qualification gates

Before integration: exact baseline manifest, Qwen3.8 route test, serving smoke, state/page tests, and a rollback tag/worktree. After each coupled slice: configure/build, focused unit tests, full CTest, identity/ownership checks, and no Windows path regressions. For the complete stack: private/shared restore, partial-page COW, cancellation/eviction, host enabled/disabled, first-repeat and second-repeat with MTP, media reuse, C1-C8, and soak.

No upstream code has been integrated from this audit. Full source-level findings from the read-only audit are recorded in the session evidence and summarized here to keep provenance reviewable.
