# Maintenance and cleanup

## Repository hygiene

Build trees, raw `results/`, model/tensor artifacts, profiler captures, executables, local `.codex`
state, and `.mcp.json` are excluded from source publication. Small curated evidence belongs under
`evidence/` and must retain receipt identity, provenance, and transformation notes.

`tools/maintenance/Invoke-VoidInferCleanup.ps1` is dry-run by default. Apply mode requires exact
allowlisted roots, canonical-path containment, protected exclusions, unchanged inventory metadata,
and optional content hashes. It refuses reparse points, nested repositories, changed inputs, and
paths outside the allowlist. Its disposable-fixture test covers dry-run/apply, protected paths,
changed files, invalid roots, a nested Git directory, and a junction escape.

Recommended sequence:

1. Inventory ownership, active processes, leases, dirty worktrees, and free space.
2. Create and restore-test a source/recovery checkpoint outside the publishable tree.
3. Generate a reviewed exact-path manifest and run the utility without `-Apply`.
4. Inspect the JSONL action log, then apply the same manifest.
5. Recheck protected hashes, references, functionality, and observed free space.

Never use blanket `git clean -xfd`, extension-only deletion, aggressive Git pruning, or a recursive
delete rooted at a workspace. An ignored `.bin` can still be an irreplaceable oracle.

## Evidence promotion

Promote a change only when effective configuration, dispatch, exactness, matched workload, and
provenance support the precise claim. Preserve useful negative results. Mark an implementation as
`PREPARED_NOT_RUN` or `NOT_BUILT` when that is the actual boundary. The optimization campaign is
paused until explicitly resumed.
