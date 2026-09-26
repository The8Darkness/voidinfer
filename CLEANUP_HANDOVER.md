# Cleanup handover — 2026-09-22

Status: **CLEANUP_IMPLEMENTED_CPU_CHECKED_GPU_PENDING**.
Engine/CUDA/harness: **NOT_BUILT; NOT_RUN_GPU_FORBIDDEN**.

## Source and continuation authority

- Continue in `D:\AI\voidinfer-astra-cleanup-20260922` on
  `refactor/astra-high-cleanup-20260922` with its uncommitted changes.
- B0 is publication commit `b7505d07df6114e3039c1783b19b09f38ed3ee25`.
  Publication main remains clean and untouched.
- B1 is B0 plus this cleanup only. No optimization candidate was added or enabled.
- B2 must be a later candidate on B1, with the cleanup-only snapshot retained.
- No commit, merge, push, replacement session or subagent was created.
- Research `D:\AI\voidinfer-adaptive-dflash2` remains at old HEAD `dd398429`
  with its newer dirty work preserved. Current research/publication `src` differed
  only by one whitespace fix in `dflash2_draft.cu`; `include` and `apps` matched.
  Publication test differences were the MTP host-test linkage fix, portable artifact
  paths, maintenance fixtures and whitespace. None required importing research code.
- The suggested `voidinfer-gaming-prep-20260922` tree was absent. No newer engine
  candidate was found in the inspected source. Existing R49, WMMA32, MTP and other
  prepared paths remain recoverable in both B0 and B1; qualification is unchanged.
- Lightweight process lists showed no active compiler/inference process. Detailed
  Win32 process command-line ownership was unavailable (`Get-CimInstance`: access
  denied). Only the newly created cleanup tree was edited.

Use this owned tree for the next user-started Sol session. Do not restart from
publication main and lose B1. Alternatively, create a new worktree at the exact
B0 commit, apply the four patches below in order, and compare against the saved
source manifest (`normalized_sha256` permits Git's CRLF checkout conversion;
`sha256` records the exact current bytes). There is no cleanup commit to cherry-pick.

## Implemented batches and audit disposition

| Batch | Decision and removed complexity | Authoritative behavior/callers |
| --- | --- | --- |
| 01 | MERGE 79 strict boolean parsers: 32 full-attention, 25 linear-workspace, 22 Engine setup sites | `environment_options.h`; original names, absent/0/1 semantics, exception messages, read ordering and dependency predicates |
| 02 | MERGE nine environment-restoration classes into one test-only owner at 13 call sites | Existing draft/H6/K6/K7 qualification fixtures; identical ordered key sets and scope lifetimes; shared M1 list retained once for five callers |
| 03 | SIMPLIFY focused host checking into one closed two-test runner; add parser/restoration regression assertions and a verified process-tree launcher | Existing host-contract CMake separation retained; direct MSVC runner bypasses full CUDA configuration and repeated campaign build setup |
| 04 | Update active source/build map and this continuation authority | `docs/architecture.md`, `docs/build-and-run.md` |

No numerical test, oracle, failure input, threshold, dispatch counter, supported
target, option name or runner selector was removed. The actual deletions are the
repeated parser bodies and bespoke restoration classes. No device kernel,
arithmetic, dispatch predicate, synchronization, ownership mechanism or lifecycle
timing boundary was changed. The test helper changes only host fixture setup
storage; it does not enter the grouped research workload timer.

KEEP: native `.ninfer` targets/protocols; qualified small gains such as K6 stream
reduction, represented device prefix and K6 N16; device working state/greedy,
committed-tap D2D/direct staging and resident draft ring; independent oracles and
state/rollback/tap/ring/liveness gates. Host lifetime descriptors were inspected
as correctness boundaries, not assumed redundant.

DEFER: removal of rejected R612 segmented-prefix code (intertwined storage and
attention lifetimes), R49 device implementations, kernel relocation, GDN/full-layer
projection consolidation, text-model ownership refactors and graph restructuring.
Those require device/build/performance evidence. R610/R611 do not justify a graph
rewrite. R612 remains rejected/off; WMMA32 and native MTP remain unqualified/off.
Dynamic split controls and permissive/default-on/presence-only controls were left
alone: caching or tightening them would alter their current interfaces.

## Options and evidence

The exact changed option names, destinations, dependencies and original diagnostics
are in `D:\AI\voidinfer-cleanup-control-20260922\option-map.json` (79 entries).
All migrated options default to false before their existing owner/dependency gate.
Other defaults are unchanged. No option was promoted or assigned new qualification.
Engine shared preparation/projection/page/upload controls retain their documented
unqualified boundaries; profile-specific attention/linear experiments retain their
original receipt requirements. Options without reviewed runtime evidence remain
unqualified by this cleanup, rather than being inferred dead or safe.

Current evidence authority remains `docs/current-status.md`, `docs/benchmarking.md`,
`docs/exl3-serving.md` and the preserved R612/R608 handovers under
`D:\AI\voidinfer-engine-compare-20260918\campaigns\SOL_HIGH_AGGRESSIVE_FAST_20260921`.
R608 means native same-weight ordinary-FP16-device-KV DFlash2 physical C1, three
4096-prefix/128-output requests per arm, including lifecycle teardown in grouped
wall time. It is not target-only or HTTP/client throughput. Keep
`VOIDINFER_PARITY_DFLASH_TIMING=0`. Historical receipts do not qualify B1.

## CPU validation and limits

- `Test-CleanupJob.ps1`: parent/child/grandchild are members of the exact named
  Job Object, affinity `0xFF`, BelowNormal; attempted widening leaves affinity
  unchanged. An initial probe incorrectly assumed the setter must return failure;
  Windows returned success with the effective mask still `0xFF`. The probe was
  corrected to inspect the effective mask and all three levels passed.
- `Run-Exl3HostContracts.ps1`: environment-options/restoration and existing candidate
  contracts compiled with MSVC 14.44 and passed. Single compiler, `/MP1`,
  `/cgthreads1`, no CUDA includes/libraries/configuration/probes. DLL import review
  showed only Windows/CRT dependencies.
- One-time baseline-versus-cleanup comparison compiled the actual old/new host
  setup fragments: **79 sites, 1,048 comparisons passed**, covering absent, 0, 1,
  malformed values and every boolean dependency combination present at those sites.
  Empty-string rejection is also checked directly by the permanent parser test.
- Permanent restoration checks cover nested scopes, exception unwinding, existing
  values and removal of initially absent variables; parser checks preserve exact
  diagnostics and validation before a disabled owner gate.
- Three PowerShell files parsed successfully. Scoped diff whitespace and patch
  checks passed. All 23 changed/new paths replayed forward to B1 and backward to
  B0 in a disposable source-only fixture, comparing normalized line endings.
  Qualification key lists/call sites were checked against B0.
- Source reads and edits were lightweight. All compiler/test work ran inside the
  common eight-logical-processor mask, BelowNormal job, with sequential compute
  commands and single-threaded nested workers. No GPU query, context, model load,
  server, CUDA test or benchmark was run. No owned watcher/service remains.

Host evidence is limited to configuration and test setup. Full translation-unit
compilation and runtime behavior remain pending; no zero-regression or speed claim
is made. Do not run the full CTest suite during the no-GPU phase.

## Recovery and inventory

Recovery directory: `D:\AI\voidinfer-cleanup-control-20260922`.

- `01-engine-options.patch`: three EXL3 implementation files and shared parser.
- `02-test-scopes.patch`: nine fixture headers, harness include and shared owner.
- `03-host-checks.patch`: host test, CMake registration, launcher/probe/runner.
- `04-docs-handover.patch`: active docs and this handover.
- `source-inventory.json`: B1 changed-path SHA-256/size inventory.
- `patch-verification.txt`: successful forward/reverse replay; the initial raw-byte
  reverse comparison detected Git line-ending conversion and was replaced with
  the explicit normalized-content comparison, not a semantic relaxation.
- `option-map.json`, `scope-map.json`: bounded change maps.
- `research-status.txt`, `publication-tests.patch`: pre-cleanup source comparison.
- `containment-result.txt`, `option-migration-result.txt` and the one-time
  `check_option_migration.py`/`option_migration.cpp`: host comparison evidence.

Patches include new files; the Git index remains unstaged. Inspect `git status
--short` for the exact uncommitted inventory. To undo a batch, first check its
reverse patch with `git apply --reverse --check <patch>`; apply only if no newer
changes conflict. Batch 03's tests consume 01 and 02, so reverse 03 before removing
either shared helper. Then reverse only the unwanted production/test batch and
update the documentation. Never reset/clean the tree or undo another session's work.

## Deferred validation after explicit GPU release

1. Preserve B1, then create a fresh B0 worktree at the pinned commit if needed.
   Audit the new build path before running it. Use explicit `120a`, one build,
   normally four jobs, nested compiler/library workers one, and the same eight-CPU
   Job Object. Large CUDA units may still need reduced concurrency.
2. Acquire the existing `Local\NInferEngineGpuLease` through the authorized owned
   runner; do not steal ownership. Build both states with identical toolchain,
   workload, CPU settings and effective options. The original frozen binaries
   and receipts stay untouched.
3. Build `ninfer_exl3_dflash2_accept_test` and affected Engine/lifecycle targets.
   Repeat the retained draft/H6/K6/K7 qualification selectors touched by batch 02;
   retain independent expected inputs, dispatch liveness and all exactness gates.
4. Use the preserved runner reference
   `D:\AI\voidinfer-engine-compare-20260918\campaigns\MIA_PARITY_LUNA_FAST_20260918\native-r608-dflash-untimed-4096x128-r1\Run-Native-DFlash-CurrentFastStack-Common-r1.ps1`.
   Prepare a new scoped runner bound to B0/B1 source and fresh binaries/receipts;
   do not execute its frozen hardcoded old binary as a B1 test. Preserve its
   three-request code/prose cold/reused 4096/128 workload, final teardown and
   useful-output accounting, with timing instrumentation off.
5. Compare B0/B1 tokens, complete state, taps/ring, rejection/rollback/settlement,
   graph/lifetime gates and representative repeated counterbalanced wall times.
   Exercise Engine C1/C2 accepted/rejected configurations and applicable HTTP
   cancellation/schema lifecycle checks from `docs/exl3-serving.md` separately.
   B1 must pass before attributing any B2 candidate improvement.

Next safe step while GPU access is forbidden: review these patches or prepare a
separately recoverable candidate on preserved B1. Do not launch the next campaign
automatically. Current task ends at this handoff.
