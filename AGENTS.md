# AGENTS.md

These rules apply to the whole repository.

## Governing objective

Complete the user's explicit deliverable within the applicable product contract. For the declared
product target and requested outcome, choose the technically strongest coherent solution. Optimize
for architectural integrity, clear ownership, functional and numerical correctness, and maximum
relevant performance. Never optimize a solution for a small diff, few changed files, low
implementation effort, short-term simplicity, backward compatibility, or preservation of a
superseded internal path. Make every affected implementation, test, tool, and active authority
consistent with the selected design.

Correctness, performance, tests, profiling, documentation, provenance, cleanup, and tooling are
means to the requested outcome, not independent objectives. Do not let supporting work replace,
delay, or materially enlarge the requested deliverable.

## Responding to user corrections

When the user points out an error in the agent's execution or reasoning, never reply with the
formulaic opening “你说得对，……” ("you're right, ..."). This phrase and cosmetic variants of it
are strictly prohibited in correction responses because they sound reflexive and insincere. Do not
replace it with another generic agreement such as “确实如此”, “完全正确”, or “好问题”. Instead,
state the specific mistake directly, explain its concrete effect when relevant, and say what has
been or will be changed. Keep the response proportionate; do not add performative apology or praise.

When choosing between possible work, use this order:

1. respect applicable product and external-contract constraints;
2. satisfy the user's explicit deliverable and acceptance criteria;
3. preserve functional and numerical correctness of supported behavior;
4. choose the strongest architecture and clearest ownership for the declared product model;
5. maximize performance at the scope relevant to the task;
6. gather only the evidence and provenance needed to support the result.

The product and architecture described here are the current contract for ordinary work. A task may
explicitly change that contract; when it does, update the affected implementation, tests, and active
authorities consistently rather than treating the current description as an immutable prohibition.

## Scope control

Before substantial work, determine the requested output, the behavior or decision it must support,
and the conditions under which it is complete. This is an execution discipline, not a requirement
to create a separate planning artifact.

Work is in scope only when it:

- directly contributes to the requested deliverable;
- is necessary to preserve an applicable product, semantic, or external contract;
- resolves uncertainty that could materially change the result; or
- checks a realistic regression introduced by the change.

An architectural redesign, cross-cutting refactor, or replacement of an existing path is in scope
when it is necessary to deliver the strongest solution for the requested outcome. Do not use scope
control as a reason to ship an inferior patch. Do not expand into unrelated audits, cleanup,
hardening, compatibility work, benchmark campaigns, or documentation projects. General engineering
preferences, possible future scenarios, and concerns outside the declared product model do not
create requirements by themselves.

Handle incidental findings proportionally:

- address them when they block the requested outcome or make it materially incorrect;
- include them when they are inseparable from a coherent implementation;
- otherwise leave them unchanged and mention them only when they are useful to the user.

For analysis, review, or design work, the requested explanation or design artifact is the
deliverable; experiments and code inspection serve only to resolve material questions. For
implementation work, implement the selected design completely across its affected boundaries,
remove the superseded project-owned path, and validate its supported observable behavior. For
diagnosis, establish the cause and supporting evidence without turning the task into an unrequested
fix or redesign.

## Evidence, provenance, and completion

Select evidence from the claim or decision it supports. The availability of a tool, test suite,
artifact, or profiler does not make its use necessary. Prefer representative evidence over
exhaustive evidence, and do not repeat an experiment unless the previous result is invalid or
inconclusive, or the new result could change a live decision.

Verification must match the semantic contract: use exact comparison for exact formats and
transformations, and numerical or behavioral criteria for floating-point and probabilistic work.
Do not substitute final-output plausibility for verification of an operator or state transition.

Record only the provenance needed to interpret a material result. By default, this is the relevant
target, hardware/toolchain, workload or command, and summarized outcome. Fixed hashes, clean
worktrees, full command transcripts, raw profiler inventories, byte-identical regeneration, and
exact probabilistic outputs are not validity requirements unless a concrete contract or the user
requires them.

Stop when:

- the requested deliverable exists;
- applicable contracts are satisfied;
- material claims have sufficient evidence;
- relevant checks pass, or their limitations are stated clearly; and
- no known in-scope issue prevents the result from being used.

Do not continue merely to eliminate all uncertainty, collect more metrics, complete a process loop,
improve descriptive provenance, investigate unrelated observations, or make working notes
exhaustive. The final result should lead with the deliverable, key decisions, relevant verification,
and material limitations. Raw logs, experiment diaries, exhaustive command histories, hashes, and
intermediate artifacts are excluded unless requested or themselves the deliverable.

## Current product contract

VoidInfer is a native Windows C++20/CUDA inference engine for maximum single-GPU quality and
long-context capacity, then speed, on one model: Qwen3.8-27B as an EXL3 6 bpw target with its
DFlash2 draft (K=7) and, optionally, the BF16 Vision tower. It is compiled for `sm_120a` and tuned
and measured on an NVIDIA GeForce RTX 5090.

The Engine is the EXL3 coherent-device lane engine (`src/exl3/exl3_engine_core.*`): one resident
model, a startup-fixed one to eight lanes batched into one DFlash2 round, FIFO admission with
VRAM-aware lane admission and no preemption. Attention history is L0 OSCAR (INT4 codes, exact sink,
recent window and hot rows on the device; exact FP16 planes in host memory); every published token
is verified by exact VeriCache. FP16 device KV for context and non-exact VeriCache are diagnostics,
not product routes. The Engine applies its benchmarked kernel/route profile itself
(`src/exl3/engine_profile.cpp`); `VOIDINFER_*` variables only override it.

CLI, OpenAI/Anthropic serving and measurement all go through the public Engine. Additional models,
quantization formats, other GPUs, large-scale or preemptive batching and priority scheduling are
outside the current product. This is a local, single-owner project; the model directories and the
local workflow are trusted. Requirements derived from a different workload, trust model, or
deployment model are out of scope until that product contract is explicitly changed.

## Engineering priorities

Prioritize functional correctness, architectural quality, clear ownership, direct code, and maximum
requested inference performance. Change size, implementation difficulty, short-term simplicity,
and backward compatibility for project-owned contracts are not quality criteria. Low maintenance
cost may distinguish otherwise equivalent designs, but it never justifies worse architecture or
performance. Generality, defensive hardening, formal completeness, broad compatibility, and test
coverage are not goals by themselves.

Prefer explicit target-specific implementation over framework-like abstraction. Do not add generic
model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, runtime weight repacking, or placeholders for hypothetical models or hardware unless an
explicitly changed product contract requires them.

## Sources of truth

Read only current authorities relevant to a live decision in the task. The following list is a
routing map, not a mandatory reading list:

- `README.md` and executable `--help`: delivered capabilities and exact commands;
- `docs/README.md`: documentation map;
- `docs/cli.md` and `docs/serving.md`: CLI and OpenAI/Anthropic HTTP behavior;
- `docs/architecture.md`: Engine, lane, KV-tier and VeriCache structure and the source map;
- `docs/current-status.md`: supported defaults, measured state and rejected routes;
- `docs/OSCAR_KV_FIDELITY.md`: KV-tier quality instruments and results;
- `bench/engine-compare/README.md`: the benchmark harness and comparison protocol;
- `include/voidinfer/engine.h` and `include/voidinfer/types.h`: in-tree C++ product interface.

Do not survey unrelated references for completeness.

## Product and ownership boundaries

These boundaries govern ordinary implementation work. An explicit architecture task may revise
them, but must update the corresponding active authorities and affected implementation together.

- `include/voidinfer/engine.h` and `include/voidinfer/types.h` are the opaque Engine interface used
  by in-tree applications. VoidInfer does not install or export a C++ SDK.
  `include/voidinfer/ops/` holds repository-internal Op contracts.
- `src/exl3` owns the model: EXL3 weight loading and linears, the text model (attention, GDN, MLP,
  L0 OSCAR), the DFlash2 draft, VeriCache, lanes, plane sets and the Engine core.
- `src/core` owns device primitives, tensors/views, arenas, graph RAII and raw transfer mechanisms.
- `src/ops` owns the shared semantic Ops the engine still launches (Vision tower ops, gated
  RMSNorm, OSCAR INT2 history). Engine-specific kernels live in `src/exl3`.
- `src/targets/qwen3_6` owns the Qwen family frontend: tokenizer/template and output semantics,
  media preprocessing and M-RoPE prompt construction, prepared prompts and output sessions.
- `src/runtime` owns common request contracts, sampling resolution and the public Engine PIMPL.
- `src/media/decode` consumes already-owned bytes. URL/path/data acquisition belongs to
  `src/product/media_acquire`, the CLI or serving.
- `src/serve` owns protocol translation and transport. CLI, server and benchmarks call only the
  public Engine for inference; research tools under `tests/research_exl3_*` may drive the text
  model directly.

## Compatibility and document lifecycle

Project-owned C++ APIs, CLIs, Python tools, fixtures, reports, formats, and active documentation do
not preserve backward compatibility. When a task replaces project-owned behavior, remove the
obsolete aliases, fallbacks, transition branches, and tests in the affected contract instead of
maintaining two paths. Do not turn that rule into unrelated repository-wide cleanup.

The advertised OpenAI and Anthropic protocol surfaces are real external contracts. A change to
their behavior must update the affected schema tests and serving documentation together.

Integrate stable requirements into the existing active reference. Use a temporary dated plan only
when active work genuinely needs one; a plan is not a substitute for the requested deliverable.
Remove completed or abandoned plans instead of retaining a historical documentation tree. Do not
create parallel `final`, `v2`, or `new-design` references.

## Numerical correctness

When a task changes numerical behavior or makes a numerical claim, identify the mathematical
oracle, represented public inputs, explicit semantic cast/quantization/state boundaries, output
criterion, and real model shapes relevant to that claim. If a route's private precision or
reduction profile matters to the evidence, describe it as an implementation profile rather than a
semantic requirement. Apply exact, tolerance-based, or behavioral comparison according to the
actual semantic contract.

Every floating-point Op has one independent naive FP32/FP64 mathematical oracle; exact transforms
and codecs have one independent exact oracle. The oracle evaluates the complete logical formula
from the represented public inputs and, for packed weights, decodes the signed code with the exact
stored scale. It does not copy a production kernel's staging casts, reduction tree, workspace dtype,
or another implementation's output.

The oracle does not prescribe a production arithmetic path. Unless an intermediate value is an
observable Op output, explicit Cast/quantize/dequantize result, registered codec value, or specified
persistent state, kernels may choose the natural intermediate precision, instruction operands,
reduction association, workspace representation, and kernel decomposition for their route. A fused
kernel is neither required to reproduce an unfused BF16 materialization nor forbidden from using a
lower-precision intermediate when that is the natural qualified implementation. Every production
route is checked directly against the same oracle with a criterion appropriate to its output and
implementation profile; pairwise implementation parity is supplementary evidence only.

Where relevant to the changed behavior, account for EXL3 trellis decode, FP16/BF16 fusion order,
FP32 GDN state, L0 OSCAR codes and exact history, DFlash2 accept/commit and rollback state, arena
lifetime, and CUDA Graph address stability.
This is a risk map, not a checklist for every numerical task.

## Performance work

Define a performance claim at the level where it matters: operator, schedule, request phase, or
end-to-end inference. Measure that level directly when practical. An isolated microbenchmark can
support an operator-level claim but does not establish an end-to-end improvement.

Use whole-inference profiling when end-to-end attribution remains unresolved. Use kernel profiling
only after a relevant kernel has been identified and a kernel-level answer could materially change
the current design or implementation decision. Do not collect additional profiling data once the
relevant alternatives can be distinguished and the requested claim has adequate support.

Retain concise context sufficient to interpret a reported result: relevant hardware/toolchain,
model identity at the descriptive level, workload or command, and summarized measurements. Raw
reports and fixed repository or model hashes are not required by default.

## Tests and verification

Add or retain a test only when it protects supported observable behavior or a realistic regression:
numerical kernel/model correctness, EXL3 weight loading, external schema/report behavior, a small
real integration route, GPU lifetime, or a reproduced bug. Do not add tests for coverage,
private file/class shape, getters/constructors, deleted compatibility, source-string scans,
hypothetical failures, or test ceremony.

Run a focused set of checks sufficient to support the changed behavior and its material claims.
The following are typical choices, not a cumulative checklist:

| Change | Relevant evidence |
|---|---|
| documentation | affected active-link/stale-reference review and `git diff --check` |
| C++ runtime/API | affected explicit targets and meaningful tests |
| Python tooling | `py_compile` and affected Python tests |
| EXL3 loading / model numerics | affected contract tests and the real model when semantics require it |
| CUDA math | independent numerical oracle at relevant shapes |
| memory/lifetime | the affected execution; sanitizer only for a concrete lifetime risk |
| performance | measurement at the claimed scope; attribution tools only when needed |
| serving | affected OpenAI/Anthropic schema tests and observable request/stream behavior |

Do not replace weak verification with low-value tests. State clearly when a relevant check could not
run and why.

## Local environment

Use unrestricted build-tool parallelism for repository compilation. Invoke CMake builds as
`cmake --build <build-dir> -j`; do not supply a numeric job limit such as `-j2` or `-j32`.

These are conventional project resources, not a checklist of resources every task must use:

| Purpose | Path |
|---|---|
| repository | current checkout |
| EXL3 target | `C:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6` (OSCAR rotations in `l0_oscar/`) |
| DFlash2 draft | `C:\AI\models\Qwen3.8-27B-DFlash2-EXL3-5bpw` |
| BF16 checkpoint / Vision tower | `D:\AI\models\Qwen3.8-27B-bf16` |
| normal build | `build/` |
| benchmark harness | `bench/engine-compare/` (`ours_env.ps1` sets the model package) |
| profiler output | `profiles/ncu/`, `profiles/nsys/` |
| hardware/toolchain | RTX 5090, `sm_120a`, CUDA 13.3, MSVC 14.51 (VS 2026 Build Tools) |

Do not install or upgrade dependencies unless the task requires it. Never select a model directory
by glob, modification time, or an unqualified "latest" name. Model weights and profiler outputs are
local prerequisites; do not download or regenerate them unless that work is in scope.

## Commits

Create a commit only when the user requests one. Use Conventional Commit-style subjects, for
example:

```text
perf(exl3): fuse the verifier history pass into the exact line
```

Use concise lowercase types consistent with repository history (`feat`, `fix`, `perf`, `bench`,
`test`, `build`, `refactor`, `docs`, `chore`).
