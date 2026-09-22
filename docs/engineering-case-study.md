# Engineering case study

## Context

VoidInfer explores how far a specialized native Windows C++/CUDA runtime can push Qwen3.8-27B on
Blackwell while preserving exact target authority. The hard part is not just writing a fast kernel:
model loading, KV and recurrent state, speculative rollback, GPU memory, asynchronous lifetime,
measurement, and reproducibility all have to agree.

## Story 1: keep draft working state on the device

**Problem.** The same-weight DFlash2 route paid avoidable host/device and staging overhead around
selection, committed taps, and the draft ring.

**Investigation.** Dispatch counters and exact state traces separated actual execution from flags.
Earlier attempts showed that reducing one transfer or matching tokens alone did not establish a safe
route. The alternatives were isolated micro-kernels, deeper host caching, or a composed device-state
change.

**Implementation.** R599 composed device greedy selection, committed-tap D2D movement, direct tap
staging, and a resident draft ring. Transaction boundaries and the exact target remained unchanged.

**Validation.** An eight-output liveness gate preceded the full R600 matched workload. R600 used
three cold and three reused requests for each frozen code/prose fixture and passed all 2,822 checks,
including token, target-state, tap, and reconstructed-ring equality.

**Result.** Versus R597, throughput improved in every matched arm: code cold 42.24→45.05 (+6.65%),
code reused 44.67→47.60 (+6.57%), prose cold 24.23→25.65 (+5.88%), and prose reused 25.03→26.59
(+6.22%). The gain is real but well short of the larger program targets.

**Tradeoff.** More device-resident state improves locality but tightens lifetime and rollback
requirements. The route remained tied to its exact configuration rather than becoming a universal
default claim.

## Story 2: repair the benchmark before optimizing the wrong signal

**Problem.** Phase timing suggested large proposal/verifier costs, but `NINFER_DFLASH2_TIMING`
inserted synchronization after proposal phases.

**Investigation.** R604 used NVTX/Nsight to locate the verifier as the broad bottleneck. It also
showed profiler-heavy `cudaEventSynchronize` time, making its API percentages unsuitable as normal
production shares. Source inspection then identified the timing fences themselves.

**Implementation.** The runner gained an explicit `VOIDINFER_PARITY_DFLASH_TIMING=0|1` contract and
recorded the effective state. R607 tested the instrumentation-disabled route; R608 repeated the full
4,096/128 workload without attribution fences.

**Validation/result.** R608 again passed 2,822 checks and 12 exact rows. Compared with timed R600,
throughput moved only +0.22%, +0.48%, -0.19%, and +0.08% across the four arms. The fences distorted
phase attribution but were not the missing end-to-end speedup.

**Tradeoff.** Attribution remains available, but it is never mixed with the production-style
baseline. This prevented a compelling profiler screenshot from becoming a false performance claim.

## Story 3: reject an exact route that is slower

**Problem.** A segmented resident-prefix design aimed to reduce verifier data movement and launch
structure.

**Investigation and alternative.** R610 first measured graph submission at only 1.28–1.99% of
decode. R611 disabled full-layer graphs and saw mixed changes from -0.43% to +1.17%, consistent with
noise. That narrowed the next bounded test to the segmented prefix rather than a broad graph rewrite.

**Implementation/validation.** R612 enabled the route only behind an explicit flag, disabled
full-layer graphs for isolation, and ran the 12-row eight-output exact screen. All scoped exactness
checks passed.

**Result.** Median decode regressed 17.08–21.15%, while initial H2D remained about 269 MB/request.
The route was rejected and left default-off.

**Tradeoff.** Correctness is necessary but not sufficient for a performance change. Keeping the
negative result prevents repeated work and demonstrates that defaults follow evidence rather than
implementation effort.

## Development and provenance

The project combines upstream NInfer foundations, adapted third-party EXL3 utilities, original
Windows/CUDA integration, and AI-assisted implementation and test iteration. The defensible role is
maintaining the system contract: selecting and integrating changes, preserving provenance, designing
gates, interpreting evidence, and rejecting unsupported claims—not claiming sole authorship of every
algorithm or line.
