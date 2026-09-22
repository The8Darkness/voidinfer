# Explicit EXL3 text serving

VoidInfer's existing `ninfer-serve`, `GenerationService` and public `Engine` now accept a typed EXL3 target/draft package. The qualified slice is greedy text with one persistent physical execution lane by default, or two with explicit `--max-concurrency 2`, actual five-layer DFlash2 B8 proposals, ordinary FP16 host-backed L2 KV and FP32 recurrent state. The coordinator authorizes and establishes resident/VirtualLocked ownership before any generated window becomes visible.

This is an explicit route. The existing T23 default and separately retained FD2/native16 options have not been promoted or composed into this serving profile. V6 is still required in the pinned target package; media requests fail explicitly while the encoder's full numerical and downstream attachment gates remain open. Positive-temperature sampling is also explicitly unsupported on this route.

## Package and launch contract

The qualified receipts used pinned target and draft manifests. For portable use, set
`VOIDINFER_TARGET_MODEL` and `VOIDINFER_DRAFT_MODEL` to matching local directories and record their
manifest revisions in every result.

Use `results/autonomous/Assert-P9-Models.ps1` to validate the existing artifact manifests. The loader uses the pinned tokenizer, added tokens, ordinary `chat_template.jinja`, generation and preprocessing metadata. It does not choose `chat_template.jinja.fixed`. EXL3 and `.ninfer` packages cannot be mixed.

The verified loopback arguments are:

```powershell
ninfer-serve.exe `
  --exl3-target $env:VOIDINFER_TARGET_MODEL `
  --exl3-draft $env:VOIDINFER_DRAFT_MODEL `
  --host 127.0.0.1 --port 18129 --max-context 4352 `
  --no-thinking --temperature 0 --default-max-tokens 32 `
  --max-shared-prefixes 4 --log-stats-interval-ms 0
```

For this checkout, execute GPU/build work through `results/autonomous/Run-Owned.ps1` and its `Local\NInferEngineGpuLease`. Do not start this command beside another owned engine or experiment. Use a binary bound to a successful source archive/build receipt. The campaign's `Build-Lifecycle.ps1` builds and freezes the existing server, frontend comparison and Engine lifecycle executables; `Run-Engine-Current.ps1` is a bounded loopback example that validates the binding, loads the following qualified configuration, tests requests and stops its own server.

Load the environment from `DECODE_ROT_QUALIFIED_CONFIGURATION.json` followed by `SOL_HIGH_DRAFT_K5_ASYNC_A_QUALIFIED_CONFIGURATION.json`'s overlay. The retained text profile additionally specifies:

```text
NINFER_EXL3_EXACT_HOST_KV=1
NINFER_EXL3_OSCAR_L0_ONLY=0
NINFER_EXL3_EXACT_ATTENTION_PARALLEL=1
NINFER_EXL3_NATIVE_CONTINUATION16=0
NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7=0
NINFER_EXL3_PINNED_RECURRENT_EXPORT=1
```

The recurrent slab option remains default-off. Its exact pageable fallback is visible in counters. Three144MiB slots per context, a1GiB process cap including cached/reload survivors, and an8GiB physical host reserve are enforced; CUDA registration does not replace publication's VirtualLock requirement.

### Unqualified registered HostKV upload boundary

`NINFER_EXL3_ENGINE_REGISTERED_KV_UPLOAD=1` is a default-off research route,
not part of the qualified serving profile. It applies only to immutable complete
64-row ordinary-FP16 KV page backings. A short represented view may use a
registration for its complete backing, but a partial page, resized backing,
foreign owner, overlapping incompatible registration, exhausted reservation or
busy cache remains on the ordinary pinned gather-and-copy path.

Registration is a CUDA transfer authority only. Published host pages still need
their independent resident/VirtualLock ownership. Successful `cudaHostRegister`
is platform- and driver-dependent and must be established in the future target
environment; a registration acquisition refusal is an exact fallback, not
qualification. Copy and event completion must also be established on the actual
device route. An uncertain copy/record/wait or failed unregister conservatively
retains its source, destination and accounting credit and refuses reusable Engine
retirement/reload rather than silently freeing them.

Future qualification must use a newly built source-bound lifecycle executable
in isolated fresh processes before the full C1/C2 route. It must compare ordinary
pinned and registered uploads for tokens, complete target state and private draft
conditioning; exercise both transfer slots and physical lanes; cover cancellation,
turnover and staged fallback; and close with zero readers/pending transfers and
conserved registered-host/metadata credits. Prepared command selectors are kept
in the implementation campaign's `PREPARED_TESTS.md`. Until executed under
separate authorization, this route is NOT_BUILT, PREPARED_NOT_RUN and
qualification-pending.

### Unqualified shared represented-page boundary

`NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGES=1` is a separate default-off research
route. It admits only complete published 64-row ordinary-text FP16 pages whose
model owner, page revision, range and rotary-position contract match. The coarse
shortlist is never reuse authority. Media, partial/mutable tails, foreign roots,
changed backing and incompatible whole-prefix or attention-staging routes fall
back or refuse according to their existing exact contracts.

One cache entry owns one physical 4 MiB all-bank page fill. A view becomes ready
only after all 32 planes and the readiness event complete. Each D2D or direct
attention consumer then owns a separate acquisition/execution/event-bound reader
until its actual final use; lookup liveness alone cannot protect storage. Cache
pressure retires lookup first and reclaims storage only after fills/readers drain.
When no bounded victim is reclaimable, the request uses ordinary HostKV rather
than waiting or allocating beyond the coordinator budget. Failed or uncertain
work remains charged and blocks reusable retirement/reload.

The copy adapter reuses a ready page through the context's private contiguous
attention destination. With `NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGE_ATTENTION=1`,
supported exact segmented attention instead receives immutable K/V ranges
directly while mutable rows remain private. Neither mode removes the authoritative
host backing, and descriptor accounting is not physical free-memory telemetry.
Future qualification must separately cover C1/C2 copy and direct-attention modes,
mixed arrival, cancellation, typed collision, multi-page/noninitial history,
failure/quarantine, resource conservation and exact token/state/private-draft
parity. No throughput estimate is implied by this prepared source.

### Unqualified segmented-attention research boundary

The exact segmented reader is limited to ordinary FP16 GQA24:4, head dimension
256, B1..B16 and a validated chronological union of local history, staged prefix
and immutable shared-page ranges. Numerical coordinates do not establish
ownership: the context retains local allocations, staging/page owners and every
producer/consumer final-use token separately. Missing, overlapping, reordered,
stale-position, MRoPE or unsupported-profile inputs refuse before the candidate
kernel; unselected profiles keep the inherited attention path.

Exact opt-in variants share only represented Q or K/V loads while preserving
each head/query row's canonical score, softmax and value accumulation order.
Each native row has its own causal endpoint, including query-pair execution. The
next-bank staging route reserves both planes and separate producer/consumer
events before use. Input-normalization/MLP-input scratch aliasing is allowed only
for proven nonoverlap on the same ordinary eager stream and is refused for trace,
capture or pending consumers.

`NINFER_EXL3_NUMERIC_ATTENTION_SPLITK=1` remains the sole actual numerically
classified online/split-K research route. It owns distinct workspace, is
default-off, excludes exact reconstruction/native-continuation composition and
does not inherit exact status from represented inputs. Existing failed or
unexecuted numerical gates remain authoritative. Native-64K, staged, shared-page,
query-pair and coalesced cases in the implementation campaign are prepared source
only; none expands the qualified serving profile.

### Unqualified bounded reconstruction/prefill boundary

Exact K7 reconstruction and the added compatible K6 Down family are explicit
wide-prefill routes over one context-owned decoded-weight slice, never a decoded
whole model. Supported slice widths are 256/512/1024/5120 columns. The binding
includes the strong model owner, decoded address/extent, K family, ordered stream
and reserved scratch; incompatible or unavailable reservations use the original
direct projection without a timing-derived strategy guess. Failed or cross-stream
use poisons/refuses reuse and retirement follows the owning device after drain.

Canonical paired input transforms, gate/up activation colocation and residual/
RMSNorm colocation remain default-off and preserve their material FP16 rounding
boundaries. Packed native-fragment vector loads require their represented
alignment and use scalar handling at misaligned/tail boundaries. Prefill chunk
and row-tail selection is limited to source-declared partitions; changed complete
partitions remain numerically unqualified even when final tokens happen to match.
Head omission preserves every verification, seed and final-root logit consumer.

The separately retained alternate reconstruction GEMM/FD2 routes remain numeric
or optional and do not qualify these exact paths. Prepared held-activation,
intermediate, full-state, graph-invalidation, tail-poison and real continuation
cases are NOT_BUILT and PREPARED_NOT_RUN.

### Unqualified speculation-policy boundary

The supplied-cost outer policy is request-local, default-off and restricted to
2/4/8 useful verifier rows. Observations keep fixed neural B8 generation,
verification, repair, publication and complete-window costs separate; unavailable
or overlapping stages remain unknown. Acceptance accounting distinguishes seed,
neural and suffix sources and censors unproposed or terminal-truncated rows. Policy
updates occur only after committed publication, carry request/acquisition/frontier/
revision provenance and use bounded storage plus a publication hold interval.

Suffix hits save neural work only after L2 commits the represented prefix. Near
output/context/stop limits, seed/suffix routes may avoid work, but an executed
neural proposal is still charged as fixed B8 with returned and discarded rows
reported separately. A conditional second B8 is legal only after the first B8 is
fully accepted and its new target/ring conditioning is retained; mismatch, stop,
control, stale scope or failure invalidates every dependent second-block result.

That conditional source uses two dependency-ordered native8 verifier calls. It is
not native W16 and is never labeled as such. Native16 remains an exclusive
separate authority until a genuine temporal W16 plan with legal intervening
conditioning exists. Prepared comparisons separate inner draft cost, outer
horizon, suffix/neural choice, native width and C1/C2 physical batching; they
contain no inferred acceptance, latency or throughput improvement.

### Unqualified exact repair-checkpoint boundary

`NINFER_EXL3_REPAIR_CHECKPOINT=1` selects a default-off ordinary-FP16 B2..B8
repair route in the real DFlash L2 caller. Engine startup reserves one complete
checkpoint arena per physical target context through the serving coordinator;
the arena owns every GDN recurrent and convolution plane, canonical OSCAR cache,
position, logits, five target taps and embedding trace. The source supports one
root boundary per attempted verifier window, not an unbounded chain of dense
checkpoints. Disabled requests retain the original root-restore/replay route.
If the coordinator refuses the complete arena reservation before allocation,
that physical lane also retains replay and reports a reservation fallback; it
does not attempt unreserved checkpoint growth.

After target authority finds a mismatch, only teacher-forced rows whose proposal
token was actually accepted may be reconstructed from the immediate continuation
scratch. The mismatched row is never retained. The authoritative correction is
then executed as a separate scalar target row. A first-row mismatch has no valid
nonempty retained boundary and falls back to the saved root plus scalar correction.
A terminal-truncated but otherwise accepted prefix may retain only through the
terminal row. Taps are copied in semantic order before correction overwrites the
attempt buffer, and the draft ring is rebuilt from those committed taps under its
existing root/revision/final-use rules.

Per-request counters separate checkpoint bytes captured, checkpoint restores,
reconstructed target rows and first-row fallback rows from verifier invocations
and replay rows. They are work/accounting observations, not latency savings.
Prepared B8 mismatch/stop matrices compare committed tokens, full exact state,
taps and draft rings against scalar authority, including C1/C2 startup ownership.
All checkpoint source remains NOT_BUILT, its tests PREPARED_NOT_RUN, and native16,
sampled decoding and public performance claims remain outside this boundary.

Optional `NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION=1` enables the qualified exact target K6 small-row reduction path for C1/C2. It preserves the original split and output arithmetic, requires no extra device workspace, and remains default-off. Engine lifecycle and HTTP cancellation composition pass in `K6_STREAM_REDUCTION_RESULT.md`; backend4K/128 paired throughput gains are1.32045% C1 and0.80479% C2. They are not client-speed measurements or additive with other cohorts. Graph capture is refused for this option.

Optional `NINFER_EXL3_HOST_KV_DEVICE_PREFIX=1` retains a redundant4096-row FP16 KV prefix per physical context (256MiB) while preserving complete resident/VirtualLocked host publication. The pilot requires context capacity at least4096 and uses the ordinary eager pinned-chunk path. A1GiB process cap and2GiB observed CUDA-free allocation reserve bound the optional caches; missing reserve falls back to exact streaming. It is qualified with the explicit K6/slab profile above in C1/C2 Engine/HTTP. `DEVICE_PREFIX_CACHE_RESULT.md` records4K128 paired throughput gains2.60678% C1/1.04990% C2 and a small stopped16K C1 check, with memory and first-release tradeoffs. Default remains off. These are backend workload results, not added percentages or HTTP throughput guarantees.

## Request behavior

The existing `/v1/chat/completions` endpoint supports aggregate and streamed greedy output, tools in the chat history, exact repeated prefixes, branched tool turns, stops, reasoning control, limits and cancellation. A complete reusable hybrid root includes recurrence, represented KV, positions and projected draft conditioning. Input token identity remains exact: whitespace and message order are not rewritten to obtain a cache hit.

Every visible token comes from L2 authorization on its actual causal prefix. DFlash2 conditions on the five FP16 L2 post-layer taps at layers5/19/33/47/61, with the pinned BF16 embedding/H6 head. Draft acceptance is not permission to publish. Aborted or stale completions cannot publish; request epochs and fenced release protect reuse.

### Unqualified shared-projection research boundary

The shared target/private-draft projection implementation is default-off and is
not part of the qualified serving profile above. Its current source boundary is
exactly two physical lanes. Private draft work combines two independent fixed B8
linear projections into one physical M16 operator; each lane keeps its own seed,
positions, full-block attention mask, ring/tap history, verification, repair and
publication authority. This is not a temporal W16 proposal horizon.

The implemented private-draft families are Q/K/V/O/down and ordered gate/up for
the pinned five-layer K5 draft. Each family falls back to its ordinary B8 caller
when a compatible peer is unavailable. Unsupported combinations remain explicit:
one or more than two physical lanes, an underfilled draft operator, a single
request's sixteen consecutive tentative positions, different models or head
backing, unequal request controls/prepared-input identities, public media,
positive-temperature sampling and any draft shape outside the pinned K5 extents.
Target projection families have separate width/shape flags and do not become
qualified by composing them with the draft route.

Future qualification must compare fresh-process target-only, draft-only,
composed and both-disabled C2 workloads against the same two serial authorities.
It must establish represented M8/M16 arithmetic and transforms, complete tokens,
exact target state and private rings, repair/terminal/cancellation ownership,
physical and discarded B8 accounting, family fallbacks, retirement/resource
closure and matched complete-workload timing. Until those checks are separately
authorized and executed, all such source and prepared cases are NOT_BUILT,
PREPARED_NOT_RUN and qualification-pending.

### Sampling source and public scope

The EXL3 submit boundary normalizes every resolved sampling request before it
allocates Request/result ownership. Its supported product route is deliberately
narrower than the internal research contracts:

| Route | Actual caller | Public status | Qualification status |
| --- | --- | --- | --- |
| Temperature zero with no occurrence penalty | `Exl3EngineCore::submit` through ordinary DFlash2 greedy verification | Enabled for the pinned text profile | Existing qualification remains profile-specific; current source changes are not rebuilt |
| Positive temperature or an occurrence penalty | The same submit boundary, rejected before Request ownership because current DFlash2 has no complete autoregressive `q` | Disabled with `unsupported_sampling` | Not qualified |
| Independent rejection-with-residual reference | Isolated analytic test source only | Not an Engine/API route | Prepared, not run |
| Complete synthetic `p`/`q` authority | Internal `unqualified_source_api` decision surface only; no execution caller | Cannot report publicly enabled | Source-prepared, not numerically or behaviorally qualified |

The request RNG, distribution authority, workspace and commit/rollback types are
implementation foundations for a future sampled route. They do not change the
public matrix above, and no environment flag promotes them. A future enablement
must provide real predecessor-conditioned normalized draft probabilities and
owned target probabilities, connect the actual model caller, retain exact
publication rollback, and pass distribution, quality, cancellation, resource
and public-client gates. Prepared tests or analytic probability examples are not
evidence that this has happened.

### Unqualified device-seed handoff boundary

The compact device greedy packet is enabled by default for the qualified greedy
text route. Set `NINFER_EXL3_DEVICE_GREEDY=0` to retain the host-logit control
path; explicit `1` is accepted for pinned configurations. This default does not
enable sampling, device-seed handoff, or cross-request packet batching.

`NINFER_EXL3_DEVICE_SEED_HANDOFF=1` is a separate default-off greedy-only route
and requires device greedy to be effective (unset or explicitly `1`). For eligible neural B8 proposals,
the actual DFlash execution passes the target row0 argmax directly from the
compact CUDA result to the pinned draft. A device-ready event is recorded before
the independent host readback, and the draft records its own final-use event
after copying the seed into private block storage. The descriptor retains packet
and model owners and binds acquisition, execution, request generation, position,
serial, target embedding and H6 projection metadata before its one-shot claim.

Scalar output remainders, suffix proposals, the option-disabled route and any
unsupported greedy configuration retain host authority. Positive-temperature
sampling remains refused. Engine dispatch now resolves an explicit selection-
authority requirement before request ownership: compact packets validate only
the greedy requirement, while sampled authority requires an independently owned,
ready full-distribution descriptor with exact row, vocabulary and request scope.
The two authority types cannot satisfy one another, and the descriptor does not
enable sampled production dispatch.
Prepared source covers real first/conditional B8 callers, provenance and target-
projection mismatches before claim, duplicate use, scalar fallback, exact output
and state parity, and abandoned/failed owner retention. The per-call transport
allocation is not reusable storage; bounded reservation/reuse remains T188.
This source is NOT_BUILT, its tests are PREPARED_NOT_RUN, and no performance or
qualification claim follows from removing this host dependency.

### Unqualified greedy-packet transport batching boundary

`NINFER_EXL3_ENGINE_BATCHED_GREEDY_PACKET=1` is a default-off physical-C2
transport option and requires device greedy to be effective. Each target context
still executes its own complete greedy reduction. The Engine-owned rendezvous
only gathers two finished device decision ranges, records final use against each
private producer, and performs one bounded device-to-host transfer. It therefore
does not represent shared target numerical work.

The gather has capacity 16 rows and carries explicit per-offer row counts and
offsets; no fake row enters state, counters, validation or publication. Different
valid extents may pair when their sum fits. An incompatible or over-capacity
offer uses the same exact single-result transport without changing numerical
authority. Result validation and delivery remain scoped to each context's model,
acquisition, execution, generation, position and serial, so an invalid row fails
only its owning result after a successful transport. A transport failure instead
poisons the shared owner and fails both dependencies.

The fixed device buffer, pinned host buffer, stream, event and owner metadata are
reserved before lane construction and included in Engine scratch accounting.
Preclaim cancellation wakes the rendezvous and drains the abandoned private
producer; a claimed producer is always completed. Teardown occurs only after
both Engine workers join and the device drain completes, and uncertain cleanup
is quarantined and blocks later startup. Prepared source covers reversed lane
arrival, independent mixed-result failure, M1+M7 and M8+M3 row maps, total-row
overflow, a third-offer capacity overflow, partial cancellation, public counters
and both concurrent Engine arrival orders. This source is NOT_BUILT, its tests
are PREPARED_NOT_RUN, and it establishes no latency, transfer-saving or
qualification claim.

### Unqualified committed-tap segment transport boundary

`NINFER_EXL3_COMMITTED_TAP_D2D=1` now accepts an owned request scope on the
actual DFlash lane for B1 through B16 verification. Every target segment carries
the strong context and model owners, acquisition/execution scope, source writer
generation, five device planes, exact source/destination rows, original attempt
partition, replay partition and repair/correction classification. A missing
plane, foreign owner, changed scope, invalid range or later target writer refuses
the descriptor before a copy.

The compact verifier consumes each descriptor synchronously on the authoritative
lane stream and assembles it into the existing private 16-row draft staging
buffer. Native16 remains one target segment; native8+native8 remains two target
segments and is not reclassified as an M16 target operator. On mismatch or stop,
only final replay/reconstruction and scalar-correction segments survive in the
published metadata; tentative segment metadata is cleared and replay overwrites
the only rows the draft consumer is allowed to ingest. Callers without a strong
scope retain the host-staging fallback.

The lane's existing completion event/generation is recorded on the final tap set
and checked with Pending ownership before repair or publication. The same-stream
copy, draft ingestion and completion fence serialize slot reuse; extending a
conditional predecessor also extends its tap readiness generation. Mandatory
host tap/root/state artifacts are still produced by this source and are not
claimed as optional diagnostics. Prepared source covers B1/B8 mismatch/stop and
host controls, native8/native16 windows, multi-segment replay, scalar correction,
missing planes, foreign/stale owners and event-generation refusal. The source is
NOT_BUILT, tests are PREPARED_NOT_RUN, and no host-transfer or performance saving
is claimed.

Successful request results expose `committed_tap_device_bytes` from the actual
lane verifier. It counts only bytes submitted by the selected committed-tap D2D
consumer and remains zero on the host fallback. This transport-volume counter is
paired with existing seed-handoff and packet-batch counters plus terminal
state/ring comparisons in the Engine fixture. Decode intervals are printed only
as future matched-run inputs; byte removal is not reported as application
throughput or a qualified speedup.

Continuation-graph reuse is authorized by a fixed-capacity compatibility
fingerprint, not by row count alone. The key retains the context/model and
optional reconstruction owners and compares every directly captured address
and extent, the stable pointer-role record address, B4/B6/B8 shape,
prepared/native capacity, hidden width, OSCAR split
position policy, precision, selected kernel-route bits, capture generation,
origin stream and the bounded projection-option snapshot. Capture, reset reuse,
typed readiness and final replay reconstruct and compare the same key. A changed
owner, directly captured pointer, route, width, epoch or option forces
refusal/recapture; invalidating admission does not release resources that an
executable may still address.
Prepared descriptor and real graph-lifecycle source remain NOT_BUILT and
PREPARED_NOT_RUN.

Each captured continuation shape also has a fixed-capacity lifecycle owner
adjacent to its definition/executable pair. It retains the fingerprint's bound
resource groups, the declared capture-peak record, capture generation, latest
same-stream replay frontier and first invalidation reason. Reset-compatible
entries keep these owners. Replacement and context teardown first establish
final use, invalidate admission, destroy executable then definition, and only
then release the retained groups. An uncertain replay/drain/destroy quarantines
the whole context with the first error and keeps the entry intact. Current
recapture/teardown still uses the existing device drain; the narrower reusable
completion-event pool is a separate workstream requirement, not implied here.

The captured continuation embedding boundary uses one stable device role record
per B4/B6/B8 graph. Token source, hidden destination and embedding-trace
destination are validated and replaced as one host-authoritative generation;
partial tables, reused roles, incompatible geometry and writable overlap refuse
without changing the active generation. The table retains every bound owner,
publishes the complete record on the replay stream before graph launch, and
shares the lifecycle entry's exact replay serial as its final-use frontier.
Changing a source pointer is therefore permitted only through this retained,
all-or-nothing role update; directly captured numerical pointers remain part of
the compatibility fingerprint. The three device records are preallocated in
the prepared B8 continuation allocation, included in its resource accounting,
and never allocated at replay. Source and prepared coverage remain NOT_BUILT and
PREPARED_NOT_RUN.

Reserved serving contexts that enable continuation graphs also reserve exactly
the supported B4/B6/B8 menu before request admission. Each shape declares two
retained native handles and one serialized capture-construction slot. The
coordinator keeps the complete six-handle-plus-one-capture peak charged for the
context lifetime, so later capture or recapture cannot assume resources will
become available. Menu overflow, duplicate/unsupported shapes, unknown extents
and resource-bearing eager fallbacks are rejected before a factory/capture.
Failure to reserve the optional peak records an eager-fallback reason; actual
capture admission checks that reservation before entering CUDA capture. This is
a bounded shape menu, not a Cartesian product of numerical options. Source and
prepared coverage remain NOT_BUILT and PREPARED_NOT_RUN.

Continuation-graph eligibility is revoked centrally before a context strategy,
projection workspace/callback, cache layout, segmented page view, registration,
prefix sharing, OSCAR class or restored physical state is replaced. Revocation
marks every B4/B6/B8 entry invalid without clearing its generation, final-use
frontier or retained resource group. A failed replacement therefore cannot
restore the old executable to service; an in-flight entry remains owned until
its exact stream/serial completes and native handles are destroyed. A compatible
request reset first drains pending use and may retain an unchanged exact key for
later reactivation. A different bound root contract is rejected before mutation
rather than silently rebuilding under a foreign contract. The current mutation
drain is deliberately conservative; bounded request-local completion events are
tracked separately by T198. Source and prepared coverage remain NOT_BUILT and
PREPARED_NOT_RUN.

Each reserved DFlash execution lane now owns a fixed pool of four precreated
completion events, inventoried in an `event_count` domain distinct from device
bytes and graph handles. A slot acquisition binds the request root, acquisition
epoch and execution epoch to a monotonically increasing pool generation;
submission adds a separate final-use generation. Ready is not recyclable:
publication and tap consumers match the exact pool generation/event, and only
explicit retirement releases the request owner and makes the slot reusable.
Cancellation before submission retires immediately, while cancellation after
submission retains the owner until the exact completion notification. Stale or
duplicate notifications cannot certify a reused event address, errors keep the
slot failed and retained, and pool exhaustion is explicit. The actual lane fence
continues to synchronize as the blocking fallback; nonblocking exposure belongs
to T199. Lane teardown retains the complete pool on device-drain or event-handle
destruction failure. Source and prepared coverage remain NOT_BUILT and
PREPARED_NOT_RUN.

The public generation handle also exposes nonblocking `poll()`. It returns a
three-state result: Pending leaves the handle and request ownership live,
Completed transfers the final result, and Error transfers an exception pointer;
both terminal states consume the handle. The generic runtime and EXL3 backends
implement the same contract, while `wait()` remains a blocking loop over their
completion path. Streaming deltas are copied under the request mutex and user
callbacks run only after that mutex is released, with no Engine queue or
coordinator lock held. A callback/cancellation exception becomes terminal Error
and cancels logical work without discarding the worker's physical cleanup duty.
The handle refuses same-handle recursive polling, while callbacks may safely
inspect or advance unrelated Engine work. Prepared C2 source holds one worker,
polls it as Pending, completes a peer, reenters Engine statistics from a stream
callback, and covers cancellation and worker Error. Source remains NOT_BUILT and
the test is PREPARED_NOT_RUN.

EXL3 request execution now carries an explicit control/publication boundary for
every control or model round. Proposal, verification, conditional continuation,
repair and rollback transition through numerical pending/ready states. Private
frontend preview may be prepared after a ready result and may survive a bounded
follow-up numerical step, but control append and resident publication cannot
start while numerical work is pending. `publish_window` is the resident commit;
only its successful return authorizes result-token insertion and staging output
ranges for `poll()`/`wait()` callbacks. Cancellation, deadline and capacity
preflight are explicitly classified as independent observations, while private
output is not available from a pending same-request result. Publication failure
leaves the boundary failed and the existing lane rollback/cancellation path
retains physical cleanup. Prepared passive source covers illegal control append,
follow-up repair, stale tickets and callback order; a real Engine fault before
resident commit must produce poll Error with no visible delta. Source remains
NOT_BUILT and tests PREPARED_NOT_RUN.

Each EXL3 request now also prepares an immutable host prefix plan before its
first numerical operation. The plan contains the exact input fingerprint,
input extent and cacheable/declared frontier only; it does not grant cache
admission, physical-lane ownership or publication authority. A request-local
monotonic ticket must still match when the plan is accepted. Cancellation can
invalidate a completed but unaccepted plan, and stale or mismatched completion
cannot advance prefill. In C2 the Engine records when this plan completes while
the other lane retains numerical ownership, without treating that observation
as elapsed time or shared state. Prepared real Engine source uses nonblocking
poll to keep the peer Pending, completes the host plan on the other worker, and
covers cancellation before acceptance plus unchanged token order. Source
remains NOT_BUILT and tests PREPARED_NOT_RUN.

The selected nondefault-stream DFlash model slice now carries a bounded event
dependency graph from draft selection through KV upload, attention consumption
and exported exact state. The existing T119 bank/plane producer and final-use
events remain the concrete upload/attention subgraph. The lane's pooled final
completion event is recorded on the same explicit retained stream after all
four commands and certifies the composed chain. Every node owner plus the
stream owner remains retained until resident publication retires a successful
graph; cancellation releases only after the rollback stream fence, and failure
retains the graph through an enclosing stream/device drain or quarantine.
Missing edges, cycles, stale terminal events and default/unowned stream identity
are refused. Counts report complete graph submissions, completions,
cancellations and failures rather than kernels or additive time. Source remains
NOT_BUILT and tests PREPARED_NOT_RUN.

Continuation graph admission now constructs an explicit numerical boundary
before either fixed-B8 or B4/B6 capture. The captured body must contain
numerical device work only, use fixed geometry, own its context/input/output,
and carry a current external-dependency generation. A nondefault stream also
requires its retained stream owner. Capture setup allocation and pointer-role
upload, HostKV registration/transfer, exact host export, resident publication
and public callbacks are classified outside the numerical body. If HostKV,
instrumentation callbacks, stale dependencies, unowned streams or dynamic
geometry would enter that body, the existing eager fallback is selected before
native capture. The fingerprint, role table and bounded graph entry remain the
actual input/output ownership path. Source remains NOT_BUILT and tests
PREPARED_NOT_RUN.

### Required versus optional exact-verifier artifacts

Authoritative committed tokens and the complete exact host state are produced by
every scalar, native-B8 and windowed verifier route. The host tap vectors and
their segment diagnostics are demand-driven for standalone exact callers, but
the compact DFlash request path always requests them because publication,
conditional composition and repair consume that payload. It is therefore not a
diagnostic-only transfer on that path and must not be disabled for speed.

A device tap consumer may omit the host vectors only when it presents a strong
context/model/acquisition/execution binding. An unbound consumer is rejected
before restoring or advancing the exact context. The separate history/profile
runner remains an explicitly selected diagnostic and does not change production
artifact demand. Prepared source compares diagnostic-off/on scalar, native-B8
and windowed-B9 results, requires identical tokens and exact state, checks that
only requested host taps materialize, and covers missing mandatory compact taps
and unbound consumer refusal. This source is NOT_BUILT and PREPARED_NOT_RUN.

The resident draft-ring witness is also a content revision plus acquisition and
execution scope, not a pointer or geometric match. Tap commits invalidate the
old witness before the first possible projection submission; successful export
or restore is the only operation that binds a new parent. Device-seed proposal
does not write ring K/V, but it now requires the seed scope to equal the current
completed ring scope before claim and rechecks the same parent/revision after
its final use. Thus a seed cannot borrow a ring restored for another request,
and a subsequent D2D tap commit replaces rather than extends the old witness.

Deferred greedy packets now borrow one of two context-owned transfer slots.
Each slot preallocates its 16-row device destination, portable pinned host
destination, producer/host/consumer events and failure-retention record. A slot
is eligible only when the context is its sole shared owner and the prior
producer plus any claimed consumer have completed; retained seed/batch readers
therefore prevent reuse even after host diagnostics finish. Exhaustion refuses
without growing storage or advancing packet serial authority.

The two device destinations are included in persistent device bytes, their host
destinations are charged in the CUDA-registered-host inventory domain, and the
bounded shared owners plus device retirement records are charged as metadata.
On an uncertain completion the slot is detached from the pool and quarantines
all three domains. The existing 16-row-per-plane lane tap stage remains fixed
and is reused only behind the lane completion/tap-generation fence.

Draft proposal context/block position commands now use fixed arrays owned by
each private draft execution. Their addresses and capacity do not change across
row extents, and a proposal-wide guard rejects reentrant command mutation.
Only these non-escaping commands are reused: the returned token vector remains
an owning value, while the existing position-confidence reference remains
explicitly test-only and valid only until the next proposal. Over-capacity input
is refused rather than growing the command storage.

`--temperature 0` sets the default while preserving explicit positive-temperature request refusal (`400`, `unsupported_sampling`). The existing `--greedy` option forces temperature zero, including requests asking for a positive temperature; use the former when checking refusal behavior. Media, unsupported hierarchy/snapshot modes and incompatible package combinations are rejected rather than silently mapped to another backend.

## Evidence and limits

Campaign: `results/autonomous/astra-solo-real-dflash-20260912/`.

- `ENGINE_TEXT_RESULT.md`: public Engine and loopback streaming/cache/stop/refusal checks. Current common-path regression uses frozen lifecycle-build-r7, lifecycle-r4 and engine-loopback-r6. Two graceful Engine cycles release every recurrent slab and return to the same measured CUDA free bytes. The HTTP runner terminates its bounded server; graceful teardown is established by the separate Engine test.
- `RECURRENT_EXPORT_RESULT.md`: three paired4K/128 cohorts per fixture; complete three-request wall improves about16.72% code/16.78% prose with exact slab export. These are in-process workload results, not HTTP throughput or client TTFT.
- `P1_4K_RESULT.md`: actual B8 direct DFlash2->L2 and persistent-resource comparison, with full state/tap/independent draft-ring checks.
- `DRAFT_OWNERSHIP_RESULT.md`: shared immutable draft weights and qualified simultaneous physicalC2 research. `C2_ENGINE_RESULT.md` qualifies explicit `--max-concurrency 2` in the existing Engine/server, including concurrent clients, reasoning control, bounded queues and SSE disconnect/peer recycling. The default remains C1; backend throughput comparisons are separate from client measurements.
- `V6_RESULT.md`: native encoder/merger implementation, successful input/ownership checks and preserved failed full numerical gate. Image/video understanding, L2 media injection, three-axis MRoPE and media-conditioned DFlash2 serving remain unqualified.

Client TTFT, in-process first publication, cold model startup, context construction, cached-prefix work and steady decode are separate measurements. Payload accounting is not a physical-memory peak; inspect the raw sampled telemetry and allocation/free snapshots before planning another physical lane.


Internal media execution remains research-only. `NINFER_EXL3_MEDIA_EXECUTION_RESEARCH=1` reserves163936 bytes per ordinary eager context for up to8 FP32 embedding rows and three-axis positions. Exact state preserves a rotary offset separately from logical KV extent. Scalar text equivalence, MRoPE formula, restore/continuation and complete prepared image chat with actual DFlash proposals have bounded functional evidence in `results/autonomous/astra-solo-real-dflash-20260912/MEDIA_PREPARED_RESULT.md`. The Engine rejects this flag: the full V6 CPU/GPU numerical gates still fail, and public media identity/cache/publication/image-video quality remain unqualified. These tests do not enable media HTTP or replace T23 defaults.

An explicit physicalC1 menu also accepts `NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS=16384` with cache flag1 and context capacity at least16384. It reserves1GiB within the same1GiB processcap; competing reservations use visible exact fallback. PhysicalC2 with this row setting is refused. The4096-row default remains256MiB per context. Stopped16K six-request timing versus4Kcache passed three exact pairs with median paired+1.7074%, additional768MiB and no first-release improvement; see `DEVICE_PREFIX_16K_RESULT.md`. C1 Engine/HTTP and default4K C2 regression pass.

`NINFER_EXL3_EXTENDED_STREAM_REDUCTION=1` is an additional explicit exact option for target K7N32 and K6/K7 large-downN16 rows1..8. It preserves original split/MMA/reduction arithmetic and uses the existing scratch. Three4K128 pairs qualify median gains of0.6140% C1 and0.9761% C2 with K6N32/slabs on both arms; this is a separate cohort with device cache off, not an additive combination claim. Engine C1 with16K cache and C2 with4K cache, plus existing C2 HTTP, pass functional/resource composition checks. H6, draft/K5, wide/native16 and graph profiles are outside the extension. Default0 remains.
# T204 and T209-T211 source reconciliation (implementation only)

The serving Engine cannot legally adopt the isolated OSCAR continuation graph:
its exact profile requires ordinary-FP16 HostKV, while the graph captures the
canonical OSCAR layer stack and excludes HostKV transfer/registration. This is
an exclusive-alternative disposition, not an unimplemented switch to enable.

Existing serving ownership already covers the adjacent recurrent/state work.
Recurrent slab growth and borrower controls enter the coordinator inventory;
cached roots deduplicate the same slab allocation; process live, peak and
quarantine bytes remain explicit. Exact export reserves metadata and borrows a
recurrent destination before snapshot/page construction, retaining exact
pageable fallback. Token histories use immutable bounded-depth radix pages so
branch appends copy only the affected path and preserve old roots.

Status: T204 `INAPPLICABLE_EXCLUSIVE_ALTERNATIVE`; T209-T211
`COVERED_BY_EXISTING_SOURCE` / `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`.
NOT_BUILT / PREPARED_NOT_RUN / QUALIFICATION_PENDING.

## T205 graph accounting boundary

The isolated OSCAR continuation graph reports preparation, compatible hits,
reactivation, eager fallback, replay submission, final-use observation and
destruction independently. B4/B6/B8 retained resources remain separate because
their entries can bind the same physical context/model allocations. Startup
peak reservation is not steady retained memory, and CUDA driver allocation
bytes remain marked unknown. These counters do not imply Engine integration or
an additive end-to-end duration.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T206 deferred graph failure source

Graph executable upload and launch use injectable providers whose defaults are
the native CUDA operations. This permits no-device source coverage of failed
launch while retaining the executable. The bounded lifecycle fixture couples
that case to exact replay generation, resource bytes and quarantine ownership;
existing menu/binding/retirement fixtures cover pressure, stale pointers and
shutdown failures.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T207-T208 graph scheduling disposition

No continuation-graph stage enters the ordinary-FP16 HostKV Engine scheduler.
The isolated OSCAR graph is synchronous and context-local, with eager refusal
before capture when resources or dependencies are unavailable. Graph fairness
and graph/Engine overlap are therefore inapplicable to this selected route;
existing Engine host overlap and OSCAR graph tests remain separate.

## T212 exact export copy plan

Exact-host recurrent FP32 and convolution FP16 exports use fixed descriptor
arrays. A copy is coalesced only when both physical address ranges are adjacent,
precision matches and padding is absent. Logical ranges, submitted copies and
coalesced ranges are reported separately; no staging buffer or copy-elimination
claim is introduced. First-error submission leaves the candidate state private.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T213 publication scratch ownership

Selected entries, replacements and resident-root traversal reuse coordinator-
owned startup vectors. Publication and root traversal have independent epochs
and active guards under the coordinator mutex; all exits clear temporary strong
roots while retaining backing capacity. This mutable scratch is never embedded
in immutable request roots.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T214-T215 admission and resident coverage

Existing staged Engine/coordinator admission retires provisional logical and
physical owners before queue publication on every failure edge. Resident
assessment uses actual rounded address ranges and owner identities, merges the
authoritative union, locks only added coverage and preserves the old union on
foreign-lock or partial-lock failure.

Status: T214-T215 `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T216 optional snapshot-side ownership

Exact host snapshots do not discard any tensor, identity or extent needed by
restore, replay or publication. Compact request roots release their immutable
full-tap history only after a replay-capable draft ring has been exported; the
ring remains strongly owned by the root. Diagnostic represented-value copies
are explicit independent shared owners and retain no raw context, model,
registration or source-page lifetime.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T226 bounded longest-prefix range index

The EXL3 prefix index supports at most 64 entries, matching its fixed length-
order and retention-decision metadata. Longest-prefix lookup lower-bounds that
descending order and stops below the caller's minimum frontier; larger
configured capacities are rejected instead of selecting a full-scan fallback.
The length order is only a shortlist and never replaces exact model, contract,
prepared-media identity or token comparison.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T228 shared immutable token history

Token history is a bounded-depth immutable radix spine. Forks retain common
nodes; append reconstructs only changed partial leaves and affected paths, and
bulk append materializes each changed subtree once. Exact comparison may skip
identical node owners but recursively compares all distinct represented pages.
Prepared identity and compatibility configuration remain separate immutable
root contracts.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T229 cost-aware retention metadata

Each prefix entry reports its immutable root and generation, reusable tokens,
unique retained bytes, optional measured preparation/access observations and a
separate saturating lookup-selection counter. Missing observations remain null;
the cache does not infer a calibrated value from tokens or lookup counts.
Serving cold preparation records elapsed time for the admitted generation, and
Engine policy decisions remain bound to the exact root/generation snapshot.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T230 bounded optional retention policy

Policy trim and policy-aware admission require a complete ordered root/
generation snapshot. Required entries remain protected. When all eligible
entries have a supplied fixed-point value-per-byte, the lowest value is removed
first; missing values select stable FIFO fallback, including deterministic equal
priority. Accounting and destination allocation finish before cache mutation,
and readers retain independent strong ownership after cache eviction.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed test source NOT_BUILT,
tests PREPARED_NOT_RUN, qualification pending.

## T227 committed completed-turn frontier

The Qwen3.6 output session publishes a completed conversational frontier only
after committing a model preview terminated by the canonical special
`<|im_end|>` delimiter. Generic EOS, caller stop strings or tokens, limits,
cancellation and discarded previews do not set it. Actual EXL3 completed-root
admission consumes this frontend semantic state rather than inferring completion
from the last generated token.

Completed roots retain the immutable token-history spine. Append materializes
only the changed partial tail and radix path and advances the exact fingerprint;
unchanged ancestors remain shared, immutable owners.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T231 exact request-affine lane attachment

`NINFER_EXL3_REQUEST_AFFINITY=1` is a default-off physical-C2 route requiring
exact acquired-root preservation. It does not reorder requests: only the FIFO
head is considered. An idle lane is preferred only when its current context and
generation prove exact residency of a weak immutable text root whose full token
history prefixes the queued input. Busy or invalid hints fall back immediately.
Hints retain no resource owner and are published only after successful lane and
coordinator completion.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T232 exact target/draft acquisition preservation

`NINFER_EXL3_PRESERVE_ACQUIRED_ROOT=1` remains default-off. The actual DFlash
lane preserves target payload only after draining prior work and proving the
current request generation, serving compatibility contract and exact resident
immutable state. Reset still clears request-local graph, transaction, staging
and publication state.

The private draft ring has an independent witness. A new current coordinator
lease may inherit it only when the identical immutable projected ring remains
resident under a completed nonzero prior acquisition/execution, with unchanged
model, content revision, base/count and healthy transfer state. Otherwise the
new scope invalidates old residency and performs the full restore. Typed/media
roots deliberately use full target and draft restoration. Target and draft
decisions have separate counters.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T233 declared stable tool/chat checkpoints

With `NINFER_EXL3_ENGINE_DECLARED_PREFIX=1`, actual Engine preparation may cache
only frontend-declared `SharedStablePrefix` message/tool frontiers within the
immutable rewrite cap. The retained exact root contains complete target state
and target taps/private-conditioning reconstruction authority, and lookups still
require exact tokens, model ownership and the full serving configuration
contract. Tool text and argument order are not normalized, and generated output
is not part of this input-frontier decision.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed test source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T234 bounded concurrent prefix preparation

`NINFER_EXL3_ENGINE_SHARED_PREPARATION=1` is a default-off C2 text route. One of
eight startup-reserved flights owns each exact model/epoch/serving-contract/
token/prefill preparation. The producer may start cold or extend a strongly
owned cached root; it admits the completed immutable root before publishing to
joiners. Joiners restore into private contexts, while pool saturation runs
independently. Suffix-only extension time is not recorded as full cold cost.

Cancellation belongs to one consumer. A waiting consumer can leave without
poisoning the producer, and a producer callback cancellation is deferred only
while a registered waiter needs the synchronous result. Preparation or admission
failure fans out, reload wakes old-epoch waiters, and pooled state clears after
the last participant. Prepared media is not supported by this selected Engine
and is rejected before text-flight registration rather than losing payload
ownership or being treated as text.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T235 explicit prepared-media payload retention

Prepared prompts now distinguish live `ReplayRequired` encoder inputs from an
`IdentityOnly` descriptor state and report the current logical owner count plus
deduplicated payload-object/patch-array bytes. The historical preparation stat
remains a snapshot and is not confused with current ownership.

Actual Qwen request materialization declares the strictly ordered Vision items
that the selected suffix can still encode. The declaration is validated before
any payload is released, retains shared immutable owners only once physically,
and retires every unused item. Each used owner is released after its encoder
handoff is consumed. Exact EXL3 typed roots retain complete represented target
state, taps/private conditioning and immutable media identity but no preprocessing
payload; they continue to refuse token-only media reconstruction.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T236 exact ancestor attachment

Ordinary shared text pages are attached only when the acquired immutable request
root owns the exact full-page control block under the same strong model owner and
RoPE offset. Equal page bytes, equal offsets, a sibling branch, a partial page or
a typed MRoPE/media root cannot authorize the page-copy or direct-attention path;
those cases retain the private exact restore/attention route.

Committed tap D2D assembly now carries the authorizing request revision and root
frontier in addition to context/model ownership, writer generation and execution
scope. The real verifier derives that binding from the acquired root before each
regular, conditional-second or repaired-second continuation and rechecks the
strong revision control block and exact frontier before copying any tap plane.
A same-address foreign control block is not an ancestor proof.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T237 bounded canonical-render stream

Qwen prompt preparation now feeds both real text and multimodal branches through
a bounded append/finalize stream over the one canonical chat-template result.
Each append must match the next authoritative rendered bytes exactly and checks
request cancellation. The stream borrows no bytes beyond synchronous preparation,
pre-reserves its declared segment capacity and cannot grow past it.

Text chunks are never tokenized independently: finalization applies the existing
whole-string tokenizer and byte-boundary mapping once, preserving cross-chunk BPE,
UTF-8, added control tokens, rewrite checkpoints and message/tool/cache frontiers.
Expanded media items carry complete ordered byte ranges; a range cannot be split,
reordered or relabeled as token text. Prepared media owners remain under their
existing replay-required/identity-only lifetime contract after tokenization.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T238 checkpoint-root and shared-page coherence

A current prefix-retention transaction now reconciles the coordinator's host
resident union before returning. Removing exact lookup metadata therefore drops
that logical root's ranges immediately, rather than waiting for an unrelated
request publication. Queued or active roots are still forced required and
cannot be selected as retention victims.

Shared device-page fills and readers remain independent owners. Their bounded
runtime-source binding holds the exact host page in the same resident union, so
cache replacement, explicit trim or model-namespace reload cannot invalidate a
delayed transfer or attached reader. The source range and its resource charge
leave the union only after the fill/read lifetime expires and a current
coordinator reconciliation observes that final release. Exact ordered root and
generation metadata remains mandatory for every trim.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T239 deferred multi-turn branch/cache suite

The existing Serve TTFT corpus and black-box client already define the required
future public-client graph: anonymous and stored multi-turn continuations,
repeated and branched marked system prefixes, stable and changed repository-tool
schemas, capacity-one shared replacement, Device/Host/evicted/catalog pressure,
transport cancellation followed by an isolation probe, and exact/append/changed
media histories. Frozen request files and media have committed hashes and token/
frontier facts; the timed client does not tokenize, normalize or consult private
Engine diagnostics.

External cold, hot and pressure TTFT observations remain distinct report rows
and comparisons, never evidence that a private hit or eviction occurred. The
host-side frontend/prefix fixtures separately establish exact rendered-token,
marker, model/configuration and prepared-media identity; changed bytes, tokens,
tool/system content, unmarked common text and stale generations miss without
normalization. They also cover cancellation, pressure eviction and strong roots
surviving lookup-slot removal.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T240 actual prefix economics attribution

The EXL3 Engine now publishes existing prefix-selection and prompt-work counters
at the successful preparation boundary. A cold or rebuilt root increments
`root_selections`; an exact cache hit or joined preparation increments
`shared_stable_prefix_selections`. `reused_prompt_tokens` and
`last_selected_frontier_tokens` use the exact selected frontier, while
`computed_prefill_tokens` adds only input tokens actually evaluated after that
frontier.

Attribution occurs before compaction, request admission and output publication.
Consequently a later cancellation, failure or exact max-context return cannot
erase prefill work already performed. It also cannot convert token counts into
saved time: cache retention keeps its separately supplied/null preparation and
access observations, affinity retains its exact assignment/fallback counters,
and external cold/warm TTFT remains a future black-box comparison.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T241 immutable ready-work descriptor

Every EXL3 submission now publishes one immutable ingress descriptor before it
can enter the reserved request queue. The descriptor owns the exact converted
prompt-token vector formerly borrowed from mutable request storage, a monotonic
generation, submission time, the exact logical-host charge, the absence of
request-local device allocation, the one-lane/preallocated-context requirement,
and a categorical strong-owner execution signature for the existing ordinary-
FP16 B8 greedy text HostPreparation stage.

Cancellation, dependency availability, lane availability, resource
availability and observed acquisition generation are separate current-state
inputs. Their assessment distinguishes logical queue admission from physical
executability and reports cancellation, stale generation, blocked dependency,
missing lane or missing resource without rewriting the published numerical or
resource identity. Actual affinity and execution consume the descriptor's owned
tokens, and the worker checks its current assessment before device selection.
Later numerical-stage descriptors remain the existing strong-owner projection
rendezvous records; this change does not add a second scheduler.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T242 atomic readiness and resource snapshot

The existing Engine mutex now guards a bounded per-lane ready-work snapshot
authority. Selection captures the immutable T241 observation and the current
descriptor generation, dependency, physical-lane, logical-resource and
cancellation facts in one ticket. The same critical section advances the queue
revision, consumes the newest lane ticket exactly once, removes the queue head
and publishes active ownership. Queue insertion/removal/completion and idle
changes to retention, metadata headroom, HostKV/staging routes and supplied
numerical policy advance the same authority revision.

Only a current executable assessment can yield a grant. A changed revision,
generation, signature, age origin, input identity/extent, resource requirement,
dependency or cancellation state refuses consumption. After the Engine lock is
released, the worker rechecks cancellation and the consumed lane/generation
before device selection. Host preparation and numerical execution never run
while the Engine mutex is held. Later projection pairing retains its existing
rendezvous-mutex lease/epoch recheck for those stage-specific acquisitions.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T243 context-locality scoring with hard constraints

The default-off C2 request-affinity path now describes each idle lane with hard
physical-resource and request execution-signature feasibility, exact reusable
root tokens, and optional measured root-attachment and history-restore
microseconds. A retained root contributes locality only after current context
generation, model identity, non-media input identity, exact token prefix,
resident host state and context extent all pass. An incompatible or stale root
therefore leaves that lane as a legal cold fallback and contributes no warm
cost.

Costs are used only when every feasible candidate has both observations and the
sum is representable. If any value is absent, the selector does not treat it as
zero or infinity; it falls back to longest exact reusable prefix and stable lane
order. Actual observations come from completed preparation attachment timing and
cache-hit/join restore timing. They remain local scheduling inputs, not
calibrated performance claims, and the existing FIFO queue head is never
reordered.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T244 age-fair feasible request choice

C2 workers no longer wait solely on the queue front when that request is
physically assigned to the other lane. Under the existing Engine mutex each
worker scans the reserved queue without allocation, applies T241/T243 hard
readiness and lane-affinity constraints, ignores cancelled entries, and selects
the oldest submitted request feasible on that lane. Queue position is the
stable tie-breaker. Removing one selection wakes the other worker so it can
reassess the remaining queue under the new active-lane state.

This is an equivalent fairness rule rather than a latency promise: repeated
newer arrivals cannot overtake older eligible work; an old resource-ineligible
entry cannot block a feasible peer; and once that entry becomes eligible its
original submission age takes precedence. Cancellation remains a separate
state and cannot gain priority merely by being old. The T242 snapshot is still
consumed before the selected entry changes from queued to active.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T245 batch formation from compatible ready rows

The Engine does not form numerical batches from ingress queue membership.
Target and private-draft projection calls first reach their real suspended
layer boundary and publish immutable offers containing exact row extents,
stage/family/layer identity, model and request owners, weights/metadata shapes,
activation witnesses, control-token identity, cancellation state, coordinator
lease and bounded wait budget. Only two current offers that satisfy the same
execution contract, supported family geometry, distinct-request ownership and
combined M2..M16 extent may claim the packed dispatch.

An incompatible contract, target/draft cross-stage offer, invalid/overlapping
row geometry, stale authority, cancellation, missing peer or unsupported
candidate falls back before physical shared dispatch and never converts queued
requests into claimed rows. Unequal extents retain explicit packed offsets and
private result destinations. Minimum request progress is therefore preserved by
the ordinary per-request fallback, including a lone numerical row.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T246 short-request latency guard

Optional projection waiting is authorized only by the fixed supplied-cost
policy for the exact family, matrix shape, bit width and first-row extent. An
absent or non-profitable entry immediately uses private execution. Otherwise
the immutable offer owns a positive wait budget bounded by both the entry's
declared maximum age, the strict independent-minus-shared saving and the
Engine's finite rendezvous cap.

The first offer uses one absolute `wait_until` deadline. The arriving peer then
passes the same offered-time/budget boundary again at membership claim; arrival
at the deadline is expired. Cancellation wakes the unclaimed waiter, and every
timeout removes that stack-owned offer before private fallback. Claimed work is
never revoked by this latency guard. A near-final request therefore receives at
most the same bounded single-stage delay and cannot wait indefinitely for a
larger batch or accumulate credit across later offers.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T247 resource-safe preparation and verification interleaving

The supported interleave is the existing default-off C2 concurrent-prefix path,
not a dynamic workspace transition. Before either worker exists, the single
coordinator retains both complete context/draft/lane allocations, two tap
staging owners, two execution-stream owners and the fixed preparation-flight
pool sized for the configured context. Constructor failure rolls the whole
unpublished Engine back; saturation of the bounded flight pool performs
independent caller-owned preparation instead of uncredited growth.

Each request retains canonical order: immutable host plan acceptance, prefix
lookup/restore or numerical suffix, compact-draft construction, coordinator
acquisition, private proposal, target verification, resident commit and output
publication. Cache/service locks are released around numerical preparation and
callbacks, so a different fully preallocated lane may verify while preparation
is pending. Cancellation only declines that request's result; it cannot revoke
a successful shared root needed by a peer.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed test source NOT_BUILT,
PREPARED_NOT_RUN, qualification pending.

## T248 fixed-capacity disposition for context expansion

Live context expansion is an inapplicable alternative for the current Engine
contract. `EngineOptions::max_context` is the declared supported limit and every
physical lane is constructed with that exact immutable capacity. Target KV,
HostKV/OSCAR storage, attention workspaces, continuation storage, optional graph
classes, reconstruction, registration/staging and prefix owners are selected
and retained before worker publication. There is no smaller live context that
may be resized or silently substituted.

Prepared inputs longer than the configured limit are rejected before request
storage or queue publication, even if a caller supplies a misleading summary.
At the exact limit, completed prefix state remains authoritative and generation
returns `ContextCapacity`; a request with fewer remaining rows receives only
that exact allowance. Strategy changes that would alter cache or graph layout
require a pristine inactive context and invalidate retained graphs rather than
replacing storage under old readers. A future dynamically growable Engine would
need a new old/new union transaction satisfying T026/T028 before becoming an
applicable implementation.

Status: `INAPPLICABLE_EXCLUSIVE_ALTERNATIVE` (fixed preallocated context);
existing source/tests NOT_BUILT, PREPARED_NOT_RUN, qualification pending.

## T219 bounded streaming ownership

A streaming request reserves two bounded delivery slots in addition to its
result/session/input storage. Each `OutputDelta` carries an opaque internal
lease, so a sink that retains the delivered value also retains the request's
logical-host charge. While both slots are outstanding, polling returns Pending
without copying or consuming another committed range. Releasing a value returns
one slot and wakes delivery. Sink callbacks remain synchronous and outside all
Engine locks; application-created copies after delivery are application-owned.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T224 composed state/workspace regression

The prepared Engine lifecycle mode
`NINFER_TEST_ENGINE_STATE_WORKSPACE_COMPOSITION=1|2` selects physical C1/C2.
It composes pinned recurrent-slab ownership, immutable prefix roots, execution
dependency graphs and bounded output delivery with cancellation, cache eviction,
exact target-state/private-ring comparison and replacement Engine construction.
The optional existing C2 shared-page route is checked for completed page/source
ownership; with the route off, page traffic and independently gated shared
projection must remain absent.

This is test source only. It has not been built or run and provides no numerical,
resource, performance or qualification result.

## T225 retained prefix lookup

Prefix-index lookup keeps Entry metadata inside the index lock and returns only
strong ownership of the selected immutable request root. Hash and length
shortlists remain accelerators: model, compatibility contract, prepared-media
identity and exact tokens are checked before selection. Serial and concurrent
serving callers may safely restore or extend the returned root after the index
lock is released, even if the slot is subsequently replaced or evicted.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T220 request-root allocation inventory

Resident publication obtains logical metadata allocations from the immutable
request root. The root visitor covers request and revision blocks, prepared
identity, draft ring and pages, exact KV pages, tap blocks, token-history nodes
and the exact snapshot. It preflights all supplied roots, deduplicates aliases
by owner and allocation kind, and retains a strong erased owner beside the
exact lifetime-credit callbacks.

Logical metadata units are intentionally separate from physical tensor/vector
backings. Physical ranges continue through the payload visitor for residency
and registration planning; recurrent slab metadata remains charged by its
independent pool owner rather than once per snapshot.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T221 epoch-safe request arenas

The fixed 16-token conversion/repair arena exposes borrowed views only inside
one request worker's synchronous round. Each view captures a monotonic arena
epoch; assign or truncate advances the epoch and all operations on an older
view refuse. Committed tokens are copied into the request's owned result before
the next numerical round can reuse the backing.

Public results, streaming deltas, immutable roots, device-page readers,
compact greedy packets and diagnostic host returns preserve their owning
lifetime contracts because clients may retain them asynchronously or across a
reload.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T222 cache retention versus resident publication

Prefix-index retention is a cache policy, not a page-lock authority. Before a
retention transaction, the coordinator marks every matching queued or active
request root required. The cache operation cannot publish, replace resident
coverage or unlock pages. Resident coverage is instead rebuilt transactionally
from cache roots, logical request roots and registered source owners during the
normal admission/publication/completion lifecycle.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T223 resource attribution

Current resource attribution records identify the actual allocation control
owner, object address, slot, domain and exact units. A live record is distinct
from metadata retained pending the final original owner. Removal deletes the
record; it is not reconstructed from allocator or driver observations.

The Engine reconciles its final existing startup availability observation with
the exact retained device inventory. Any remaining observed use is reported as
driver-unknown bytes. It is neither a leak claim nor a cache claim, and
component-specific quarantine records remain separate.

Status: `IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; changed source NOT_BUILT, tests
PREPARED_NOT_RUN, qualification pending.

## T218 fixed request workspace classes

Each actual Engine lane owns startup-allocated, strategy-bound ordinary target,
B8 continuation, private draft and tap-staging storage, plus only the selected
optional transaction/reconstruction/device-prefix classes. The 2/4/8 horizon
policy stays within the B8 class. Capacity changes and context growth reject;
optional reservation failure uses its declared fallback and never resizes a
live workspace. Request result/session/input backing is established before
queue publication.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.

## T217 workspace interference boundary

Engine target, draft, transfer/tap and dependency-graph storage remain separate
owners. Asynchronous transfer and captured/failed graph consumers retain their
owners through terminal observation or an enclosing drain; stream separation
does not prove reusable lifetime. The only enabled alias is the proven
disjoint input-normalization/MLP interval inside one ordinary eager attention
forward, and diagnostic/profile, OSCAR, graph-capture and pending-consumer
profiles reject it.

Status: `COVERED_BY_EXISTING_SOURCE` /
`IMPLEMENTATION_COMPLETE_NOT_EXECUTED`; NOT_BUILT, PREPARED_NOT_RUN,
qualification pending.
