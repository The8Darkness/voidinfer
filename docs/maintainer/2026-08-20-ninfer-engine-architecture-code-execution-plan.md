# NInfer 引擎架构代码闭合执行方案

> 状态：临时代码执行文档
>
> 日期：2026-08-20
>
> 适用阶段：现有功能的架构闭合，不包含 KV offload、COW 或新产品能力
>
> 生命周期：本阶段代码完成并通过验收后，由后续统一文档任务吸收结论并删除本文件

## 1. 本阶段交付结果

本阶段把已经确定的四层架构落实为唯一代码路径：

```text
Gateway
   ↓
Frontend
   ↓
Engine
   ↓
Runtime
   ↓
core / ops
```

目标不是重新论证顶层架构，也不是先实现未来缓存能力，而是闭合当前源码中尚未对齐的所有权、生命周期和 Engine/Runtime contract。完成后应同时满足：

- 公共 `.ninfer` `Engine` 路径及本次受影响的可观察产品 contract 保持不变；
- `Engine` 明确拥有请求生命周期、调度、公平性、逻辑资源策略和 continuation catalog；
- `Runtime` 明确拥有模型状态、KV、request transient、pending execution transaction 和物理资源变换；
- 27B 与 35B-A3B 继续作为同一 Qwen3.6 family runtime 的两个对等 compile-time package；
- 各 artifact identity 在自身已支持的 Text、Vision、MTP、DFlash、prefix reuse 和 1–8 路并发范围内，均通过同一闭合的 Engine/Runtime contract；
- 不保留旧接口别名、兼容层或并行执行路径；
- 不在当前代码中加入 offload、host KV、COW、共享页引用计数或空实现占位。

除本执行文档外，本阶段不修改任何长期维护文档。README、CLI、serving、模型、并发架构和 paged KV 文档的统一修订属于后续独立任务。

### 1.1 约束层级

本文将实施约束分为两类，避免把示意表示误作语义 gate：

- **固定语义**：产品边界、所有权、状态转移、资源方程、调度顺序、失败后置条件和不保留旧路径是完成条件；本文中的“必须”、“不得”和“唯一”只用于这一类。
- **局部表示**：私有 helper 名称、aggregate 字段排布、整数底层类型、header-only 或 `.cpp` 的选择可按 C++ 直接性实现；但不得改变前一类语义。文中标明为示意的 struct 不是 source-shape test 目标。

## 2. 固定产品边界

本方案只服务当前产品矩阵：

| 执行 package | 注册 artifact identity | 当前能力 |
|---|---|---|
| `qwen3_6_27b` | `qwen3.6-27b/groupwise-int` | Text、Vision、MTP、prefix reuse |
| `qwen3_6_27b` | `qwen3.6-27b/nvfp4` | Text、Vision、MTP、prefix reuse |
| `qwen3_6_27b` | `qwen3.8-27b/groupwise-int` | Text、Vision、MTP、prefix reuse |
| `qwen3_6_27b` | `qwen3.8-27b/nvfp4` | Text、Vision、MTP、prefix reuse |
| `qwen3_6_35b_a3b` | `qwen3.6-35b-a3b/groupwise-int` | Text、Vision、MTP、prefix reuse；另有 text-only DFlash |

固定运行模型是单 GPU、单 resident model instance、启动时固定 1–8 active requests、bounded FIFO ingress、一个 GPU worker、每个 round boundary 形成 maximal compact decode batch、无 request preemption。

以下内容不因架构重构而改变：

- `.ninfer` 是唯一产品 artifact；
- `include/ninfer/engine.h` 与 `include/ninfer/types.h` 是唯一公共 C++ 产品表面；
- CLI、OpenAI、Responses、Anthropic、benchmark 都只调用公共 `Engine`；
- OpenAI/Anthropic schema、SSE 事件和 finish reason 行为不变；
- Qwen3.8-27B 仍由 27B package 静态承载，不增加运行时 family 选择；
- Vision 与 MTP 可按现有路径组合；DFlash 仍是 35B-A3B 的 text-only speculative backend，与 Vision/MTP 互斥；
- CUDA Graph、ReplaySSM、GDN state、Main/MTP/DFlash KV 的数值与物理布局不在本阶段重写。

## 3. 最终所有权

| 边界 | 代码位置 | 唯一所有者职责 | 明确不得拥有 |
|---|---|---|---|
| Gateway | `src/serve`、CLI/application | wire schema、HTTP/SSE、协议错误、response store、消费 committed events | tokenizer、prefix policy、KV、CUDA |
| Frontend | `src/product`、`src/media/decode`、`src/targets/qwen3_6/impl/frontend` | 输入适配、媒体准备、template、tokenizer、Vision/MRoPE、`PreparedPrompt`、`OutputSession` 语义 | pending queue、lane、KV capacity、decode scheduling |
| Engine | `src/runtime/engine` | bounded ingress、request record、Scheduler、ResourceManager、continuation catalog、worker、output transaction/publication | 模型 tensor、物理 KV、GDN state、request device arena |
| Runtime | `src/targets/qwen3_6/impl/runtime` 及两个 exact package | request planning、exact prefix verification、sequence/continuation、物理 entitlement、prefill/decode/speculation、commit/finish/abort | FIFO、公平性、协议、session/cache 价值策略 |
| core / ops | `src/core`、`src/ops` | 通用 device primitive、arena、paged KV container、graph RAII、语义闭合 Op | request policy、target registry、协议 |

`include/ninfer/Engine` 是公共 facade，可以同时委托 Frontend 与 Engine 内核；这不构成层级混合。`Package::Program` 继续作为 Runtime 的具体实现名，不为概念命名而机械重命名。

## 4. 必须分开的四类生命周期

现有耦合的根因之一，是将 product response lifetime、模型执行、continuation residency 和单轮 GPU transaction 混为一套状态。最终代码分别表达以下四类事实。

### 4.1 产品请求/响应生命周期

当前产品有两个不同的 bounded permit，不得合并计数：

```text
Gateway RequestLifetime:
ACQUIRED_BEFORE_PREPARE → HTTP_RESPONSE_RESOURCE_RELEASED

Engine executed request:
SUCCESSFULLY_SUBMITTED → response_done
                       └→ consumer_released
capacity_released := first(response_done && consumer_released)
```

- Gateway `RequestLifetime` 从 media acquisition/Frontend prepare 之前占用，由 HTTP response 持有的 resource 生命周期归还；它不进入 Engine `RequestRecord`。
- 对实际进入 executor 的非零输出请求，Engine outstanding capacity 从成功提交起占用。
- `response_done` 与 `consumer_released` 是两个可任意先后到达的 latch，不是线性 enum；两者首次同时成立时 exactly-once 归还 outstanding capacity。
- 销毁尚未 wait 的 `GenerationHandle` 先设置 cancellation 和 `consumer_released`，worker 之后完成 response latch；如果 response 已先完成，销毁 handle 则立即闭合 handshake。
- `requested_output_tokens == 0` 继续走 `ImmediateSubmission`，不进入 executor/`RequestRecord`，也不占用 Engine outstanding capacity。

### 4.2 Engine 模型请求生命周期

```text
WAITING → PREFILL → DECODE_READY → MODEL_FINISHED
             └───────────────→ MODEL_FINISHED
```

状态存在性固定为：

| 状态 | prompt/base plan | sequence/budget | Runtime lane/resource |
|---|---|---|---|
| `WAITING` | owning prompt；base plan 可已建立 | 无 sequence、无 generation budget | 无 active entitlement |
| `PREFILL` | prompt 已由 successful start 消费 | 恰有一个 sequence 和 budget | 恰有一个 active lane；全 Engine 最多一个 prefill owner |
| `DECODE_READY` | 无 owning prompt/candidate plan | 恰有一个 sequence 和 budget | 恰有一个 active entitlement，可进入 maximal compact batch |
| `MODEL_FINISHED` | 无 Runtime plan/prompt ownership | 无 sequence、无 active budget | 无 active lane/entitlement；response consumer 可继续存在 |

只有 worker 的显式 transition helper 修改 model state，转移顺序固定为：

| 事件 | 转移 |
|---|---|
| 任意 successful start 返回（partial 或 complete） | 先核对实际 A，再无抛安装 sequence/budget/A、提交 ResourceManager `Free/Resident → Active`，并执行 `WAITING → PREFILL` |
| 已接管的 start/advance 返回 licensed token，commit 后未终止 | `PREFILL → DECODE_READY` |
| prefill 未产生 licensed token | 保持 `PREFILL` |
| decode commit 后未终止 | 保持 `DECODE_READY` |
| successful terminal commit | 同一 worker 调用栈内继续 `finish` 和 ledger 转移，然后进入 `MODEL_FINISHED` |
| active boundary cancellation | Runtime `abort` 和 ledger 释放后进入 `MODEL_FINISHED` |
| pending-batch snapshot cancellation | commit row disposition 确认 sequence 已释放、ledger 转移后进入 `MODEL_FINISHED` |
| waiting cancellation/request-local error | 不接触 Runtime，直接进入 `MODEL_FINISHED` |

successful start 的接管发生在解释其 `PrefillProgress` 之前；即使 start 直接返回 complete 加单 row `PendingBatch`，RequestRecord 也不会以 `WAITING` 身份持有已激活的 Runtime sequence。terminal commit 与 `finish` 之间只有一个调用栈内的 finish-pending 条件，不增加可调度 Engine state。Engine request state 不包含 `PENDING_RESULT`、`RETAINED` 或 host/device residency。

### 4.3 Continuation 生命周期

成功终止后是否保留 continuation 是独立决策：

```text
ACTIVE_SEQUENCE → RETAIN_RESIDENT → CATALOG_ENTRY → CLAIMED 或 RELEASED
                └→ RELEASE
```

当前阶段 continuation 的语义固定为：

- resident、lane-affine、独占 capability；一个 lane 最多一个 catalog entry；
- catalog entry 被 claim 后即从 catalog 移除，handle 由当次 start 消费；
- 不允许两个请求共享一个 continuation，不允许跨 lane restore、host residency 或 COW；
- successful terminal 的当前 policy 固定为 `RetainResident`；取消、abort 和 engine failure 走显式 release/Program cleanup；
- ResourceManager 可在 admission boundary 为完整 request entitlement 驱逐 resident continuation。

`MODEL_FINISHED` 与 `RETAINED` 不得成为同一个状态或同一个布尔值；前者属于 RequestRecord，后者只由 ResourceManager catalog 与 Runtime continuation 表达。

### 4.4 Runtime 局部 transaction

```text
IDLE → PENDING_RESULT → COMMITTED
                      └→ ABORTED
```

这是 Program 内部一次 prefill-output 或 decode batch 的短生命周期，不是 request state。Program 同时最多存在一个 unresolved `PendingBatch`。`PendingBatch` 是 move-only linear token，正常控制流中恰由 `Program::commit` 或 `Program::abort_pending` 之一消费；析构不触发 Program/CUDA mutation。只有 engine-wide failure 的最终 `fail_all_cleanup` 可以越过该 token 清空整个 Program，且此后 Engine 不再执行请求。

在 unresolved batch 上不得逐 row 调用普通 sequence abort。preview 或 commit 前的 Engine 失败先调用 `abort_pending(PendingBatch&&) noexcept`，然后进入 `fail_all`；`commit` 自身若失败，其后置条件固定为：消费 batch、清除 unresolved 标记、释放并使全部 batch member handles 失效。worker 使用冻结的 membership/A 将这些 row 的 ledger 从 `Active(A)` 转为 `Free`，`fail_all` 不再对它们逐 handle abort。

## 5. 最终代码形态

### 5.1 Engine 内部组织

`src/runtime/engine` 最终组织为：

```text
engine.cpp
engine_core.h
request_record.h
scheduler.h
admission_policy.cpp
admission_policy.h
resource_manager.h
kv_capacity.cpp
kv_capacity.h
public_types.cpp
```

职责如下：

- `EngineCore<Instance>`：原 `ConcurrentExecutor` 的唯一继任者；持有 worker thread、队列同步、Scheduler、ResourceManager 和 request records；
- `RequestRecord<Package>`：一个已提交请求的 owning state，不拥有任何 device pointer、KV object 或 physical state tensor；
- `Scheduler`：拥有 FIFO/protected-head/backfill/one-prefill-owner/maximal-batch/service-work 规则；
- `ResourceManager<Package>`：拥有 logical lane occupancy/ledger、continuation catalog、candidate 选择、retain/claim/evict/release 策略；不拥有 Runtime physical epoch；
- `EngineCore` 的私有 worker 是 model lifecycle、Scheduler、ResourceManager/catalog、lane/sequence 和 Program execution mutation 的唯一线程 owner。response latch 与 cancellation request 是第 4.1/5.3 节明确列出的跨线程同步事实；`memory_summary`/`reset_memory_peaks` 等冷路径维护操作继续在 `execution_mutex` 下与 worker 串行化，但不获得 request/lane mutation authority。可以使用无状态 helper，不再引入第二个持有 queue、catalog 或 Program execution mutation authority 的 Coordinator。

现有 `admission_policy.*` 准确表达 protected-admission 数学，保留文件与测试名，由 Scheduler 调用。原 `concurrent_executor.h` 在切换后删除，不保留 type alias。以上是预期代码布局；私有 helper 是 nested 还是独立文件不单独构成验收 gate。

### 5.2 Package-neutral request types

公共调用顺序保持当前真实语义：`Engine::prepare()` 产生 opaque、move-only、语义不可变的 `PreparedPrompt`；对进入 executor 的非零输出 submission，`Engine::submit()` 基于 prompt 与 stop/output options 创建 target `OutputSession`，并由新建的 `RequestRecord` 独占该 mutable output state。零输出请求保持第 4.1 节的 `ImmediateSubmission` 旁路，不创建 executor RequestRecord/target OutputSession。当前代码不再引入另一个 Engine-facing `PreparedRequest` 聚合体。

Responses 的 `previous_response_id` 与 `ResponseStore` 继续由 Gateway 管理。当前 Engine 没有 protocol session，也不在本阶段新增 `SessionKey`、`CacheHints` 或跨请求协议状态。

`EngineCore` 只能通过 `Package` 引用 target-dependent Frontend/Runtime 类型。两个 exact package 为 Engine contract 导出以下闭合集合：

```cpp
using PreparedPrompt;
using OutputSession;
using PublishedOutput;

using RequestBasePlan;
using AdmissionPlan;
using SequenceHandle;
using ContinuationHandle;
using PendingBatch;
using StartResult;
using PrefillProgress;
using CommitResult;
using DiscardResult;
using FinishResult;
using AbortResult;
using ReleaseResult;
using Program;
```

`LaneId`、`RoundBudget`、`CommitDecision`、`RetentionDecision` 以及 Engine 使用的纯 value summaries 属于 `src/runtime/contract`。除 EngineCore 实际消费的类型外，不为抽象完整性增加 package alias。package identity 由 compile-time `Package` 保证，handle 不增加运行时 package tag/check。

必须删除 `EngineCore` 对以下具体 family 类型的直接引用：

```cpp
targets::qwen3_6::PreparedPrompt
targets::qwen3_6::OutputSession
targets::qwen3_6::PublishedOutput
```

也必须删除 `src/runtime/engine` 对 Qwen family frontend export header 的直接 include。family 共享类型仍可由两个 package alias 到同一实现，但 Engine 只依赖 package contract。

`serve::PreparedRequest` 属于 Gateway/serve 的 owning product value；其命名不影响本 contract，本阶段不重命名。

### 5.3 `RequestRecord<Package>`

最终 request record 表达下列 owning facts，不要求为每组事实机械创建包装类型：

| 分组 | 必要事实 | mutation owner |
|---|---|---|
| identity/options | request id、resolved options、deadline/submission time、prompt summary | submit 建立后不变 |
| Frontend state | owning prompt（仅 `WAITING`）、唯一 mutable `OutputSession` | prompt 由 worker start 消费；output 只在 worker transaction 预览/提交 |
| model state | 第 4.2 节的 `EngineRequestState` | 只有 worker transition helper |
| Runtime binding | optional base plan、active `SequenceHandle`、logical lane、generation budget、active entitlement | 只有 worker；存在性必须与第 4.2 节一致 |
| scheduling/accounting | service work、backfill class/epoch、begin/timing/speculative/output-token accounting | 只有 worker |
| response handshake | `response_done`、`consumer_released`、`capacity_released` | worker 设置 done，consumer 设置 released，共用 request mutex 完成 exactly-once capacity transition |
| publication/result | event queue、assembled output、result/error | worker 提交，consumer 只读/消费 |
| cancellation | atomic cancellation request | consumer/sink 可设置；只有 worker 使其影响 Runtime |

最终实现消除以下隐式状态组合：

- `lane.has_value()` 同时表示 admitted、active、complete 和 retained；
- `decode_ready` 同时承担 lifecycle；
- Runtime lane index 被当作 Engine sequence identity；
- per-request `lane_plans[kMaximumConcurrency]` 和 `lane_plan_versions` 承担资源管理策略。

### 5.4 Engine/Runtime compile-time contract

最终 Program contract 的调用方向和后置语义固定如下，不增加 virtual interface。result 可以是 aggregate 或带 accessor 的 owning value，但必须提供本节列明的信息：

```cpp
RequestBasePlan plan_request(
    const PreparedPrompt&,
    const ResolvedExecutionOptions&);

AdmissionPlan inspect_admission(
    const PreparedPrompt&,
    const RequestBasePlan&,
    LaneId destination,
    const ContinuationHandle* source);

StartResult start_request(
    AdmissionPlan&&,
    PreparedPrompt&&,
    std::optional<ContinuationHandle>&& source);

PrefillProgress advance_prefill(SequenceHandle);

PendingBatch decode(
    std::span<const SequenceHandle>,
    std::span<const RoundBudget>);

CommitResult commit(
    PendingBatch&&,
    std::span<const CommitDecision>);

DiscardResult abort_pending(PendingBatch&&) noexcept;

FinishResult finish(
    SequenceHandle,
    RetentionDecision) noexcept;

AbortResult abort(SequenceHandle) noexcept;

ReleaseResult release_continuation(ContinuationHandle&&) noexcept;

void fail_all_cleanup() noexcept;

AdmissionResources admission_capacity() const noexcept;
MemorySummary memory_summary() const noexcept;
void reset_memory_peaks() noexcept;
```

Contract 语义：

- `plan_request` 与 lane/candidate 无关、只读、无物理 mutation；
- `RequestBasePlan::summary()` 只暴露 Scheduler 和 permanent-feasibility 所需的 request-wide 信息；`AdmissionPlan::summary()` 暴露 candidate 的 active resources、reuse path/count 和 service-work quanta；
- `inspect_admission` 执行 exact prefix/checkpoint/backend verification；匹配时形成 reuse plan，不匹配时形成消费并替换该 source 的 full-reset plan，全程不 claim、restore、分配或驱逐；
- `AdmissionPlan` 是 Runtime-owned、opaque、move-only，并绑定 Program 产生的 destination/source physical epoch；ResourceManager 不复制该 epoch；
- `start_request` 是 probe 之后创建 active sequence 的唯一 mutation 点；continuation eviction/release 是另外的显式 Runtime mutation；
- full-reset 可以没有 source；复用或替换 resident state 时 source 必须与 probe 使用的 handle 一致；
- deadline/cancellation、catalog/source/eviction identity 在首次 mutation 前统一复核；choice 不跨 GPU unit 缓存，因此 Program physical epoch 在 start 内再失配属于 engine-wide invariant failure，不增加 `StalePlan` retry 分支；
- `start_request` 在 API 入口取得 moved plan/prompt/source 所有权；任何异常（包括 mutation 前的 physical epoch/invariant 失配）都消费/释放 source，清空 destination pending/KV/transient，并进入 engine-wide failure；不返回半所有权、不做 request-local retry；
- successful `start_request` 总是执行恰好一个 prefill/finalization GPU unit，返回新 `SequenceHandle`、实际 active resources 和首个 `PrefillProgress`；start/advance 每个 prefill unit 消耗一个 service-work quanta；
- `start_request` 返回后，worker 以局部 cleanup guard 暂时承担尚未接管 sequence 的清理义务：先核对实际 A，再通过预建 budget 与固定容量 ledger 无抛安装 handle/A 并执行第 4.2 节的 `WAITING → PREFILL`。核对或接管失败时 guard 调用 `abort`；只有 RequestRecord/ResourceManager 接管成功后才解除 guard，再解释返回的 `PrefillProgress`；
- `PrefillProgress` 提供 processed prompt-token count、complete 状态和 complete 时的 begin summary；只有在 prompt complete 并产生 licensed token 时携带单 row `PendingBatch`，该 token 与 decode candidate 共用同一 commit contract；
- `SequenceHandle` 是可复制、non-owning、generation-checked capability；RequestRecord 是 Engine 逻辑 owner，所有 Runtime mutation 使用 handle 而不是 raw lane；
- `ContinuationHandle` 是 opaque move-only unique capability；析构不调用 Program/CUDA，正常路径恰由 start 或 `release_continuation` 消费。非空 handle 不得被 move-assignment/reset 覆盖；实现可删除 move assignment 或强制目标为空；
- `PendingBatch` 固化 row 到 `SequenceHandle` 的 membership 并提供 commit 前有效的只读 row/token view；Program 是 mutation owner，`PendingBatch` 不持有可回调 Program 的反向指针。unresolved token 不得被覆盖或静默丢弃；
- `commit` 在函数入口取得 `PendingBatch` 所有权。worker 在调用前完成 row/decision shape 检查并解除 pre-commit scope guard；从进入 `commit` 起，无论成功或抛出都由 Program 消费 token，caller 不再调用 `abort_pending`。Program 内部捕获任何 validation/GPU/host-state 异常，清除 unresolved、释放全部 frozen members 后才重新抛出；
- successful terminal commit 只把 sequence 置为可 finish 状态，不隐式 retain、不释放为无主 lane；
- `CommitDecision` 的非取消 row 满足 `1 <= accepted <= produced`，非 terminal 时 `accepted == produced`；partial accepted prefix 只允许 terminal。snapshot-cancelled row 以 accepted 0 rollback provisional state，释放并使 sequence handle 失效，不再调用 `finish`；
- `CommitResult` 逐 row 返回 `Active`、`Finishable` 或 `CancelledReleased` disposition。`CancelledReleased` 同时返回 released-active acknowledgement，以及与 `AbortResult` 等价的最终 `GenerationTimings`/`SpeculativeStats`；`Finishable` 的 final stats 由紧随其后的 `finish` 返回；decode 的非取消 accepted token 数消耗对应 service work；
- `abort_pending` 在 commit 前失败路径消费整批 token，释放全部 members，并在 `DiscardResult` 中返回 row-aligned release acknowledgement；当 sequence 属于 unresolved batch 时不得调用普通 `abort`；
- `finish` 是 no-throw post-commit transition，返回本次请求的最终 timings/speculative stats，并在 `RetainResident` 时返回 continuation handle 与实际 resident resources；
- `finish(RetainResident)` 返回后，worker-local continuation guard 持有该 handle 的显式 release 义务；先核对 R，再无抛登记 fixed-capacity catalog 并提交 ledger transition，成功后解除 guard。核对/登记不变量失败时 guard 调用 `release_continuation`，若 capability acknowledgement 也失配则进入 `fail_all_cleanup`；
- `abort` 只用于不属于 unresolved batch 的 active sequence，返回取消结果所需统计与实际释放摘要，并清除 KV entitlement 和 transient activation；
- `release_continuation` 只由 worker 调用并返回实际释放的 resident resources；`fail_all_cleanup` 是 engine-wide failure 的显式清理，Program 对象析构只是最后的进程内资源兜底，二者都不是正常 release 路径；
- `DiscardResult`、`FinishResult`、`AbortResult` 和 `ReleaseResult` 均须区分 capability 已成功消费与 invariant mismatch；成功时 release acknowledgement 表示本次调用移除的 entitlement，失配时不得返回空摘要冒充成功，也不得 terminate。worker 将失配视为 `fail_all`，不继续 ledger transition；
- `fail_all_cleanup` 只替代当前 `fail_all` 中按 raw lane 全量 clear 的失败/停机能力：它 no-throw 清空残余 pending、active sequence、resident continuation、KV/transient 并使全部 opaque capabilities 失效，不执行 retention/admission policy，也不允许 Engine 清理后恢复服务；
- public `GenerationHandle` 的冷路径 type erasure 可以保留；禁止 virtual 的范围是 target execution hot path。

### 5.5 Common summaries

`src/runtime/contract/types.h` 保留真正跨 package 的小型 value contract：

```cpp
struct AdmissionResources {
    std::uint32_t active_lanes;
    std::uint32_t main_kv_pages;
    std::uint32_t backend_kv_pages;
};
```

`backend_kv_pages` 的解释由当前 Program 固定：

- ordinary：`0`；
- MTP：MTP KV entitlement；
- DFlash：DFlash full-context KV entitlement。

Scheduler 和 ResourceManager 不出现 `if (MTP)` 或 `if (DFlash)`。它们只处理三维闭合资源向量。

Engine-facing summaries 分为两个语义：

- `RequestBasePlan::summary()` 包含 prompt/requested/effective output token 数、limit reason、完整 request-lifetime active `AdmissionResources` 和 cold/full-reset service-work quanta，用于 permanent feasibility 与 protected-head 建立；
- `AdmissionPlan::summary()` 包含同一完整 active resources、candidate-specific reusable token count/`PrefixReusePath` 和 service-work quanta，用于 ResourceManager 选择与 backfill 安全性。

`transient_bytes` 和 `transient_alignment` 从两个 Engine-facing summary 删除，转为 Runtime 私有 plan 内容。common values 还可包含 resident continuation 实际占用、commit row disposition、资源 release acknowledgement 以及 Engine 结果所需 timings/speculative stats。Runtime physical epoch 只存在 opaque plan/handle 内，不进入 Engine common summary。

不得向 Engine 暴露 GDN tensor、block table、MTP bridge、DFlash frontier、checkpoint storage kind 或 target state pointer。

### 5.6 ResourceManager 与 continuation catalog

ResourceManager 内部维护：

```text
capacity
active entitlement ledger
resident continuation entitlement ledger
lane occupancy: Free | Active | Resident
ContinuationId → CatalogEntry
```

当前 `CatalogEntry` 只需要：

```cpp
struct CatalogEntry {
    ContinuationId id;
    Package::ContinuationHandle handle;
    LaneId resident_lane;
    AdmissionResources resident_resources;
};
```

当前最多八个 lane，catalog 使用启动固定容量、无 request-time allocation 且插入/删除不抛异常的存储。`ContinuationId` 是 ResourceManager 在 Engine 生命期内单调发放、不复用的逻辑 identity，不是 Program epoch，也不参与价值或 tie-break 排序。ResourceManager 直接枚举 resident handles，不复制 target prefix identity、rewrite-checkpoint 细节或尚未使用的 LRU/cost metadata；exact match 由 Runtime `inspect_admission` 完成。

Program 是 physical lane epoch 的唯一 authority，在 opaque plan/handles 中铸造并验证它。ResourceManager 只拥有逻辑 lane state；当前每个 lane 恰为：

```text
Free
Active(SequenceHandle, active_resources)
Resident(ContinuationId, ContinuationHandle, resident_resources)
```

`resident_resources.active_lanes` 恒为 `0`，且任何 worker boundary 均逐维满足：

```text
Σ active_resources + Σ resident_resources <= admission_capacity
```

下表以 ledger 中进入调用前记录的 `A`/`R` 为准。所有 `released_resources`/release acknowledgement 都表示“本次调用移除的 entitlement”，不是调用后的 residency：active release 等于 A，continuation release 等于 R。

允许的 ledger transition 及顺序固定为：

| 物理动作 | 逻辑转移 | 核对 |
|---|---|---|
| selected source claim + successful `start_request` | worker guard 核对并无抛接管后 `Free/Resident → Active` | `StartResult.active_resources == AdmissionPlan.summary().active_resources` |
| `finish(RetainResident)` | Runtime 先返回 handle/R，catalog 无抛登记，再 `Active(A) → Resident(R)` | `R.active_lanes == 0`，且 `R.main/backend <= A.main/backend` |
| `finish(Release)` | `Active(A) → Free` | 无 resident handle/resources，released-active acknowledgement 等于 A |
| ordinary `abort` | `Active(A) → Free` | released-active acknowledgement 等于 A，Runtime active state/KV/transient 已释放 |
| successful `abort_pending` | 每个 frozen member `Active(A) → Free` | row-aligned acknowledgement 各自等于 A，batch token 已消费 |
| cancelled `CommitResult` row | `Active(A) → Free` | disposition 为 `CancelledReleased`，released-active acknowledgement 等于 A，不再 finish |
| `commit` 抛出 | 每个 frozen member `Active(A) → Free` 后进入 `fail_all` | exception 后置条件保证 token 已消费、各 member 已物理释放且 handle 失效 |
| `release_continuation`/eviction | Runtime 释放成功后 `Resident → Free` | release summary 等于 catalog R |
| `fail_all_cleanup` | 清空剩余 `Active`/`Resident` ledger 与 catalog | 只在 Engine 终止服务的失败/停机路径执行，不生成业务资源 delta |

choice 执行时，worker 先核对 deadline/cancellation 与 catalog identities，建立 generation budget 并预留已知上限的 token-accounting host capacity；这些无副作用步骤失败时只结束当前 request。之后 worker 按选定顺序释放 evictions，将 source handle 移出 catalog 到带显式 release 义务的局部 guard，并立即调用 start；只有进入 `start_request`、由 Program 接管 source 后才解除该 guard。这些 mutation 之间没有可能抛出的业务步骤，也没有 scheduling/GPU boundary；start 返回后由第 5.4 节的 active-sequence guard 核对并一次提交 `Free/Resident → Active` 与 RequestRecord 接管，然后才处理首个 progress。start mutation 后失败时 Program 消费/释放 source 并清理 destination，worker 直接 `fail_all`，不回滚为可继续的业务状态。

Logical ledger 是 policy authority，Runtime physical pools 是 realization authority。Runtime 返回摘要违反上述方程时，worker 进入 engine-wide failure；不修正计数后继续运行。production 使用的 candidate-selection 和 ledger-transition value logic 保持不依赖 fake Package/Program 的直接可测 seam。

Scheduler 的 protected-head/backfill 数学只使用 active request entitlements。resident continuation 的资源虽然由 ResourceManager ledger 真实记录，但它们可在 admission boundary 被驱逐，因此不得被伪装成不可回收 active work，也不得改变 donor frontier。

### 5.7 当前 candidate 选择算法

本阶段不引入未经测量的通用 cost model。对 Scheduler 当前允许考虑的每一个 request，ResourceManager 按下列纯 value 算法保持现有行为：

1. 只枚举逻辑状态不是 `Active` 的 destination lane，即 `Free` 或 `Resident`；
2. 对每个 lane 的 resident continuation（若有）调用 exact `inspect_admission`；exact mismatch 仍是消费并替换该 source 的 full-reset candidate，空 lane 使用无 source 的 full-reset probe；
3. 对每个 destination 纯计算 `used_active + used_resident - source.R - Σeviction.R + candidate.A <= capacity`；source-replacing full reset 与 reuse 都可回收该 source 的 resident resources；
4. 第一轮令 eviction 集合为空，只考虑无需驱逐非 source continuation 就可满足完整 entitlement 的 candidate；
5. 第一轮中选择 `reused_prompt_tokens` 最大者，平局按 destination `LaneId` 升序；当前一 lane 最多一个 continuation，不引入 `ContinuationId` 排序语义；
6. 若第一轮无解，对每个 candidate 按 resident `LaneId` 升序取使不等式成立的最短 non-source eviction prefix；所有 candidate 算完后仍按最大 reuse、destination lane 升序选择；
7. selected choice 固化 opaque plan、source catalog identity 和 eviction catalog identities；比较和 Scheduler 决策期间不做任何 Runtime/catalog mutation；
8. Scheduler 接受该 choice 后，worker 在同一 boundary 复核 request control 与 catalog identities 并立即执行；没有正常 stale/retry 分支，identity/physical epoch 在该点失配属于内部不变量错误。

缓存命中不能形成第二套请求优先级。Scheduler 决定先考察哪些 FIFO/protected-head/backfill request；ResourceManager 只为当前 request 生成 read-only choice。Scheduler 只看 candidate summary，handles 和 eviction list 对 ResourceManager/worker 私有；被 Scheduler 拒绝的 choice 无副作用并立即销毁。protected head 用 base/cold active resources 建立，backfill safety 与 temporal credit 使用 selected candidate 的 service work，保持现有语义。

Admission 必须使用 Runtime `RequestBasePlan` 给出的完整 request-lifetime peak entitlement：

```text
reused prefix + remaining suffix
+ 最大有效输出所需的最终 context peak
+ selected backend 的 provisional growth margin
```

具体 token/page 边界和 off-by-one 由 Runtime plan 统一计算；ResourceManager 不按 suffix 重算 entitlement，只计算 source/eviction replacement delta。不得只为下一个 prefill chunk 或下一轮 decode 预留。

### 5.8 Scheduler 与 worker 顺序

重构必须保持现有 worker boundary 顺序，而不是照搬理论伪代码后改变调度：

```text
1. 处理 pending expiry 与 boundary cancellation
2. 构造当前 maximal decode membership
3. 若存在 prefill owner：在 decode 与一个 prefill unit 间按既有 alternation 执行
4. 若无 prefill owner：依据上一 GPU unit 是否为 decode 决定 admission opportunity
5. 若未 admission：执行 maximal compact decode batch
6. 无工作时等待
```

保留以下规则：

- one-prefill-owner；
- protected-head 与 persistent/temporal backfill；
- frozen donor frontier 与 temporal credit；
- decode batch membership 在调用 Runtime 前一次性冻结；
- successful admission start 执行恰好一个 prefill/finalization GPU unit；
- 本阶段没有 `ResourceMove` action，因为没有 host/device transfer 能力。

Scheduler 可以为它按公平性规则考察的每个 request 调用 ResourceManager probe；ResourceManager 不得扫描、删除或重排 pending queue。

### 5.9 Output transaction

prefill licensed token 与 decode candidate 共用同一提交原则：

```text
Runtime candidate/PendingBatch
→ 冻结 row-aligned cancellation snapshot
→ 各 RequestRecord 的 OutputSession staged preview
→ Runtime commit 精确 accepted prefix
→ 处理 CommitResult 的 Active/Finishable/CancelledReleased ledger transition
→ 对全部 successful terminal row 执行 ResourceManager retention decision + Runtime finish + catalog/ledger transition
→ Engine generation budget、token accounting、metrics
→ OutputSession commit_preview
→ 将 committed delta 放入 event queue
→ 唤醒 response consumer
```

强制约束：

- `OutputSession::preview()` 只产生未对外可见的 staged state；它不新增独立 rollback API，失败时随请求错误收尾/析构而丢弃；
- preview/decision 阶段由 scope guard 持有 unresolved `PendingBatch`；任一 pre-commit preview/Engine invariant 异常由 guard 调用 `abort_pending`，再进入 `fail_all`。worker 完成 decision shape 验证后在 `commit` 入口转移 token 并解除 guard；`commit` 返回或抛出后都不得再次调用 `abort_pending`；
- Runtime commit、全部 finish 和 ResourceManager ledger/catalog transition 成功前不得 `commit_preview()` 或发布任何 row event；
- commit 失败后不发布本轮任何 row；worker 依据冻结 membership/A 完成第 5.6 节的失败 ledger 转移并进入 `fail_all_cleanup`。`finish` 和 fixed-capacity catalog registration 是 no-throw，ledger/capability mismatch 同样在发布前终止 Engine；
- cancellation snapshot 后到达的取消不改变已冻结 row decision：若该 decision 已 terminal，本轮以原 finish reason 完成；若仍 nonterminal，则下一 boundary 执行 Cancelled abort；
- snapshot 中已取消的 row 接受 0 token，Runtime commit rollback provisional state 并释放 sequence，ResourceManager 执行 `Active → Free`，不调用 finish；
- waiting、active-boundary 和 snapshot 三类产品取消都恰好一次 staged `preview_terminal(Cancelled)`，以保留 `OutputSession` 的 decoder flush 语义；有 Runtime state 时只有 abort/commit 与 ledger release 成功后才能 `commit_preview()`/发布，engine-wide failure 则丢弃 staged cancellation preview 并完成 error；
- successful terminal row 的 continuation 决策在 `finish` 前由 ResourceManager 选定，Runtime 不做第二次 policy decision；
- Engine 完成 generation budget/result 后才将 delta 交给 Gateway。

上述原子边界到 Runtime/ResourceManager state 成功为止。一旦全部模型状态已成功提交，event queue/stream 继续沿用现有逐请求发布语义；不为极端 host allocation/sink 失败新增跨请求 batch-wide event transaction。sink/consumer 异常仍只设置该 request cancellation，不从 consumer thread 调用 Runtime。

### 5.10 Request transient

当前 `src/runtime/engine/request_memory.*` 是启动时冻结的物理 device arena，却由 Engine 激活并把 `TransientRegion` 传入 Program。这违反物理资源归 Runtime 的边界。

最终形态：

- 将机制移动并重命名为 `src/core/request_transient_arena.h/.cpp`；
- `RequestTransientArena` 继续只在 startup 分配一次，保持 256-byte device allocation alignment、active prefix、peak accounting 和固定地址；
- `SequencePlan` 继续给出 target-computed frozen capacity；
- `ProgramImplCore` 在构造时从 `SequencePlan` 消费该 capacity 并拥有 arena；
- Runtime 在 `start_request`/prefill 内根据 private plan 激活，在 prompt complete、abort 和 failure 时关闭；
- `EngineCore` 不 include `transient_region.h`，不调用 activate/deactivate，不获得 device pointer；
- `Program::memory_summary()` 和 `reset_memory_peaks()` 直接汇总/reset request transient；
- `Qwen3_6_27BInstance` 与 `Qwen3_6_35BA3BInstance` 删除 `RequestMemory` 字段；
- Engine-facing `RequestBasePlan`/`AdmissionPlan` summaries 删除 transient size/alignment。

不改变“不存在 NInfer-owned request-time device allocation path”这一性能与地址稳定性约束；不对 CUDA driver/library 内部实现设置无法由本代码证明的 `cudaMalloc/cudaFree` gate。

## 6. 分阶段代码切换

下列阶段是可构建的验收点，不约束工作分支内的局部编辑顺序或 commit 历史。每个阶段出口和最终 diff 不得保留新旧 contract 并行使用。

### 阶段 0：建立基线

代码修改前记录：

- 当前受影响的 focused tests 与三个既有 real-engine tests 结果；
- 一个明确可用的 27B groupwise MTP3 decode-saturation 基线，只取 C=1 与 C=8；
- `MemorySummary::request_transient`、active/request stats 和 prefix reuse 路径；
- 当前能够通过公共 API 确定性复现的 handle abandonment/cancellation 行为；不存在稳定复现方式的竞态不为建立基线而新增 hook 或 sleep test。

基线只用于本次回归判断，不生成长期报告或新维护文档。

### 阶段 1：解除 Engine 对 family frontend 类型的硬绑定

修改范围：

- `src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h`
- `src/targets/qwen3_6_35b_a3b/export/ninfer/targets/qwen3_6_35b_a3b/package.h`
- `src/runtime/engine/concurrent_executor.h`
- `src/runtime/engine/engine.cpp`

动作：

1. 两个 package 补齐 `PublishedOutput` alias；
2. executor 的 prompt/output/published-output 全部改为 `Package::*`；
3. 删除 executor 对 Qwen family frontend header 的直接 include；
4. 保持公共 `PreparedPrompt` PIMPL 与 `GenerationHandle` type erasure 行为不变。

阶段出口：

- Engine execution 模板不出现 `targets::qwen3_6::{PreparedPrompt,OutputSession,PublishedOutput}`；
- 全量编译通过；serving/Gateway 类型、schema、response store、streaming 和 request capacity 没有改动。

### 阶段 2：显式化 RequestRecord 与 Scheduler

修改范围：

- 新增 `src/runtime/engine/request_record.h`
- 新增 `src/runtime/engine/scheduler.h`
- 保留 `src/runtime/engine/admission_policy.h/.cpp`
- `concurrent_executor.h` 重命名为 `engine_core.h`
- `src/runtime/engine/engine.cpp`
- `src/CMakeLists.txt`
- 保留 `tests/test_admission_policy.cpp`
- `tests/CMakeLists.txt`

动作：

1. 将 nested `Request` 提升为 `RequestRecord<Package>`，按第 5.3 节拆分 state；
2. 引入唯一 `EngineRequestState`，移除由 `decode_ready`/lane presence 推断 lifecycle 的写法；
3. 将 FIFO snapshot、protected-head、backfill、service-work、decode membership 构造放入 Scheduler；
4. Scheduler 只返回 decision/view，不调用 Program、不修改 KV；
5. `EngineCore` 保持唯一 worker、队列锁、execution lock、stats publication 和 response wait；
6. 保持第 5.8 节的现有 loop 次序，不加入 `ResourceMove`；
7. 保留直接保护 protected-head、donor frontier、persistent/temporal backfill 和三维 capacity 语义的断言；删除只绑定旧 nested type/source shape 的断言。

阶段 2 先显式化 model/response state，active Runtime binding 仍单路使用当前 raw-lane Program contract；不在这一阶段预先引入半套 handle API。阶段 4 在同一出口将该 binding 替换为第 5.3/5.4 节的 `SequenceHandle` contract，最终 diff 不保留 raw-lane 路径。

阶段出口：

- request lifecycle 的每次改变都由显式 transition 完成并验证前置状态；
- Scheduler 无 target include、无 Runtime mutation；
- greedy fixtures 的 token/output 满足既有精确条件，`RuntimeStats` 计数语义不变；不比较 wall-clock timing 或 stochastic token 序列；
- 原 `ConcurrentExecutor` 名称和兼容 alias 已删除；`admission_policy.*` 由 Scheduler 继续使用。

### 阶段 3：把 request transient 归还 Runtime

修改范围：

- 新增 `src/core/request_transient_arena.h/.cpp`
- 删除 `src/runtime/engine/request_memory.h/.cpp`
- 删除 `src/runtime/contract/transient_region.h`
- `src/targets/registry.h/.cpp`
- `src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h`
- `src/targets/qwen3_6_35b_a3b/export/ninfer/targets/qwen3_6_35b_a3b/package.h`
- `src/targets/qwen3_6/impl/runtime/layouts.h`
- `src/targets/qwen3_6/impl/runtime/program.h`
- `src/targets/qwen3_6/impl/runtime/program_impl.h`
- `src/targets/qwen3_6/impl/runtime/request_plan_impl.h`
- `src/targets/qwen3_6/impl/runtime/api_impl.h`
- `src/targets/qwen3_6/impl/runtime/vision_context.h`
- `src/targets/qwen3_6/impl/runtime/vision_context_impl.h`
- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h`
- `src/runtime/contract/types.h`
- `src/runtime/engine/engine_core.h`
- `src/CMakeLists.txt`、`tests/CMakeLists.txt`
- `tests/test_request_memory.cpp` 重命名为 `tests/test_request_transient_arena.cpp`

动作：

1. 保留并移动 frozen arena 机制，不重新实现为 request-time allocator；raw region view 收敛为 core primitive 的私有/nested view，不再是 Runtime common contract；
2. Program 构造时拥有 arena；Engine/Instance 不再拥有；
3. private request plan 保存 transient extent，Program 自己 activate/deactivate；
4. 删除 `start_prefill_lane(..., TransientRegion)` 参数；
5. 所有 success/abort/exception 路径在 Runtime 内关闭 active extent；
6. memory summary 与 peak reset 改由 Program 汇总；
7. 机制测试继续验证固定地址、capacity rejection、alignment、active/peak/reset；测试名随 ownership 更新。

阶段出口：

- `src/runtime/engine`、exact package export 与 `src/targets/registry.*` 不出现 `RequestMemory`、`TransientRegion` 或 device arena；
- Vision real test 的 transient peak 与 baseline 一致；
- DFlash 的零 request-transient 配置仍报告零；
- 代码和 frozen-arena 机制中不存在 NInfer-owned request-time device allocation path。

### 阶段 4：一次性切换 ResourceManager 与 Runtime transaction contract

这是本方案的核心切换。阶段出口和最终 diff 必须同时切换 ResourceManager 与 Runtime transaction contract，不得留下 Engine 继续调用旧 lane policy API 的产品路径；局部编辑顺序不构成额外 gate。

修改范围：

- 新增 `src/runtime/engine/resource_manager.h`
- `src/runtime/engine/request_record.h`
- `src/runtime/engine/scheduler.h`
- `src/runtime/engine/engine_core.h`
- `src/runtime/contract/types.h`
- `src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h`
- `src/targets/qwen3_6_35b_a3b/export/ninfer/targets/qwen3_6_35b_a3b/package.h`
- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h`
- `src/targets/qwen3_6/impl/runtime/program.h`
- `src/targets/qwen3_6/impl/runtime/program_impl.h`
- `src/targets/qwen3_6/impl/runtime/request_plan_impl.h`
- `src/targets/qwen3_6/impl/runtime/api_impl.h`
- `src/targets/qwen3_6/impl/runtime/instance.h` 及必要实例化文件
- 新增 `tests/test_resource_manager.cpp`
- focused unit tests 与 `tests/CMakeLists.txt`

内部实施顺序：

1. 定义 `LaneId`、可复制 generation-checked `SequenceHandle`、move-only 且无析构 mutation 的 `ContinuationHandle`；
2. 将 `RequestPlan` 重命名/收敛为 `AdmissionPlan`，拆清 base plan 与 candidate-specific plan；
3. 实现 `inspect_admission` 的只读 exact probe，由 Program 在 opaque plan/handles 中绑定 physical epoch；
4. 实现 `PendingBatch`、入口即接管 token 的 `Program::commit`、`Program::abort_pending` 和 failure-only `fail_all_cleanup`，让 prefill licensed output 与 decode 共用同一 transaction；
5. 把 terminal commit 与 retain 分开，实现带 consume acknowledgement 的 no-throw `finish(RetainResident|Release)`/`abort`/`release_continuation`；
6. 实现 ResourceManager 的 production value candidate/ledger logic、固定容量 continuation catalog 和第 5.6 节转移方程，并建立第 8.1 节的小型直接测试；
7. 将 Engine admission 替换为 Scheduler-request → ResourceManager-choice → Program-start；
8. 将 batch execution 替换为 `SequenceHandle` membership 与 `Program::commit(PendingBatch&&, ...)`；
9. 按第 5.9 节重排 output commit/publication；
10. 删除全部旧 lane policy/mutation API 和 Engine per-lane plan cache。

本阶段必须删除：

```text
Program::plan_request_for_lane
Program::can_admit_lane
Program::can_admit_lane_after_retained_eviction
Program::start_prefill_lane
Program::advance_prefill_lane
Program::resolve_prefill_lane
Program::decode_batch(raw lane span)
Program::resolve_pending_batch(raw lane span, ...)
Program::abort_lane
Program::has_retained_lane
Program::evict_retained_lane
Program::generation_timings_lane
Program::speculative_stats_lane
Engine per-request lane_plans
Engine lane_plan_versions cache
SequenceState::retained 这一混合语义布尔值
```

Runtime 内部可以继续以 fixed lane index 访问预分配 tensor/graph row，但 lane index 不再是 Engine/Runtime mutation contract。

阶段出口：

- Engine 是唯一 candidate/eviction/retention policy owner；
- Program 是唯一 sequence/continuation/KV physical owner；
- terminal commit 后必须显式 finish，且不存在无 handle 的 retained state；
- PendingBatch 在 commit/abort_pending 和 commit-failure 三条路径上都有闭合后置条件；
- catalog logical resources 与 Program physical summaries 满足第 5.6 节的所有转移方程；
- ordinary、MTP、DFlash 共用同一 Engine resource/transaction 分支；
- 全量构建与当时完整的 focused tests（包括 `ninfer_resource_manager_test`）通过；三个代表性 real routes 统一放在阶段 5 验证，不重复设 gate。

### 阶段 5：代表性验证与旧路径清理

修改范围限于受本次重构影响的代码、CMake 和测试，不修改长期文档。

动作：

1. 保留三个既有代表性 real tests：27B prefix/MTP/Vision、35B-A3B MTP/Vision、35B-A3B DFlash；不扩展为 artifact identity × feature 的全组合；
2. 运行阶段 4 已建立的 ResourceManager production-value test；它只覆盖无驱逐优先、最大 reuse/lane tie-break、最短必要驱逐与 Free/Active/Resident 转移，不测正常路径不存在的 stale physical epoch，不引入 fake Package/Program、test-only getter 或通用执行接口；
3. 向一个既有 real test 增加公共 API 场景：配置 `max_concurrency=1`、`max_pending_requests=1`，提交后直接销毁未 wait handle；随后一次同步 `generate()` 完成，确保 worker 已越过该 FIFO 请求；再连续成功持有两个 submission handles，并断言第三个并存 submit 同步返回 `RequestErrorKind::Overloaded`，最后 wait 前两者。`max_outstanding=2` 下，两个成功 handles 证明 abandoned permit 最终归还，第三个被拒绝证明没有重复归还/计数过低；不使用 sleep、竞态 hook 或精确 GPU 到达断言；
4. 使用第 9.4 节的固定五个短 CLI message smokes 实际加载本机已有的五个 artifact；这是每个 identity 一次公共 CLI→Engine 产品闭环，不扩展为 identity × feature 矩阵，不新增 CTest target；
5. serving/CLI/benchmark 继续只调用公共 Engine；未修改 wire translation 时，schema golden 不是 completion gate，调试期间可按实际现象运行；
6. 删除阶段 4 列明的 superseded symbols、直接 include/CMake source 与相关 stale 注释；不做无关历史注释清理；
7. 使用一次性 `rg`/code review 完成 cutover 审查，不把 source-string scan 注册为永久测试；
8. 执行第 8、9 节收敛后的 correctness、real-route、CLI product smoke 和代表性性能诊断。

阶段出口即本执行方案的代码完成条件。

## 7. 行为与失败语义

### 7.1 Admission probe 与 mutation failure

- probe 期间的 exact prefix/checkpoint mismatch 不是异常：该 resident lane 产生 source-replacing full-reset candidate，ResourceManager 仍可继续比较其他 candidate；
- `plan_request` 发现 represented input/context/permanent feasibility 错误时返回现有 `ContextLengthExceeded`/请求错误，只结束当前 request；
- deadline/cancellation 在首次 mutation 前到达时是当前 request 的正常结束，choice 无副作用地丢弃；
- generation budget/token-accounting host capacity 在 catalog/Runtime mutation 前建立；该无副作用准备失败只结束当前 request，不先驱逐 continuation；
- choice/catalog identity 或 Program physical epoch 在 single-owner immediate-start boundary 失配、probe 内部不变量失败、physical ledger/KV summary mismatch，均说明内部不变量已破坏，进入 `fail_all`，不当作可预期 stale 重试；
- start 开始 physical mutation 后的异常、GPU unit 异常、unresolved batch 期间的 OutputSession invariant failure、commit 失败和 ResourceManager transition mismatch 均进入 `fail_all`；各 Runtime API 履行第 5.4 节的 cleanup 后置条件；
- 任一 no-throw consuming API 返回 capability 未消费/不匹配时，不把空资源摘要当作成功，也不尝试以同一 handle 重试；worker 丢弃尚未发布的 preview，调用 `fail_all_cleanup` 并终止 Engine 服务；
- 不增加“捕获异常后 full reset 再试”的隐藏恢复路径，因为它会掩盖 ownership 错误并产生双重语义。

### 7.2 Cancellation

| 到达位置 | 行为 |
|---|---|
| pending queue | Scheduler boundary 删除，执行一次 `preview_terminal(Cancelled)` 并提交该 preview，完成 Cancelled，不接触 Runtime |
| active 且 GPU unit 未开始 | worker staged `preview_terminal(Cancelled)`，Runtime `abort` 与 `Active → Free` 成功后才提交 preview 并完成 Cancelled |
| decode 已返回 candidate、尚未 commit | 以本轮冻结 snapshot 为准；已取消 row 先 staged `preview_terminal(Cancelled)`，再 commit 0；Runtime 释放 sequence，ResourceManager `Active → Free` 后才提交 preview，其他 row 正常 commit |
| 本轮 snapshot 后才取消 | 不改变冻结 decision；该 row 若已 terminal，按原 finish reason 完成且不改写为 Cancelled；若仍 nonterminal，下一 boundary 执行上述 active cancellation |
| consumer/sink 抛错 | 设置 cancellation；不从 consumer thread 调 Runtime |
| Engine shutdown/failure | 若 worker 持有 unresolved batch 则先 `abort_pending`；随后调用一次 Program `fail_all_cleanup`，清空 ResourceManager ledger/catalog，最后完成 request error |

三类正常取消都沿用现有 `OutputSession::preview_terminal(Cancelled)` 的可观察输出语义，但不会在 Runtime/resource transition 失败时把 Cancelled preview 误发布成成功结果。任何取消路径都不得留下属于该请求/当次 transaction 的 active entitlement、无 owner resident handle、pending batch 或 active request transient；与该取消无关的既有 catalog entries 保持，engine-wide failure 则释放全部 catalog。

### 7.3 Retention

为保持当前行为，本阶段策略固定为：

- successful terminal：请求 `RetainResident`，登记 catalog；
- cancelled/error/abort：`Release`；
- admission 所需资源不足：ResourceManager 按 resident `LaneId` 升序驱逐第 5.7 节选定的最短 entry prefix；
- selected resident continuation 被 claim 后从 catalog 移除，成功 finish 后可生成一个新的 continuation entry；
- full reset 替换某 resident source 时，旧 handle 被消费，不产生两个 owner。

本阶段不做基于 session、重建成本、host transfer cost 或动态热度的价值排序。

## 8. 验证选择与测试范围

验证按本次变更的语义边界选择，不按产品 feature 列表做笛卡尔积。架构 ownership 由编译依赖和一次性 cutover review 证明；调度与资源 policy 由小型确定性测试证明；模型状态 transaction 由已有真实 Program routes 证明。任何新增测试都不得要求 test-only runtime abstraction。

### 8.1 Focused tests

| 测试 | 本次保留的证据 | 明确不扩展的内容 |
|---|---|---|
| `ninfer_public_api_test` | 公共 headers 独立编译、公共 move-only 类型和 Engine API 不变 | 不测试内部 Sequence/Continuation handles |
| `ninfer_admission_policy_test` | 既有 protected-head、donor frontier、persistent/temporal backfill、三维 capacity 数学 | 不按 C=1/2/4/8 重复同一纯策略 |
| `ninfer_resource_manager_test` | production value logic 的无驱逐优先、最大 reuse/lane tie-break、最短必要驱逐、Free/Active/Resident 转移方程 | 不 mock Program，不测 physical stale、getter/class/file 形状或状态穷举 |
| `ninfer_request_transient_arena_test` | startup-frozen 地址、alignment、capacity、active/peak/reset | 不增加 request-time allocator/profile 测试 |
| `ninfer_qwen3_6_frontend_test` | 既有 `PreparedPrompt`、`OutputSession` preview/commit、Vision/MRoPE/prefix identity | 不因 Engine 重构扩展 Frontend 功能矩阵 |
| `ninfer_qwen3_6_runtime_mechanisms_test` | 既有 topology/layout、MTP alignment、Vision control 与 prefix identity 检查 | 不宣称它覆盖 Program state transaction、rewrite checkpoint 或 ReplaySSM，不搭建 fake Program |

`ninfer_resource_manager_test` 是本次新增 production candidate/ledger logic 的直接证据，不是视实现方便与否决定的可选项。如果它只能通过 fake package、mock handle owner 或 test-only API 构造，说明 policy/ledger 仍与 Program 物理实现纠缠，阶段 4 的 ownership 还未闭合；解法是收敛 production value seam，而不是增加通用测试框架。

### 8.2 代表性 real Engine routes

本次只要求三个已经存在且直接覆盖变更边界的 real tests：

| Route | 覆盖原因 |
|---|---|
| `ninfer_qwen3_6_27b_prefix_real_test` | 第一个 execution package；MTP、Vision request transient、append/zero-suffix/rewrite prefix |
| `ninfer_qwen3_6_35b_a3b_real_test` | 第二个 execution package；MTP、Vision、partial terminal、最大配置 memory summary |
| `ninfer_qwen3_6_35b_a3b_dflash_real_test` | 唯一 DFlash route；partial speculative commit、context frontier 与 C=2 公共并发结果 |

这些 routes 已覆盖两个 compile-time package、ordinary/MTP/DFlash transaction、Vision transient 和 continuation reuse。此次不修改 artifact binder、target leaf math 或 quantized kernel，因此不要求五个 identity 分别重复完整 feature suite。

- 三个 real tests 使用明确的 Qwen3.6 groupwise artifact，验证两个 execution packages 和受影响的模型状态 contract；NVFP4 与 Qwen3.8 identities 的公共产品加载/生成改由第 9.4 节的固定 CLI smokes 覆盖，不重复 real feature suite；
- 若实现意外修改 registry、binder 或 target leaf，则该修改已扩大风险面，届时只补跑实际受影响 identity 的 load/real test，而不是预先扩大当前矩阵；
- 本阶段已确认第 9.3/9.4 节列明的五个 artifact 均在本机存在，因此这些固定验证不是条件 gate；不下载或重新转换 artifact。

### 8.3 Concurrency 与 lifecycle

- Scheduler 公平性只由 `ninfer_admission_policy_test` 验证一次，不在每个 concurrency 重复；
- 既有 DFlash C=2 route 验证两个并发请求的公开结果，不把调度敏感的 `decode_row_rounds > decode_rounds` 增加为必选断言，也不宣称它单独证明 compact batching；
- 代表性 C=8 运行只验证公共 Engine 在最大支持 concurrency 配置下完成指定 workload，不把性能 campaign 宣称为数组边界证明；
- 在一个既有 real test 中增加阶段 5 已定义的确定性 handle-abandonment capacity 回归：先用同步请求越过 abandoned FIFO entry，再同时持有两个后续 submissions，并确认第三个按容量被拒绝；
- decode in-flight cancellation 的精确到达时刻、response 已完成但 consumer 尚未释放的内部瞬间、以及注入 Runtime invariant failure，不作为本阶段强制测试；这些是 transaction/ownership review 项，除非实现过程中出现可稳定复现的真实 bug；
- MTP/DFlash partial-terminal real tests 直接保护 accepted prefix、公开结果与下一次 retained frontier 一致；“Runtime commit/finish 成功前不得 publish”由第 5.9 节的控制流与 ownership review 核对，不为异常注入单独构造 fake Runtime。

### 8.4 不形成 completion gate 或永久 target 的扩张

- 不新增 artifact identity × ordinary/MTP/Vision/prefix/DFlash 全组合；
- 不新增 C=1/2/4/8 × cancellation/timeout/queue 状态全组合；
- 不为内部类名、文件名、getter、deleted compatibility 或 source string 注册测试；
- 不引入 generic fake model/package 来驱动 `EngineCore`；
- wire translation 未变化时，完整 OpenAI/Responses/Anthropic schema regression 不是 completion gate；调试实际 schema 现象时可运行相关既有测试；
- 不为满足固定覆盖率或测试数量而拆 production API。

## 9. 验证命令与性能证据

### 9.1 构建与一次性 cutover 检查

```bash
cmake --build build -j
git diff --check
```

代码完成后执行一次定向搜索，辅助确认 superseded names 已被删除：

```bash
rg -n 'plan_request_for_lane|can_admit_lane|can_admit_lane_after_retained_eviction|start_prefill_lane|advance_prefill_lane|resolve_prefill_lane|decode_batch\(|resolve_pending_batch|abort_lane|has_retained_lane|evict_retained_lane|generation_timings_lane|speculative_stats_lane|RequestMemory|TransientRegion' src tests
```

该搜索是 cutover review，不注册为 test，也不替代编译、行为测试或 ownership code review。预期旧接口为零命中；普通文字误命中按上下文处理，不为追求机械零输出改写无关代码。

### 9.2 Focused tests

```bash
ctest --test-dir build --output-on-failure \
  -R 'ninfer_(public_api|admission_policy|resource_manager|request_transient_arena|qwen3_6_frontend|qwen3_6_runtime_mechanisms)_test'
```

这些 target 中只有 `ninfer_resource_manager_test` 是本次因新 production value logic 而增加的直接回归；它不导出 test-only interface。

### 9.3 Real artifact tests

使用明确路径，不以 glob 或修改时间选择 artifact：

```bash
NINFER_QWEN3_6_27B_WEIGHTS="$PWD/out/qwen3_6_27b.ninfer" \
ctest --test-dir build --output-on-failure \
  -R 'ninfer_qwen3_6_27b_prefix_real_test'

NINFER_QWEN3_6_35B_A3B_WEIGHTS="$PWD/out/qwen3_6_35b_a3b.ninfer" \
ctest --test-dir build --output-on-failure \
  -R 'ninfer_qwen3_6_35b_a3b_(real|dflash_real)_test'
```

使用 `$PWD` 是因为 CTest 在 `build/tests` 作为 working directory 运行；相对 `out/...` 会被错误解析为 `build/tests/out/...`。NVFP4 和 Qwen3.8 不重复这三个 full real tests，由下一节的公共 CLI smokes 验证注册 identity 与生成路径。

### 9.4 CLI 产品闭环与 serving 表面

从仓库根目录运行下列固定 smokes。五个已有 artifact 各加载一次；Qwen3.6-27B groupwise 使用 `examples` 中的 image+video message 经 Vision+MTP 路径生成精确 `NIFER-9`，35B-A3B 使用 text message 经 DFlash 生成精确 `42`，其余三个 identity 用同一短 text oracle：

```bash
set -eu
CLI="$PWD/build/apps/ninfer"

run_exact() {
  expected="$1"
  shift
  actual=$("$@")
  if [ "$actual" != "$expected" ]; then
    printf 'CLI smoke mismatch: expected <%s>, got <%s>\n' "$expected" "$actual" >&2
    return 1
  fi
}

run_exact 'NIFER-9' "$CLI" "$PWD/out/qwen3_6_27b.ninfer" \
  --messages examples/cli/messages/mixed_image_video.json \
  --max-context 8192 --max-new 32 --no-thinking --greedy --vision \
  --spec mtp --draft-tokens 3 --lm-head-draft

run_exact '42' "$CLI" "$PWD/out/qwen3_6_27b_nvfp4.ninfer" \
  --messages examples/cli/messages/text_smoke_zh.json \
  --max-context 2048 --max-new 8 --no-thinking --greedy

run_exact '42' "$CLI" "$PWD/out/qwen3_8_27b.ninfer" \
  --messages examples/cli/messages/text_smoke_zh.json \
  --max-context 2048 --max-new 8 --no-thinking --greedy

run_exact '42' "$CLI" "$PWD/out/qwen3_8_27b_nvfp4.ninfer" \
  --messages examples/cli/messages/text_smoke_zh.json \
  --max-context 2048 --max-new 8 --no-thinking --greedy

run_exact '42' "$CLI" "$PWD/out/qwen3_6_35b_a3b.ninfer" \
  --messages examples/cli/messages/text_smoke_zh.json \
  --max-context 2048 --max-new 8 --no-thinking --greedy \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

这些命令直接验证 CLI→public `Engine`→Frontend/Runtime 的可执行产品路径，不注册新 CTest target，不把每个 artifact 扩展为 feature 矩阵。本次不修改 schema、protocol translation、ResponseStore 或 CLI options，因此没有独立 serving schema gate；若实际 diff 触及其中任一行为，只运行直接受影响的现有测试并重新判断 scope。

### 9.5 代表性性能检查

本次没有 kernel 性能 claim。下列一组 C=1/C=8 before/after 是 Engine CPU scheduling/transaction 的诊断证据，不是带主观阈值的独立 completion gate。修改前后使用同一 RTX 5090、CUDA/toolchain、artifact、greedy sampling、graph 和 workload：

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 --sampling greedy --suite decode-saturation \
  --concurrency 1 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/engine_architecture_27b_groupwise_mtp3_after
```

阶段 0 使用同一命令并将 output suffix 改为 `_before`；不为本次检查额外要求 NVFP4 或第二个 model artifact。

解读顺序：

1. 逐 C 核对 resolved KV capacity/page groups、CUDA Graph、backend/profile、错误、完成 request/token、full-batch membership、average batch 和 decode-round counters；C=1 与 C=8 因 `auto` 可以解析为不同 capacity，但同一 C 的 before/after 必须可比；
2. 若 workload/counter 结构变化，先判断是实际调度回归还是无效测量，不用 throughput 掩盖结构变化；
3. 在结构可比时记录 steady aggregate decode throughput 与 wave timing。不设置统一 2%/5% hard gate，不预先要求多轮重复；
4. 只有该次诊断实际暴露结构性异常或可疑回归时，才重跑受影响的 C 点并定位；确认为可复现且可归因于本次 Engine CPU/synchronization/membership 变更的回归才是代码问题。

greedy 模式下 MTP acceptance 与 round counters 用于 workload 结构核对；数值/transaction correctness 仍由 real tests 与第 9.4 节精确 CLI oracles 保证。35B MTP、DFlash 和 C=2/C=4 不重复跑性能 campaign，除非 representative run 或对应 real test 实际暴露 backend-specific 异常。

request-time allocation 不设独立 profiler gate：startup-frozen arena test、Program ownership 和代码中无 NInfer-owned request-time allocation path 已构成直接证据。只有 whole-inference 结果实际显示结构性回归且普通计时无法定位时，才使用 nsys；只有识别出相关 kernel 后才使用 ncu。

## 10. 一次性 cutover 审查（不注册为测试）

本节与第 9.1 节一起执行，只确认旧 authority 已被真正替换，不形成第二套 correctness gate：

- `ConcurrentExecutor`、Engine-owned `RequestMemory` 和 raw-lane Program mutation API 已删除；
- Engine 不直接引用 Qwen frontend concrete types，Program 不再拥有 admission/eviction policy；
- terminal commit 不隐式 retain，resident state 必须由 unique continuation handle 表达；start/finish 返回到 ledger/catalog 接管前有显式 local guard，PendingBatch 有 commit/abort_pending/failure cleanup 三条闭合路径；
- Program 是 physical epoch 唯一 authority，ResourceManager 没有复制 generation 或隐藏 stale retry；
- Engine per-lane plan cache 与旧 output commit route 不存在；`serve::PreparedRequest` 保持它原有的 Gateway 语义；
- ResourceManager 的 Free/Active/Resident ledger 在 start/finish/abort/abort_pending/cancel/commit-failure/evict 路径均满足第 5.6 节方程；
- Scheduler/ResourceManager 没有 MTP/DFlash policy branch；
- 没有新旧 contract 双路，也没有 offload/COW placeholder、host bank 或未实现 action。

## 11. 明确延期的下一阶段能力

当前实现只留下可真实承载后续工作的 ownership seam，不预埋不可执行抽象。KV offload 的 host backing/residency 表示、transfer action 和调度/价值策略，以及 COW 的 shared/clone contract，都由下一阶段结合当时真实代码和测量决定，本文不提前固定其类型或 action 形状。

未来 continuation clone 需要保存恢复后可观察执行所依赖的全部持久状态，准确 inventory 以该阶段 Runtime contract 为权威；本阶段只以 unique `ContinuationHandle` 的单消费语义作为清晰起点，不实现 refcount/shared page 或占位 API。

Gateway tool-call parser 的 ownership 调整、Responses session/cache hint、通用 cache cost model、新模型/硬件、preemptive continuous batching也都不属于本阶段。

## 12. 最终完成定义

只有同时满足以下条件，本阶段才完成：

1. 第 5 节的固定 ownership、state、resource 和 transaction contract 已落地，私有 helper 表示不作 source-shape gate；
2. 第 6 节的阶段 1–5 均达到出口条件，且阶段 0 基线可用于回归判断；
3. 第 8.1 节的 focused tests、三个代表性 real Engine routes 和第 9.4 节五个固定 CLI artifact/message smokes 全部通过；
4. 第 9.5 节的 C=1/C=8 诊断已记录并核对 workload 结构；不使用主观 throughput 阈值作为独立 gate，但诊断若实际揭示可复现且可归因的本次回归，该回归需要解决；
5. 第 9.1/10 节的一次性 cutover review 确认 superseded path 已删除；
6. 没有为了测试引入 fake runtime、竞态 hook、全矩阵 campaign 或无关 schema gate；
7. 没有 offload/COW 或长期文档工作混入当前变更。

完成后，本文件不转为长期架构权威。后续统一文档任务应基于已落地代码更新现有 active references，并删除本临时执行文档。
