# StateImage 物理容器实施计划

**文档状态：** Step 1 已实施
**实施范围：** Linear Attention state、continuation hidden、DFlash local fixed state、Host 完整镜像
**相关架构：** [Resource scheduling and context cache](resource-scheduling-and-context-cache.md)
**实施记录：** [Resource scheduling 与 context cache 实施记录](resource-scheduling-and-context-cache-implementation-record.md)

## 1. 文档用途

本文只计划 Step 1 准备落地的物理 state 容器及其传输能力。它不是新的架构
authority，不修改正式架构文档中的 checkpoint、resource scheduling 或 context-cache
contract。

本阶段完成后，当前 Program 使用的 state storage 应由统一的完整 StateImage slot 表达，
并存在一个能够保存同一完整 payload 的固定容量 Host 容器。本文不规划其后的调度、缓存
策略或 KV page 工作。

## 2. 本阶段交付

本阶段交付以下内容：

1. 收紧并补齐现有 `LinearAttentionStatePool` 的 slot-oriented physical-storage contract；
2. 新增 Qwen3.6 family-owned `StateImageDevicePool`，把同一 frontier 的所有固定 device
   state 放入同一个逻辑 slot；
3. 将当前 endpoint/rewrite continuation hidden 收敛为一份按 StateImage slot 索引的
   hidden pool；
4. DFlash 启用时，将 local cyclic K/V 作为 StateImage 的 backend fixed-state component；
5. 新增以完整 StateImage 为单位的 page-locked `HostStatePool`；
6. 提供完整 slot 的 Device-to-Device、Device-to-Host 和 Host-to-Device 异步字节传输；
7. 将当前 Program 的 state backing 切换到新容器，同时保持现有请求、prefix reuse、MTP
   和 DFlash 可观察行为不变。

本阶段不引入第二条 state 路径。新容器接管生产 storage 后，删除被替代的独立
tail/rewrite hidden backing 和 DFlash local/rewrite-local backing。

## 3. 明确不在范围内的内容

以下内容不属于本阶段：

- Text、MTP 或 DFlash full paged KV 的布局、分配、共享、迁移或回收；
- Scheduler、ResourceManager、ContinuationCatalog 或 admission policy；
- checkpoint 选择、retention、eviction、Host offload policy；
- `C + H` 容量策略、Device slot 借用规则或 active concurrency entitlement；
- logical `StateImageHandle`、checkpoint identity 或 Device/Host replica publication；
- GDN execution 的独立 source/destination selector 改造；
- Prefix Fork、Freeze、Snapshot、Restore 的 Program lifecycle；
- 新的 CLI/serving option、artifact format 或持久化文件格式；
- 性能 benchmark 或吞吐回归 gate。

`HostStatePool` 在本阶段是物理容器。它的容量由构造参数直接给出，不接入 Engine 配置，
也不自行决定何时创建或逐出 Host replica。

## 4. StateImage payload contract

一幅物理 StateImage 表示一个精确 frontier 上、继续执行当前 Program 所需的全部固定尺寸
device state。它不包含增长型 KV、checkpoint metadata 或 round transient。

### 4.1 Common payload

所有 backend 都包含：

```text
Linear Attention convolution history
Linear Attention FP32 recurrent state
Target continuation hidden
```

`continuation hidden` 是该 frontier 最后一个 committed target token 对应的 BF16 hidden，
shape 为 `[TextConfig::hidden]`。一幅 image 只有这一份 hidden。Endpoint 和 rewrite 是两幅
不同 image，不在一幅 image 中同时保存 tail hidden 和 rewrite hidden。

### 4.2 MTP payload

MTP 不增加固定尺寸 StateImage component。MTP StateImage 等于 common payload。

下列数据不进入 StateImage：

- MTP paged KV；
- `mtp_kv_valid`；
- `mtp_drafts` 和 `mtp_draft_count`；
- MTP bridge、proposal、verify 和 accept 的 round tensors。

当前 terminal commit 会清空 draft count；checkpoint 恢复所需的 bridge/proposal 可以由
continuation hidden、token 语义和 MTP KV 重新建立。因此 drafts 属于 active-round control，
不是不可重建的 fixed state。

### 4.3 DFlash payload

DFlash StateImage 包含 common payload，并额外包含：

```text
DFlash local cyclic K for every local layer
DFlash local cyclic V for every local layer
```

DFlash local cyclic K/V 是固定尺寸 backend state。虽然现有实现使用
`CyclicKVCache` 表达，它不作为独立的增长型 KV 资源调度。

下列 DFlash 数据不进入 StateImage：

- DFlash full paged KV；
- `dflash_context_frontier`；
- prefill features、prefill positions、pending features；
- proposal/verify/accept round tensors。

`dflash_context_frontier` 继续属于 Program sequence metadata。Host/device state copy 只传输
local cyclic storage 的完整物理字节。

### 4.4 Physical slot homogeneity

一个 Program 实例的 backend 在启动时已经固定，因此同一 `StateImageDevicePool` 中所有 slots
具有相同 payload：

```text
Ordinary or MTP: GDN + continuation hidden
DFlash:          GDN + continuation hidden + DFlash local cyclic state
```

每个 slot 不保存 runtime backend tag，也不存在 partial StateImage。

## 5. LinearAttentionStatePool 修改

### 5.1 保留的职责

`LinearAttentionStatePool` 继续是 caller-owned device backing 上的固定容量 GDN state
子容器。它只负责：

- layout validation；
- layer/slot views；
- slot zero；
- slot copy；
- 提供 Op 和完整 StateImage transfer 所需的物理几何。

它不拥有 slot role、generation、request/lane identity、allocation policy 或 CUDA stream。

### 5.2 封装内部状态

将可写的 `conv`、`recurrent` 和 `spec` 成员改为 private storage。公开接口收敛为：

```cpp
const LinearAttentionStatePoolSpec& spec() const noexcept;
uint32_t layer_count() const noexcept;
int32_t slot_count() const noexcept;

LinearAttentionStateLayerView layer_view(uint32_t layer) const;
LinearAttentionStateSlotView slot_view(int32_t slot) const;
LinearAttentionStateAllLayersView all_layers_view() const;

void zero_slot(int32_t slot, cudaStream_t stream);
void zero_all(cudaStream_t stream);
void copy_slot(int32_t source, int32_t destination, cudaStream_t stream);
```

保留现有 Op 使用的 `all_layers_view()`。调用方不再直接修改 Tensor inventory 或从 vector
地址自行推导 layer pitch。

### 5.3 Slot transfer view

新增的 slot view 应完整描述两个 strided component：

```cpp
struct LinearAttentionStateSlotView {
    Tensor conv_layer0;
    Tensor recurrent_layer0;

    size_t conv_layer_bytes;
    size_t recurrent_layer_bytes;
    ptrdiff_t conv_layer_pitch_bytes;
    ptrdiff_t recurrent_layer_pitch_bytes;
    uint32_t layers;
};
```

`conv_layer0` 和 `recurrent_layer0` 只覆盖选定 slot。Pitch 指向下一 layer 中同一 slot 的
起始地址。该 view 同时服务 D2D 和 Host transfer，不暴露 allocation ownership。

### 5.4 Copy implementation

整 slot copy 根据 slot transfer view 复制所有 convolution 和 recurrent layers。实现可以在
CUDA pitch/overlap contract 允许时使用 strided copy；否则保留逐 layer async copy。该局部
选择不进入公共 contract，也不为本阶段引入专用 copy kernel。Planner/constructor 在建立
pool 时验证 row bytes、layer pitch 和 slot offset 可以被 transfer view 精确表达。

普通 decode 不调用整 slot copy，因此该改动不改变 decode 热路径。

## 6. StateImageDevicePool

### 6.1 Ownership and layout

在 Qwen3.6 family state 层新增：

```cpp
struct StateImageDeviceLayout {
    LinearAttentionStatePoolLayout linear;
    TensorLayout continuation_hidden;               // [hidden, slots]
    optional<CyclicKVCacheLayout> dflash_local;
};

class StateImageDevicePool {
public:
    int32_t slot_count() const noexcept;
    StateImageDeviceSlotView slot_view(int32_t slot) const;

    void zero_slot(int32_t slot, cudaStream_t stream);
    void copy_slot(int32_t source, int32_t destination, cudaStream_t stream);
    void copy_to_host(int32_t source, HostStateImageView destination,
                      cudaStream_t stream);
    void copy_from_host(HostStateImageConstView source, int32_t destination,
                        cudaStream_t stream);

    LinearAttentionStatePool& linear() noexcept;
    Tensor continuation_hidden_store() const noexcept;
    CyclicKVCache* dflash_local() noexcept;
    const CyclicKVCache* dflash_local() const noexcept;
};
```

Device layout 和第 7 节的 Host layout 由同一个 immutable StateImage geometry 规划；Host 侧
不根据 Device tensor 指针或当前 backend 重新推导 component inventory。Planner 一次确定
linear geometry、hidden width、optional DFlash geometry 和 slot count，并生成两侧 layout。

具体字段形式可以根据现有 layout/binding 约定调整，但以下语义固定：

- 三个 component 使用同一个 absolute state slot；
- `copy_slot` 复制当前 backend 的完整 StateImage；
- source 和 destination 相同时 copy 是无操作；
- source 与 destination 不同时 source 保持不变；
- copy/zero 只 enqueue 到 caller 提供的 stream，不隐式 synchronize；
- 容器不分配 slot，也不解释 current/rewrite/cache role。

### 6.2 Current Program binding

本阶段保持当前 Program 的 lane/current/rewrite 行为，现有两类物理角色映射到完整 image
slots：

```text
current(lane) = lane
rewrite(lane) = C + lane
slot_count    = 2 * C
```

原 `LinearStateSlots` 只描述 Linear Attention component，切换后应由 target-local
`StateImageSlots` 替代，使 GDN、hidden 和 DFlash local 使用同一组 selector。

当前两份 hidden backing：

```text
tail_hidden[hidden, C]
rewrite_checkpoint_hidden[hidden, C]
```

收敛为：

```text
continuation_hidden[hidden, 2 * C]
```

`SequenceState::tail_hidden` 和 `SequenceState::rewrite_checkpoint_hidden` 可以继续作为选定
slot 的 non-owning Tensor view，从而避免在本阶段改写 sequence lifecycle。

### 6.3 DFlash local integration

DFlash local 与 rewrite-local 收敛到同一个、按 StateImage slot 索引的 local-state pool：

```text
old: local[C] + rewrite_checkpoint_local[C]
new: local_state[2 * C]
```

现有 `CyclicKVCache` 在这里作为 DFlash fixed-state physical component 使用。本阶段只为它
补充任意 source/destination slot copy：

```cpp
void copy_slot_from(const CyclicKVCache& source,
                    int32_t source_slot,
                    int32_t destination_slot,
                    cudaStream_t stream);
```

该接口替代只能复制同名 lane 的 `copy_lane_from(source, lane)`；capacity、绝对位置取模、
layer view 和 DFlash Op 消费的 tensor layout 均不改变。这项修改不涉及 `PagedKVCache`。

本阶段的 save/restore 操作继续使用当前 role mapping：

```text
capture rewrite:
    GDN snapshot、hidden capture 和 DFlash-local component copy
    分别写入同一个 rewrite(lane) image slot

restore rewrite:
    完整 StateImage rewrite(lane) -> current(lane)
```

Capture 期间不能用完整 image copy 覆盖已经由精确 frontier snapshot 写入的 GDN/hidden
component。只有 DFlash local component 使用 `copy_slot_from`；当三个 component 都已完成时，
rewrite slot 才构成完整 image。Restore 可以直接使用完整 `StateImageDevicePool::copy_slot`。

执行 DFlash model 的 current rows 仍使用 `current(lane)`。DFlash full paged cache、block table
和 context frontier 的实现保持不变。

## 7. HostStatePool

### 7.1 Packed host layout

新增 family-owned `StateImageHostLayout`。一个 Host slot 的 byte order 为：

```text
[linear conv layers]
[linear recurrent layers]
[continuation hidden]
[optional DFlash local K layers]
[optional DFlash local V layers]
```

每个 component 起点和完整 image stride 使用 256-byte alignment。Host layout 保存有效
state payload，不复制 Device arena 中只用于 region alignment 的间隙。

Host copy 保持每个 component 的原始 dtype 和 bit pattern：

- convolution：当前 Variant 规定的 dtype；
- recurrent：FP32；
- continuation hidden：BF16；
- DFlash local K/V：BF16。

本阶段不做压缩、量化、endianness 转换或磁盘序列化。

### 7.2 Physical pool

`HostStatePool` 使用一块启动时创建的 `PinnedHostBuffer`，容量为：

```text
aligned_image_stride * slot_count
```

其最小接口为：

```cpp
struct HostStateSlotHandle {
    uint32_t index;
    uint32_t generation;
};

class HostStatePool {
public:
    optional<HostStateSlotHandle> allocate() noexcept;
    bool release(HostStateSlotHandle) noexcept;

    HostStateImageView writable_view(HostStateSlotHandle);
    HostStateImageConstView view(HostStateSlotHandle) const;

    uint32_t capacity() const noexcept;
    uint32_t occupied() const noexcept;
};
```

约束如下：

- 构造期完成全部 page-locked allocation；
- `allocate/release/view` 不进行 host/device allocation；
- capacity 为零时不构造零字节 `PinnedHostBuffer`；
- release 后递增该 physical slot 的 generation；
- stale handle 不能访问或释放已复用的 slot；
- pool 不理解 checkpoint kind、session、priority 或 eviction；
- caller 保证参与异步 copy 的 Host/Device slot 在 stream completion 前不被 release 或重用。

`allocate()` 返回空表示固定容量已满，不通过 pageable allocation 扩容。

### 7.3 Transfer contract

`StateImageDevicePool::copy_to_host/copy_from_host` 传输当前 backend 的完整 payload：

```text
GDN conv          strided device <-> packed host
GDN recurrent     strided device <-> packed host
hidden            contiguous device <-> host
DFlash local K/V  strided device <-> packed host, when enabled
```

接口只 enqueue，不隐式创建 stream、event 或执行 synchronize。CUDA 错误通过现有
`CUDA_CHECK` 规则报告。成功返回只表示 enqueue 成功；测试或调用者在检查 Host/Device 数据
前负责同步 stream。传输前检查 Host view 与 Device pool 来自相同的 StateImage geometry；
layout mismatch 在任何 CUDA enqueue 之前报告。

## 8. 代码改动范围

预计直接修改：

- `src/core/linear_attention_state.h`
- `src/core/linear_attention_state.cpp`
- `src/core/cyclic_kv_cache.h`
- `src/core/cyclic_kv_cache.cpp`
- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h`
- `src/targets/qwen3_6/impl/state/decoder_state.cpp`
- `src/targets/qwen3_6/impl/runtime/layouts.h`
- `src/targets/qwen3_6/impl/runtime/layouts_impl.h`
- `src/targets/qwen3_6/impl/runtime/program.h`
- `src/targets/qwen3_6/impl/runtime/program_impl.h`
- `src/targets/qwen3_6/impl/runtime/dflash_context.h`
- `src/targets/qwen3_6/impl/runtime/dflash_context_impl.h`
- 使用 continuation-hidden store 或 DFlash local view 的 family schedule 文件；
- `src/targets/qwen3_6/CMakeLists.txt`
- `tests/test_state_store.cpp`
- `tests/targets/qwen3_6/test_runtime_mechanisms.cpp`
- `tests/CMakeLists.txt`

预计新增：

- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/state_image.h`
- `src/targets/qwen3_6/impl/state/state_image.cpp`
- `tests/targets/qwen3_6/test_state_image.cpp`

若实现中无需某个列出的文件，不把“触及全部文件”作为完成条件。反之，凡是仍直接读取被
封装成员或持有旧 hidden/DFlash backing 的实际 consumer，都在本阶段同步切换。

本阶段不修改：

- `src/core/paged_kv_cache.*`
- Qwen3.6 `PagedKVCache` / `PagedKVAllocation` contract；
- Engine、Scheduler、ResourceManager；
- `docs/maintainer/resource-scheduling-and-context-cache.md`。

## 9. 实施顺序

### Step 1：Linear Attention physical contract

- 封装 pool inventory；
- 增加 layer/slot transfer views；
- 实现 `zero_all` 和完整 `copy_slot`；
- 切换现有 Op/Program consumer 到只读 views；
- 保持现有数值路径和 slot bytes 不变。

阶段内检查：`ninfer_state_store_test` 和 `ninfer_gdn_replay_records_test`。

### Step 2：完整 Device StateImage

- 新增 `StateImageDeviceLayout/Pool`；
- 从 `DecoderState` 中抽离 Linear Attention storage；
- 合并 continuation-hidden backing；
- 将所有 current/rewrite hidden consumer 改为完整 slot selector；
- 合并 DFlash local/rewrite-local backing；
- 让 current Program 使用新 pool。

阶段内检查：family runtime mechanism test，以及普通、MTP、DFlash 现有构建目标。

### Step 3：Host physical image and transfers

- 新增 packed Host layout；
- 新增 fixed-capacity pinned Host slots 和 generation validation；
- 完成完整 image D2H/H2D；
- 在 family-focused CUDA test 中验证 roundtrip 和 source isolation。

### Step 4：清理和最终验证

- 删除 superseded hidden 和 DFlash rewrite backing；
- 删除对 `LinearAttentionStatePool` public inventory 的直接访问；
- 确认没有新增 paged-KV 行为或 Engine 配置；
- 执行第 10 节的最终检查。

## 10. 验证与完成条件

### 10.1 Focused container evidence

保留 `ninfer_state_store_test` 对 core Linear Attention container 的验证，并新增一个
`ninfer_qwen3_6_state_image_test`，使用小型 synthetic geometry 验证：

1. 多 layer、多 slot 的 conv/recurrent shape 和 dtype；
2. `zero_slot` 只修改目标 slot；
3. 完整 D2D copy 覆盖 GDN、hidden 和 optional DFlash local，且 source 不变；
4. common 与 DFlash 两种完整 payload 执行 Device A -> Host -> Device B 后 bit-exact；
5. Host capacity exhaustion、release/reuse 和 stale generation 的实际语义。

这些检查使用 production layout/view/copy 实现，不引入 fake Program、调度 hook 或故障注入。

### 10.2 Existing mechanism evidence

运行：

```bash
cmake --build build -j --target \
  ninfer_state_store_test \
  ninfer_gdn_replay_records_test \
  ninfer_qwen3_6_state_image_test \
  ninfer_qwen3_6_runtime_mechanisms_test

ctest --test-dir build --output-on-failure \
  -R '^(ninfer_state_store_test|ninfer_gdn_replay_records_test|ninfer_qwen3_6_state_image_test|ninfer_qwen3_6_runtime_mechanisms_test)$'
```

### 10.3 Real execution evidence

由于本阶段切换生产 Program 的 state backing，最终运行三个既有代表性路径：

```text
27B prefix/MTP real test
35B ordinary/MTP real test
35B DFlash real test
```

使用仓库已有 artifact 和测试环境变量，不新增 identity/backend/concurrency 矩阵。验收只要求
请求成功、既有结果断言通过和没有 state/layout failure；本阶段不声明性能改善，因此吞吐
变化不是 gate。

### 10.4 Document and diff checks

```bash
git diff --check
```

并确认：

- 正式架构文档没有被修改；
- paged-KV contract 和实现没有被修改；
- 没有保留新旧两套生产 state backing；
- 没有为了内部 API 变更增加兼容 alias。

## 11. 执行检查表

实施期间只在下表跟踪本计划内的执行状态；架构整体的实施事实同步到总实施记录，不在这里
追加其他步骤。

| 项目 | 状态 | 证据或说明 |
|---|---|---|
| LinearAttentionStatePool contract | 完成 | inventory 已封装，并提供 layer/slot view、copy、zero |
| StateImageDevicePool | 完成 | common 与 optional DFlash component 共享 absolute slot |
| continuation hidden 收敛 | 完成 | `[hidden, 2C]` 单一 backing |
| DFlash local fixed state 收敛 | 完成 | 单一 `[2C]` cyclic backing；full paged KV 未变 |
| HostStatePool | 完成 | 固定 pinned capacity、generation handle、无分配 allocate/release |
| 完整 image D2D/D2H/H2D | 完成 | caller stream 上异步完整 payload transfer |
| focused tests | 完成 | 5/5：core state、ReplaySSM、StateImage、runtime mechanisms |
| representative real routes | 完成 | 3/3：27B prefix/MTP、35B ordinary/MTP、35B DFlash |
