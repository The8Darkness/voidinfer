# Paged KV 物理容器实施计划

**文档状态：** Step 2 实施完成
**实施范围：** Device 物理页池、执行映射表、Host KV arena、完整 page-group 传输、现有 Program 切换
**相关架构：** [Resource scheduling and context cache](resource-scheduling-and-context-cache.md)
**实施记录：** [Resource Scheduling 与 Context Cache 实施记录](resource-scheduling-and-context-cache-implementation-record.md)

## 1. 文档用途

本文是 Resource Scheduling 与 Context Cache 落地工作的 Step 2，只计划 Paged KV 的物理
容器基座及当前 Program 对新基座的切换。正式架构中的 logical page、address space、replica
publication、prefix sharing、offload policy 和 materialization transaction 不在本阶段实现。

本阶段完成后：

- Device payload、active execution mapping 和 sequence-private ownership 不再由同一个
  `PagedKVAllocation` 表达；
- Main 与 optional Backend KV 使用相同的物理容器 contract，并继续拥有独立 capacity；
- 存在能够容纳任意完整 KV page-group extent 的固定容量 Host 物理容器；
- D2D、D2H 和 H2D 传输不依赖 request、checkpoint 或调度语义；
- 当前 exclusive prefix-reuse 行为切换到新容器，Op 与可观察生成行为保持不变。

本文不是新的架构 authority，也不修改
`resource-scheduling-and-context-cache.md` 或当前行为参考 `paged-kv-cache.md`。完成整个
context-cache 架构切换后，再统一更新当前行为参考。

## 2. 本阶段交付

本阶段交付以下内容：

1. 将现有 `PagedKVPoolSpec` 中的 page payload geometry、physical capacity、logical table
   capacity 和 table rows 分离；
2. 以 generation-safe page capability 和显式 capacity reservation 重写 Device 物理页池；
3. 将 block-table storage、row ownership 和 mapping publication 抽成独立执行映射容器；
4. 新增启动时固定 page-locked capacity 的 variable-size `HostKVArena`；
5. 定义与 Device slab 无关的 packed Host page-record layout；
6. 实现完整 page-group 的 selective zero、D2D copy、D2H 和 H2D；
7. 用 Program-private mapping 取代 `PagedKVAllocation`，切换当前普通、MTP、DFlash、prefix
   reuse 和 CUDA Graph preparation；
8. 删除被替代的 `PagedKVPool/PagedKVAllocation` ownership 路径，不保留 compatibility alias。

本阶段允许 Host 容器和 Host transfer 只由 focused test 直接驱动。Host KV capacity 不接入
Engine 配置，Program 也不创建或发布 Host replica；这些属于后续 logical cache 与 policy
步骤。

## 3. 明确不在范围内的内容

以下内容不属于 Step 2：

- `LogicalKVPageStore`、`KVAddressSpaceHandle` 和 logical page capability；
- content epoch、committed coverage、logical reference 或 write protection；
- shared prefix、branch、tail COW 的判定和 publication；
- Device/Host replica residency 与 transfer state；
- checkpoint validity、retention、victim selection、offload/restore policy；
- `MaterializationTransaction`、异步 transfer ticket 或跨 worker-boundary publication；
- ResourceManager/Scheduler 修改和新的资源摘要；
- active request offload、preemption、KV compression、磁盘存储或跨 Program 迁移；
- Engine/CLI/serving 的 Host KV 配置；
- attention、KV append、MTP 或 DFlash kernel 的数值路径和物理 Device layout；
- benchmark、吞吐提升声明或性能回归 gate。

容器不会保存 request、lane、checkpoint、session、priority、reuse score、logical frontier、
Main/Backend 业务标签或 eviction 状态。

## 4. 物理所有权边界

Step 2 固定以下分层：

```text
Program current private mapping
    ordered Device page leases + unmapped growth reservation
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
 DeviceKVPagePool   HostKVArena   KVExecutionTablePool
 Device replicas   Host allocations active address mapping
          │              │              │
          └──────── explicit transfer ──┘
                         │
             PagedKVLayerView / BatchView
```

未来 `LogicalKVPageStore` 将接管 logical identity、references 和 replica metadata，但继续以
本阶段的 page lease、Host allocation 和 execution row 作为物理能力。物理容器不为未来逻辑层
预建第二份 metadata authority。

## 5. Page geometry 与 layout 拆分

### 5.1 Immutable page geometry

新增 core-owned immutable geometry，语义等价于：

```cpp
struct KVPlaneGeometry {
    DType dtype;
    int32_t leading_extent;
    int32_t head_extent;
    size_t alignment;
};

struct KVPageGeometry {
    uint32_t page_tokens;
    PagedKVPlaneOrder device_order;
    vector<KVPlaneGeometry> planes;
};
```

具体字段布局可服从现有 layout 约定，但 geometry 只回答：一组完整 page-group 有哪些 planes、
每个 plane 的单页 shape/dtype 和 Device axis order。Qwen target 继续决定 plane index 对应哪些
layer K/V/scale；core 不解释 plane 语义。

本阶段继续使用固定 `page_tokens == 64`。该值进入 geometry，以便 Host layout 和 transfer
从同一个事实生成，不引入运行时可变 page size 行为。

### 5.2 Device payload 与 execution metadata 分离

现有 `PagedKVPoolSpec` 拆成两个独立 spec：

```text
DeviceKVPagePoolSpec
    physical page-group count
    KVPageGeometry

KVExecutionTableSpec
    logical page capacity
    execution row count
```

对应 layout 也分开规划：

- `DeviceKVPagePoolLayout` 只包含当前 PageMajor/HeadMajor plane slabs；
- `KVExecutionTableLayout` 只包含固定地址的 I32 block-table matrix；
- Qwen `PagedKVCacheLayout` 组合二者以及 target layer/codec facts。

Device plane shape、dtype、alignment 和 byte count保持现状，因此 attention、append、MTP 与
DFlash Op 消费的 tensor ABI 不变化。

## 6. DeviceKVPagePool

### 6.1 Capability 与所有权

Device 容器使用两个不同角色：

```cpp
class DeviceKVPageHandle;       // copyable non-owning capability
class DeviceKVPageLease;        // move-only physical-replica owner
class DeviceKVPageReservation;  // move-only unmapped-capacity owner
```

handle由 owning pool铸造，概念上携带 owner identity、physical index 和 generation；字段保持
private，调用方不能自行拼装 capability。

`DeviceKVPageLease` 唯一拥有一个 physical page-group，向 copy、zero 和 mapping publication
提供 non-owning handle。page release 后 generation 递增；旧 handle 不得访问后来复用同一
physical index 的页面。

未来共享发生在 logical page 层：一个 logical page 唯一持有一个 Device replica lease，多个
address spaces 引用该 logical page。`DeviceKVPagePool` 不维护 logical refcount，也不复制
physical ownership capability。

正常 Program 路径显式释放 reservation/lease。其 move-only 析构只回收尚未 adoption 的
host-side allocator state，不 enqueue CUDA work；Program teardown 继续是最终兜底。

### 6.2 Reservation 与 materialization

物理页池维护：

```text
allocated physical replicas + reserved unmapped pages <= physical capacity
```

最小语义接口为：

```cpp
optional<DeviceKVPageReservation> reserve(uint32_t pages) noexcept;

void materialize(DeviceKVPageReservation& reservation,
                 uint32_t target_page_count,
                 vector<DeviceKVPageLease>& pages);

void dematerialize(DeviceKVPageReservation& reservation,
                   uint32_t target_page_count,
                   vector<DeviceKVPageLease>& pages);

uint32_t capacity_pages() const noexcept;
uint32_t allocated_pages() const noexcept;
uint32_t reserved_pages() const noexcept;
uint32_t available_pages() const noexcept;
```

- reservation 不预先绑定 physical ID；
- materialize 将 reservation 中的数量原子转换为 page leases；
- 成功 reserve 后，只要输入 capability 有效，materialize 不再因 page capacity 失败；
- materialize 的 destination storage 由 Program 预留，不在 mutation 后扩展 host container；
- dematerialize 将 trailing leases 原子转回同一 reservation，保持 active entitlement 不变；
- release 返回物理页并递增 generation；
- free-list 选择只影响 physical locality，不进入公开 contract。

pool只汇总 physical allocated/reserved 数量，不区分 active-growth、transaction 或 COW
reservation；这些类别以后由 Program transaction 与 ResourceManager ledger解释。

Main 与 Backend bundle reservation 继续具有 all-or-nothing 结果。实现使用一组 move-only
reservation 先逐 pool 取得 capacity；任一 pool 失败时，已经取得的 reservation 由同一调用
栈回收，Program 不 adoption 半个 bundle。该 helper 不解释 Main/Backend，只接受 distinct
physical pools 和 page counts。

当前 active entitlement 表达为：

```text
mapped DeviceKVPageLease count + remaining DeviceKVPageReservation count
```

materialize 只在二者之间转移数量。terminal retain 释放 remaining reservation，只保留实际
mapped page leases；trim 释放超出新 frontier 的 leases。

### 6.3 Physical operations

Device pool 提供以下完整 page-group 操作：

```cpp
void zero_pages(span<const DeviceKVPageHandle>, cudaStream_t);

void copy_page(DeviceKVPageHandle source,
               DeviceKVPageHandle destination,
               cudaStream_t);
```

操作开始 enqueue 前完成 pool ownership、generation、range、distinct destination 和 geometry
校验。它们只修改 payload bytes，不改变 allocator、reservation 或 logical metadata。

source 与 destination 相同的 D2D copy 是无操作；不同页面的 copy 保留 source，并覆盖
destination 的全部 planes。这个原语以后服务 partial-tail COW，本阶段只验证物理能力，不
实现 COW 决策。

caller 保持参与异步 zero/copy 的 page leases 有效直到 stream completion；container 不用
隐式 synchronization 延长 capability lifetime。

## 7. KVExecutionTablePool

### 7.1 Row 不拥有 page

新增独立执行映射容器：

```cpp
struct KVExecutionRowHandle {
    uint32_t row;
    uint32_t generation;
};

class KVExecutionRowLease;

class KVExecutionTablePool {
public:
    KVExecutionRowLease acquire(uint32_t row);
    void publish(KVExecutionRowHandle,
                 span<const DeviceKVPageHandle>,
                 cudaStream_t);
    Tensor row(KVExecutionRowHandle) const;
    Tensor matrix() const noexcept;
};
```

每个 table pool在构造时与一个 `DeviceKVPagePool` 绑定；这个关系只用于 capability validation
和 physical-index translation，不赋予 table 对 pages 的 ownership。

row handle同样由 table pool铸造并包含 owner identity、row index 和 generation。release row
会使旧 handle失效，但不清除或释放任何 KV payload page。

row lease 只表示某个 active execution 对固定 block-table row 的独占使用权。它不持有 page
lease，也不延长 page 生命周期。resident continuation 不持有 row；重新 active 后可在任意
合法 row 上发布同一 ordered mapping。

Program 保持被 row引用的 page leases覆盖该 row参与的全部 GPU units；解除 row不会回收
pages，释放 page前则先确保没有 execution row或 in-flight unit仍会读取它。

`publish`：

- 验证所有 page handles 属于与该 table 配对的 Device pool；
- 按输入顺序写 physical page indices；
- 不改变 ordered page ownership；
- 使用启动期固定的 per-row host shadow/staging，调用期间不分配；
- 只 enqueue 到 caller stream，不隐式 synchronize。

同一 row 的 host staging 在前一次 publication 完成前不被改写。当前 worker/GPU-unit 边界
已经满足这一点；容器 contract 显式保留该 lifetime 要求。

`PagedKVLayerView` 和 `PagedKVBatchLayerView` 继续作为 Op contract。Qwen `PagedKVCache`
从 page pool planes 与 execution row/matrix 组合这些 view，不再接收 owning allocation。

## 8. HostKVArena

### 8.1 Canonical Host page record

Host layout 从同一个 `KVPageGeometry` 生成，但不复制 Device slab axis order。一个 Host page
record 为：

```text
[plane 0 one-page payload]
[plane 1 one-page payload]
...
[alignment padding]
```

每个 plane offset 与整个 record stride 使用 256-byte alignment。一个 extent 为连续的
logical-order records：

```text
[logical page 0 record][logical page 1 record] ...
```

Host payload保存各 plane 的原始 dtype、logical axis order 与 bit pattern，不执行量化、
反量化或压缩。Device `HeadMajor` 通过 stride-aware gather/scatter 写入同一 page-record
表示，不改变 Host canonical representation。

Host allocation不保存 logical page number、physical page ID、content epoch、coverage、pool
role 或 residency。正式架构中的 Program-level `HostKVExtent` 将 logical range、replica
metadata 和一个物理 allocation/slice关联起来。

### 8.2 Bounded variable-size arena

新增 core-owned：

```cpp
class HostKVArena;
class HostKVAllocation;       // move-only physical allocation
class HostKVAllocationHandle; // copyable non-owning generation capability
```

allocation handle由 arena铸造并携带 owner identity、descriptor index 和 generation；字段不向
调用方开放。release 或 split消费原 owner后，旧 handle不再形成有效 view。

`HostKVArena` 构造时接收固定 byte capacity，以及该 Program 启动时已经闭合的
`HostKVPageLayout` 集合。它创建一块 `PinnedHostBuffer`，并按最小 record stride预留足以表达
arena 内最大 page-record allocation 数量的 descriptor storage。capacity 为零时不创建零字节
page-locked allocation。运行期只在该 backing 和固定 descriptor storage 中分配、拆分、合并
extent，不调用 `cudaHostAlloc`，不增长 allocator metadata，也不使用 pageable fallback。

supported layouts只是物理 record geometry，不包含 Main/Backend、checkpoint 或 policy 标签。
arena 不接受构造后才出现的新 layout；当前 Program 的 geometry 在模型加载时已经固定。

最小能力为：

```cpp
optional<HostKVAllocation> allocate(const HostKVPageLayout&, uint32_t pages) noexcept;
bool can_allocate(const HostKVPageLayout&, uint32_t pages) const noexcept;

HostKVAllocationView writable_view(HostKVAllocation&);
HostKVAllocationConstView view(const HostKVAllocation&) const;

pair<HostKVAllocation, HostKVAllocation> split(HostKVAllocation&&,
                                               uint32_t page_offset);
```

split 只在 page-record 边界切分 ownership，不复制 payload。通过两次 split，未来调用方可以
释放一个 extent 中的任意 page-rounded 子区间。split消费原 extent capability并铸造两个新的
generation capabilities；相邻 free extents重新合并。

arena 的 `occupied_bytes` 计算实际持有的 aligned extent bytes。`free_bytes` 只用于诊断；
`can_allocate(layout, pages)` 执行与 allocation 相同的 physical-fit 检查，但不保留空间；一次
allocation 的返回结果仍是最终事实。调用方不能自行把 free bytes 换算为成功承诺。

Host allocation由 transaction 或未来 logical replica owner持有。容器本身不区分“reserved
target”和“published replica”；publication 状态属于 Program。

## 9. Device/Host transfer contract

在 core 物理层提供：

```cpp
void DeviceKVPagePool::copy_to_host(span<const DeviceKVPageHandle> source,
                                    HostKVAllocationView destination,
                                    cudaStream_t);

void DeviceKVPagePool::copy_from_host(HostKVAllocationConstView source,
                                      span<const DeviceKVPageHandle> destination,
                                      cudaStream_t);
```

语义固定为：

- span 顺序就是 Host allocation 中的 logical page 顺序；
- source/destination page count 与 geometry 完全一致；
- 每次传输覆盖所列 page-group 的全部 planes；
- physical page ID 不写入 Host payload；
- 输入 handle、extent 和 geometry 在任何 CUDA enqueue 前完成验证；
- 接口只 enqueue，不创建 event、不 synchronize、不发布 replica；
- caller 保持 source、destination 和 Host allocation 有效，直到 stream completion。

实现按输入顺序识别连续 physical page runs：

- PageMajor plane按 run 合并 copy；
- HeadMajor plane按 head/run 使用正确 stride copy；
- fragmented physical mapping拆成多个 runs；
- Host 端仍保持 page-record logical order。

最后一个部分有效页仍传输完整 physical page。`committed coverage` 以后由 Program metadata
限制有效 columns；coverage 之外的 bytes 不因传输而获得内容语义。

## 10. 当前 Program 切换

### 10.1 Private mapping

`SequenceKVBundle` 保留 Main/optional Backend 的 typed-pool 组合，但其中不再保存
`PagedKVAllocation`。每个 pool使用一个 Program-private mapping，语义为：

```cpp
struct PrivateKVMapping {
    vector<DeviceKVPageLease> ordered_pages;
    DeviceKVPageReservation remaining_growth;
    optional<KVExecutionRowLease> execution_row;
};
```

具体内部类型名可以按 family runtime 现有命名调整；上述三个 ownership 角色不再合并回 core
container。

mapping向 execution-table publisher暴露 ordered non-owning page-handle view，且该 view的建立
不分配内存、不复制 ownership。publish接口最终接收 handle span、lease span还是等价的固定
writer view属于局部表示选择；不能为此维护第二份可独立修改的 page-ID authority。

当前 lifecycle 映射如下：

```text
Active:
    ordered pages + remaining growth reservation + execution row

Resident continuation:
    ordered pages only

Released:
    no page lease, reservation, or execution row
```

当前 prefix reuse 继续通过 move private mapping实现 exclusive ownership transfer。本阶段不
增加 logical sharing/refcount；这只是当前 Program policy，不能下沉回 Device pool。

### 10.2 Existing paths

以下现有路径同步切换：

- request admission 的 Main/Backend bundle reservation；
- prefill/decode 的 incremental materialization；
- terminal/rejected suffix trim；
- retain 时释放 unmapped growth reservation；
- continuation claim/resize/rebind；
- abort、release 和 fail-all cleanup；
- CUDA Graph capture 的 temporary pages、repeated mappings 和 rows；
- ordinary、MTP、DFlash full KV views；
- memory summary 中的 KV payload bytes 与 physical capacity facts。

所有 current page mapping继续使用相同 logical page order。新容器切换不改变 Text/MTP/DFlash
frontier、off-by-one 规则、commit 时点或 prefix identity。

## 11. 代码改动范围

预计直接修改：

- `src/core/paged_kv_cache.h`
- `src/core/paged_kv_cache.cpp`
- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h`
- `src/targets/qwen3_6/impl/state/decoder_state.cpp`
- `src/targets/qwen3_6/impl/runtime/layouts.h`
- `src/targets/qwen3_6/impl/runtime/layouts_impl.h`
- `src/targets/qwen3_6/impl/runtime/program.h`
- `src/targets/qwen3_6/impl/runtime/program_impl.h`
- `tests/test_kv_cache.cpp`
- `tests/targets/qwen3_6/test_runtime_mechanisms.cpp`
- `tests/CMakeLists.txt`
- `src/CMakeLists.txt`

预计新增：

- `src/core/host_kv_arena.h`
- `src/core/host_kv_arena.cpp`

若 transfer implementation独立成文件，应只作为 core physical helper，不建立第二个 page
owner。实际受 `PagedKVAllocation/PagedKVPool` contract影响的 consumer均在本阶段切换；文件
名本身不是完成条件。

本阶段不修改：

- `src/runtime` Engine、Scheduler、ResourceManager；
- attention/KV append/MTP/DFlash Op contract 或 kernels；
- artifact、CLI、serving 和 benchmark；
- `docs/maintainer/resource-scheduling-and-context-cache.md`。

## 12. 实施顺序

### Step 1：Geometry 与 layout 分离

- 建立 `KVPageGeometry`；
- 分开规划 Device payload 和 execution table；
- 保持现有 PageMajor/HeadMajor plane shape及 arena byte accounting；
- 切换 Qwen `PagedKVCacheLayout` 的组合关系。

阶段内检查：layout-focused `ninfer_kv_cache_test`。

### Step 2：Device page ownership

- 实现 physical generation、page lease 和 reservation；
- 实现无二次 capacity failure 的 materialize；
- 实现 complete-page selective zero 与 D2D copy；
- 保留 distinct typed pools 的 bundle reservation；
- 移除 core 中的 sequence entitlement/allocation 语义。

阶段内检查：`ninfer_kv_cache_test` 的 reservation、reuse、stale handle、source isolation。

### Step 3：Execution mapping cutover

- 新增 table layout/pool、row lease 与固定 host shadow；
- 切换 single-row 和 compact-batch Op views；
- 切换 CUDA Graph preparation；
- 确认 resident continuation 不占 execution row。

阶段内检查：KV container test 与 Qwen runtime mechanisms test。

### Step 4：Host arena 与 transfer

- 新增 Host page-record layout；
- 实现 bounded pinned extent allocation、split、release 和 generation validation；
- 实现 PageMajor/HeadMajor D2H/H2D roundtrip；
- 实现 fragmented physical-run mapping。

阶段内检查：扩展 `ninfer_kv_cache_test`，不新增 fake Program。

### Step 5：Program production cutover 与清理

- 用 Program-private mapping替换 `PagedKVAllocation`；
- 切换 Main、MTP、DFlash、prefix reuse、retain/release 和 graph capture；
- 删除 superseded `PagedKVPool/PagedKVAllocation` API 和实现；
- 执行第 13 节验证。

## 13. 验证与完成条件

### 13.1 Focused physical-container evidence

重写现有 `ninfer_kv_cache_test`，使用小型 synthetic geometry覆盖本阶段新增的真实 contract：

1. PageMajor 与 HeadMajor Device plane layout保持预期；
2. reservation/materialize/release数量守恒，失败的 multi-pool bundle不留下 reservation；
3. released page generation递增，并通过一条代表性访问路径验证 centralized stale-handle
   rejection，不为每个 method复制同型测试；
4. execution row publication保持输入 logical order，row release不释放 pages；
5. selective zero和D2D完整覆盖所有 planes，source及其他 pages不变；
6. PageMajor 与 HeadMajor执行 Device A -> Host allocation -> Device B 后完整 page-group
   bit-exact；
7. fragmented physical page顺序仍恢复为相同 Host logical order；
8. Host allocation exhaustion、split、子区间 release/reuse 和 generation invalidation符合实际
   语义。

测试直接使用 production geometry、pool、mapping 和 transfer，不引入 scheduler、logical page
store、cache policy 或故障注入。

### 13.2 Existing mechanism evidence

运行：

```bash
cmake --build build -j --target \
  ninfer_kv_cache_test \
  ninfer_qwen3_6_runtime_mechanisms_test

ctest --test-dir build --output-on-failure \
  -R '^(ninfer_kv_cache_test|ninfer_qwen3_6_runtime_mechanisms_test)$'
```

若构建依赖使其他既有 focused target自然参与，不把它们追加为独立 gate。

### 13.3 Representative real execution

Device backing、block-table publication和 production Program ownership均被替换，因此最终
复用 Step 1 已存在的三个代表性 real routes：

```text
ninfer_qwen3_6_27b_prefix_real_test
ninfer_qwen3_6_35b_a3b_real_test
ninfer_qwen3_6_35b_a3b_dflash_real_test
```

使用仓库已有 artifacts 和测试环境变量。三个路径分别验证当前 exclusive prefix transfer、
Main+MTP typed pools 和 Main+DFlash-full typed pools；不扩张 identity、dtype、concurrency 或
协议矩阵。

验收只要求既有结果断言、runtime state 和 CUDA execution成功。本阶段没有性能声明，吞吐
和 transfer overlap不是完成 gate。

### 13.4 Diff 与边界检查

```bash
git diff --check
```

并确认：

- 正式架构文档和当前行为参考没有被提前改写；
- production Program 不再持有 `PagedKVAllocation`；
- block-table row不拥有 physical pages；
- Host payload不保存 Device physical IDs；
- 没有 Host KV policy、logical replica metadata 或 compatibility alias；
- attention、append、MTP 和 DFlash Op 的数值 contract未改变。

## 14. 执行检查表

| 项目 | 状态 | 证据或说明 |
|---|---|---|
| KVPageGeometry 与 layout split | 完成 | Device payload 与 execution metadata 独立规划 |
| DeviceKVPagePool | 完成 | generation lease、reservation、materialize/dematerialize、zero/copy |
| KVExecutionTablePool | 完成 | 独立 row lease、固定 pinned shadow 与 mapping publication |
| HostKVArena | 完成 | bounded pinned extent、page-boundary split、generation view |
| D2D/D2H/H2D transfer | 完成 | PageMajor/HeadMajor 完整 page-group 与 physical-run coalescing |
| Program private mapping cutover | 完成 | ordinary/MTP/DFlash/prefix/graph 共用新容器 |
| superseded path cleanup | 完成 | 已删除 PagedKVAllocation/PagedKVPool ownership path |
| focused tests | 完成 | `ninfer_kv_cache_test`、runtime mechanisms 2/2 |
| representative real routes | 完成（有环境限制） | 27B prefix 与 35B DFlash 通过；35B Text/MTP/prefix/Vision 段完成，maximum-capacity 子段受当时可用显存限制 |
