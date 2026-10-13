#pragma once
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace voidinfer::exl3 {

// Per-request sampling policy of the coherent-device route (sampled_select.cuh).
// temperature 0 = argmax. presence_penalty (OpenAI semantics, output tokens
// only) is subtracted from the logit of every token already generated before
// the position being chosen: first_occurrence[token] holds the first output
// position of each generated token (kExl3AbsentOccurrence if none), maintained
// by the Engine from its committed and drafted output. frequency penalties are
// not part of this route.
constexpr std::int32_t kExl3AbsentOccurrence=0x7f7f7f7f;   // memset byte 0x7f
struct Exl3DeviceSampling {
    float temperature=0.0f;
    std::int32_t top_k=0;
    float top_p=1.0f;
    float min_p=0.0f;
    std::uint64_t seed=0;
    float presence_penalty=0.0f;
    const std::int32_t* first_occurrence=nullptr;
    // The coupled selector runs for sampled draws and for penalized argmax.
    bool active() const noexcept {return temperature>0.0f || presence_penalty!=0.0f;}
    bool sampled() const noexcept {return temperature>0.0f;}
};

// First-occurrence table maintenance (stream-ordered): reset marks every
// token absent; truncate drops occurrences at positions >= `from`; record
// enters `tokens` at positions first, first+1, ... (keeping earlier ones).
void exl3_presence_reset(std::int32_t* table,cudaStream_t stream);
void exl3_presence_truncate(std::int32_t* table,int from,cudaStream_t stream);
void exl3_presence_record(std::int32_t* table,int first,const std::int64_t* tokens,std::size_t count,cudaStream_t stream);

} // namespace voidinfer::exl3
