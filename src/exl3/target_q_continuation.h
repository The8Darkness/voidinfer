#pragma once
#include "exl3/linear_cuda.h"
#include "exl3/activation_lifetime.h"
#include "exl3/draft_private_segment.h"
#include <functional>
#include <array>
#include <optional>
#include <memory>

namespace ninfer::exl3 {
class Exl3DraftHostRing;
enum class Exl3TargetSharedFamily {q,k,v,draft_q,o,gate,up,down,head,draft_k,draft_v,draft_o,draft_down,draft_gate,draft_up,count};
enum class Exl3TargetSharedScope {target_layer,draft_layer,output_head};
// Only batch-invariant represented arithmetic may enter a shared physical
// projection. A route matching an older independent canonical result remains an
// independent fallback, and a numeric research route remains default-off; the
// three claims are intentionally not interchangeable.
enum class Exl3TargetSharedArithmeticIntent {exact_batch_invariant,canonical_independent_only,numeric_research};
struct Exl3TargetSharedCapability {
    Exl3TargetSharedFamily family;
    int in_features,out_features;
    unsigned k_mask;
    unsigned combined_rows_mask;
    Exl3CudaLinearAdmission admission;
    Exl3TargetSharedScope scope;
    Exl3TargetSharedArithmeticIntent arithmetic_intent;
    bool supports(const Exl3CudaLinearMetadata& metadata) const noexcept {
        return metadata.K>=0 && metadata.K<static_cast<int>(8*sizeof(k_mask)) &&
            !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
            metadata.in_features==in_features && metadata.out_features==out_features &&
            (k_mask&(1u<<metadata.K));
    }
    bool supports_combined_rows(int rows) const noexcept {
        return arithmetic_intent==Exl3TargetSharedArithmeticIntent::exact_batch_invariant &&
            rows>=0 && rows<static_cast<int>(8*sizeof(combined_rows_mask)) &&
            (combined_rows_mask&(1u<<rows));
    }
    bool supports_layer(int layer) const noexcept {
        switch(scope) {
            case Exl3TargetSharedScope::target_layer:return layer>=0 && layer<64;
            case Exl3TargetSharedScope::draft_layer:return layer>=0 && layer<5;
            case Exl3TargetSharedScope::output_head:return layer==64;
        }
        return false;
    }
};
// This is the executable family table: target_shared_admission and the Engine
// rendezvous both consume it. It describes represented EXL3 arithmetic routes,
// not merely equal matrix dimensions; target and private-draft entries therefore
// remain distinct even where their shapes happen to agree.
inline constexpr std::array<Exl3TargetSharedCapability,
    static_cast<std::size_t>(Exl3TargetSharedFamily::count)> exl3_target_shared_capabilities{{
    {Exl3TargetSharedFamily::q,5120,12288,1u<<6,0x1fffc,Exl3CudaLinearAdmission::target_continuation_q,Exl3TargetSharedScope::target_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::k,5120,1024,(1u<<7)|(1u<<8),0x1fffc,Exl3CudaLinearAdmission::target_continuation_kv,Exl3TargetSharedScope::target_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::v,5120,1024,(1u<<7)|(1u<<8),0x1fffc,Exl3CudaLinearAdmission::target_continuation_kv,Exl3TargetSharedScope::target_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::draft_q,5120,4096,1u<<5,1u<<16,Exl3CudaLinearAdmission::draft_shared_q_m16,Exl3TargetSharedScope::draft_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::o,6144,5120,(1u<<6)|(1u<<7),0x1fffc,Exl3CudaLinearAdmission::target_continuation_o,Exl3TargetSharedScope::target_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::gate,5120,17408,(1u<<5)|(1u<<6)|(1u<<7),0x1fffc,Exl3CudaLinearAdmission::target_continuation_gate_up,Exl3TargetSharedScope::target_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::up,5120,17408,(1u<<5)|(1u<<6)|(1u<<7),0x1fffc,Exl3CudaLinearAdmission::target_continuation_gate_up,Exl3TargetSharedScope::target_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::down,17408,5120,(1u<<6)|(1u<<7),0x1fffc,Exl3CudaLinearAdmission::target_continuation_down,Exl3TargetSharedScope::target_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::head,5120,248320,1u<<6,0x1fffc,Exl3CudaLinearAdmission::target_continuation_head,Exl3TargetSharedScope::output_head,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::draft_k,5120,1024,1u<<5,1u<<16,Exl3CudaLinearAdmission::draft_shared_kv_m16,Exl3TargetSharedScope::draft_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::draft_v,5120,1024,1u<<5,1u<<16,Exl3CudaLinearAdmission::draft_shared_kv_m16,Exl3TargetSharedScope::draft_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::draft_o,4096,5120,1u<<5,1u<<16,Exl3CudaLinearAdmission::draft_shared_o_m16,Exl3TargetSharedScope::draft_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::draft_down,17408,5120,1u<<5,1u<<16,Exl3CudaLinearAdmission::draft_shared_down_m16,Exl3TargetSharedScope::draft_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::draft_gate,5120,17408,1u<<5,1u<<16,Exl3CudaLinearAdmission::draft_shared_gateup_m16,Exl3TargetSharedScope::draft_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant},
    {Exl3TargetSharedFamily::draft_up,5120,17408,1u<<5,1u<<16,Exl3CudaLinearAdmission::draft_shared_gateup_m16,Exl3TargetSharedScope::draft_layer,Exl3TargetSharedArithmeticIntent::exact_batch_invariant}
}};
inline const Exl3TargetSharedCapability* target_shared_capability(
    Exl3TargetSharedFamily family,const Exl3CudaLinearMetadata& metadata) noexcept {
    const auto index=static_cast<std::size_t>(family);
    if(index>=exl3_target_shared_capabilities.size())return nullptr;
    const auto& capability=exl3_target_shared_capabilities[index];
    return capability.family==family &&
        capability.arithmetic_intent==Exl3TargetSharedArithmeticIntent::exact_batch_invariant &&
        capability.supports(metadata)?&capability:nullptr;
}
inline std::optional<Exl3CudaLinearAdmission> target_shared_admission(
    Exl3TargetSharedFamily family,const Exl3CudaLinearMetadata& m) noexcept {
    const auto* capability=target_shared_capability(family,m);
    return capability?std::optional{capability->admission}:std::nullopt;
}
// Explicit eager layer suspension: the calling lane cannot advance or recycle
// its private scratch until this callback returns. False resumes the independent
// Q projection; true certifies that the destination is complete. Throwing aborts
// the layer. No callback is installed for the default route or graph capture.
// T054: input is the already-rounded private activation at this exact boundary.
// The synchronous callback may gather it only until return. A same-address later
// gate/up/down offer is not a reusable value: residual/SiLU may have overwritten
// it. No transformed-input pointer crosses this interface; every selected
// workspace applies its own weight-dependent canonical transform. Gate and up
// share the existing normalized value, never each other's transformed value.
struct Exl3TargetQContinuation {
    const Exl3CudaLinearWeights& weights;
    const Exl3CudaLinearMetadata& metadata;
    const std::uint16_t* input;
    std::uint16_t* output;
    int rows,position,layer;
    cudaStream_t stream;
    Exl3TargetSharedFamily family=Exl3TargetSharedFamily::q;
    const void* head_identity=nullptr; // draft target H6 backing; target Q/K/V use null
    Exl3ActivationLifetime::Witness activation{};
    // Draft conditioning uses the complete pinned head backing, not trellis alone.
    std::array<const void*,4> head_backing{};
    Exl3CudaLinearMetadata head_metadata{};
    bool supported_draft_head() const noexcept {
        return head_identity && head_backing[0]==head_identity && head_backing[1] &&
            head_backing[2] && head_backing[3] &&
            target_shared_admission(Exl3TargetSharedFamily::head,head_metadata).has_value();
    }
};
using Exl3TargetQExecutor=std::function<bool(const Exl3TargetQContinuation&)>;
// Metadata admission is separate; this compares physical identity without
// dereferencing storage. Empty identities remain valid for target-only offers.
inline bool same_projection_head_backing(const void* first_identity,
    const std::array<const void*,4>& first,const void* second_identity,
    const std::array<const void*,4>& second) noexcept {
    return first_identity==second_identity && first==second;
}
struct Exl3DraftSharedQContinuation {
    Exl3TargetQContinuation projection;
    std::uint64_t acquisition,execution;
    std::int64_t seed,ring_base;
    int ring_count;
    Exl3DraftPrivateSegment segment;
    // Actual completed conditioning, held through synchronous shared dispatch.
    std::shared_ptr<const Exl3DraftHostRing> conditioning_parent;
    bool matches_conditioning_owner(const std::shared_ptr<const Exl3DraftHostRing>& expected) const noexcept {
        return conditioning_parent && expected && conditioning_parent.use_count() && expected.use_count() &&
            conditioning_parent==expected && !conditioning_parent.owner_before(expected) &&
            !expected.owner_before(conditioning_parent);
    }
};
using Exl3DraftSharedQExecutor=std::function<bool(const Exl3DraftSharedQContinuation&)>;
}
