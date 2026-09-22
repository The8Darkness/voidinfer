#pragma once
#include "exl3/packed_projection.h"
#include "exl3/linear_cuda.h"

namespace ninfer::exl3 {
struct Exl3PackedProjectionReceipt {
    std::uint64_t physical_batches=0,rows=0,gather_bytes=0,scatter_bytes=0;
};
// Bounded stateless dispatcher for explicitly admitted Q, K/V, O and draft-Q
// families. Engine's owned rendezvous orders cross-stream producers before this
// call. Raw research callers must supply equivalent lifetime/ordering guarantees.
// Attention, residual state, rings and publication remain private to each lane.
// On CUDA failure the owner must drain/poison its stream before retiring any
// buffers, exactly as for the existing low-level linear forward API.
inline bool exl3_packed_target_candidate(const Exl3CudaLinearWorkspace& workspace,
    const Exl3CudaLinearMetadata& metadata,int rows,Exl3CudaLinearAdmission admission) noexcept {
    return (admission==Exl3CudaLinearAdmission::target_continuation_q &&
        workspace.target_q_k6_small_m_candidate(metadata,rows,admission)) ||
        (admission==Exl3CudaLinearAdmission::target_continuation_kv &&
        workspace.target_kv_small_m_candidate(metadata,rows,admission)) ||
        (admission==Exl3CudaLinearAdmission::draft_shared_q_m16 &&
        workspace.draft_shared_q_m16_candidate(metadata,rows,admission)) ||
        (admission==Exl3CudaLinearAdmission::draft_shared_kv_m16 &&
        workspace.draft_shared_kv_m16_candidate(metadata,rows,admission)) ||
        (admission==Exl3CudaLinearAdmission::draft_shared_o_m16 &&
        workspace.draft_shared_o_m16_candidate(metadata,rows,admission)) ||
        (admission==Exl3CudaLinearAdmission::draft_shared_down_m16 &&
        workspace.draft_shared_down_m16_candidate(metadata,rows,admission)) ||
        (admission==Exl3CudaLinearAdmission::draft_shared_gateup_m16 &&
        workspace.draft_shared_gateup_m16_candidate(metadata,rows,admission)) ||
        (admission==Exl3CudaLinearAdmission::target_continuation_o &&
        (workspace.target_o_k6_small_m_candidate(metadata,rows,admission) ||
         workspace.target_o_k7_small_m_candidate(metadata,rows,admission))) ||
        (admission==Exl3CudaLinearAdmission::target_continuation_gate_up &&
         (workspace.target_gateup_small_m_candidate(metadata,rows,admission) ||
          workspace.target_gateup_k5_small_m_candidate(metadata,rows,admission))) ||
        (admission==Exl3CudaLinearAdmission::target_continuation_down &&
         workspace.target_down_small_m_candidate(metadata,rows,admission)) ||
        (admission==Exl3CudaLinearAdmission::target_continuation_head &&
         workspace.target_head_small_m_candidate(metadata,rows,admission));
}
template<class Predicate=Exl3PackedProjectionPlan::Live>
inline Exl3PackedProjectionReceipt dispatch_exl3_packed_target_projection(
    const Exl3PackedProjectionPlan& plan,Exl3CudaLinearWorkspace& workspace,
    const Exl3CudaLinearWeights& weights,const Exl3CudaLinearMetadata& metadata,
    std::uint16_t* packed_input,std::size_t input_elements,
    std::uint16_t* packed_output,std::size_t output_elements,
    const Predicate& live,cudaStream_t stream,
    Exl3CudaLinearAdmission admission,bool reuse_completed_gather=false,
    bool fail_after_first_scatter_for_test=false) {
    // Reject incomplete physical backing before any gather is submitted. All
    // admitted families use the canonical trellis/scales/mul1 representation.
    if(!weights.trellis || !weights.suh || !weights.svh || !weights.mul1)
        throw std::invalid_argument("shared projection weight backing missing");
    const bool eligible=exl3_packed_target_candidate(workspace,metadata,plan.rows(),admission);
    const auto input_columns=metadata.in_features,output_columns=metadata.out_features;
    if(!eligible || input_columns<=0 || output_columns<=0 ||
       input_elements<static_cast<std::size_t>(plan.rows())*input_columns ||
       output_elements<static_cast<std::size_t>(plan.rows())*output_columns)
        throw std::invalid_argument("shared target projection admission/capacity");
    for(const auto& lane:plan.lanes())
        if(lane.source.input_columns!=input_columns || lane.source.output_columns!=output_columns)
            throw std::invalid_argument("shared Q descriptor shape");
    plan.require_disjoint_scratch(packed_input,input_columns,packed_output,output_columns);
    plan.validate_live(live);
    const auto check=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    Exl3PackedProjectionReceipt receipt;
    // Only the owning Engine may assert reuse after consuming an immutable
    // signature for this exact ordered plan and unchanged reserved input.
    if(!reuse_completed_gather)for(const auto& lane:plan.lanes()) {
        const auto& s=lane.source;
        check(cudaMemcpy2DAsync(packed_input+static_cast<std::size_t>(lane.packed_first)*input_columns,input_columns*2,
            s.input,s.input_stride*2,input_columns*2,s.rows,cudaMemcpyDeviceToDevice,stream));
        receipt.gather_bytes+=static_cast<std::size_t>(s.rows)*input_columns*2;
    }
    // One combined forward, preserving the admitted kernel's per-row input
    // transform, MMA and split reduction. Exact parity is still a pending gate.
    workspace.forward(weights,metadata,packed_input,packed_output,plan.rows(),stream,admission);
    receipt.physical_batches=1;receipt.rows=plan.rows();
    check(cudaStreamSynchronize(stream));
    plan.validate_live(live); // canceled work cannot scatter into private outputs
    for(const auto& lane:plan.lanes()) {
        const auto& s=lane.source;
        check(cudaMemcpy2DAsync(s.output,s.output_stride*2,
            packed_output+static_cast<std::size_t>(lane.packed_first)*output_columns,output_columns*2,
            output_columns*2,s.rows,cudaMemcpyDeviceToDevice,stream));
        receipt.scatter_bytes+=static_cast<std::size_t>(s.rows)*output_columns*2;
        // Leave the first scatter potentially in flight. The owning Engine
        // must retain both destinations and scratch until its failure drain.
        if(fail_after_first_scatter_for_test)
            throw std::runtime_error("injected shared projection partial scatter failure");
    }
    check(cudaStreamSynchronize(stream));
    // Receipt means computation completed, never token/state authorization.
    plan.validate_live(live);
    return receipt;
}
template<class Predicate=Exl3PackedProjectionPlan::Live>
inline Exl3PackedProjectionReceipt dispatch_exl3_packed_q_projection(
    const Exl3PackedProjectionPlan& plan,Exl3CudaLinearWorkspace& workspace,
    const Exl3CudaLinearWeights& weights,const Exl3CudaLinearMetadata& metadata,
    std::uint16_t* input,std::size_t input_elements,std::uint16_t* output,std::size_t output_elements,
    const Predicate& live,cudaStream_t stream=nullptr) {
    return dispatch_exl3_packed_target_projection(plan,workspace,weights,metadata,input,input_elements,
        output,output_elements,live,stream,Exl3CudaLinearAdmission::target_continuation_q);
}
} // namespace ninfer::exl3
