#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
namespace ninfer::exl3 {
// Qualification-only leaves: borrowed device spans, no allocation or environment
// reads. The fixture owns full extents and synchronizes before reading/freeing.
void gopt_gdn_conv_fixture(bool fused,const std::uint16_t* input,
    const std::uint16_t* weight,std::uint16_t* state,std::uint16_t* conv_input,
    std::uint16_t* q,std::uint16_t* k,std::uint16_t* v,std::uint16_t* packed,
    std::uint16_t* trace,int rows,cudaStream_t stream=nullptr,bool decode_fused=false);
void gopt_gdn_pack_fixture(bool fused,const std::uint16_t* core,
    const std::uint16_t* norm,std::uint16_t* trace,std::uint16_t* projection,
    int rows,cudaStream_t stream=nullptr);
void gopt_gdn_control_row_pair_fixture(bool paired,const std::uint16_t* input,
    const std::uint16_t* a_weight,const std::uint16_t* b_weight,
    const float* a_log,const float* dt_bias,float* a_output,float* b_output,
    float* beta_trace,float* g_trace,int rows,cudaStream_t stream=nullptr);
void gopt_draft_rope_fixture(bool paired,const std::uint16_t* input,
    std::uint16_t* output,const int* positions,int rows,int heads,cudaStream_t stream=nullptr);
void gopt_draft_head_fixture(bool fused,bool paired,const std::uint16_t* input,
    const std::uint16_t* weight,std::uint16_t* normalized,std::uint16_t* rotated,
    const int* positions,int rows,int heads,cudaStream_t stream=nullptr);
void gopt_draft_ring_norm_fixture(bool fused,const std::uint16_t* k,
    const std::uint16_t* v,const std::uint16_t* weight,const int* positions,
    std::uint16_t* normalized,std::uint16_t* rotated,std::uint16_t* ring_k,
    std::uint16_t* ring_v,int rows,int first,cudaStream_t stream=nullptr);
void fast_wmma_attention_fixture(bool rows32,const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,const int* position_device,
    int query_offset,cudaStream_t stream=nullptr,bool padded=false);
void fast_wmma_split2_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* split_output,float* split_stats,int rows,int position,int capacity,
    const int* position_device,int query_offset,cudaStream_t stream=nullptr,
    bool padded=false);
void fast_wmma_split4_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* split_output,float* split_stats,int rows,int position,int capacity,
    const int* position_device,int query_offset,cudaStream_t stream=nullptr,
    bool padded=false);
void fast_wmma32_register_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* split_output,float* split_stats,int rows,int position,int capacity,
    const int* position_device,int query_offset,int split_count,int heads,
    cudaStream_t stream=nullptr,int split_block_m=32);
void fast_wmma64_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* split_output,float* split_stats,int rows,int position,int capacity,
    const int* position_device,int query_offset,cudaStream_t stream=nullptr,
    bool padded=false,bool split2=false,bool register_owned=false,
    bool shared_heads=false,bool keys64=false);
void fast_fused_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,float* workspace,
    std::uint16_t* output,int position,int capacity,int segments,
    const int* position_device,cudaStream_t stream=nullptr,int rows=1);
void gopt_draft_conv_fixture(bool fused,const std::uint16_t* input,
    const std::uint16_t* dynamic,const std::uint16_t* base,std::uint16_t* conv,
    const std::uint16_t* residual,std::uint16_t* output,int rows,cudaStream_t stream=nullptr);
void gopt_draft_norm_fixture(bool fused,const std::uint16_t* left,
    const std::uint16_t* right,const std::uint16_t* weight,std::uint16_t* residual,
    std::uint16_t* normalized,int rows,cudaStream_t stream=nullptr);
void gopt_draft_dense_fixture(bool paired,bool kmajor,const std::uint16_t* input,
    const std::uint16_t* weight,std::uint16_t* output,int rows,int k,int n,
    cudaStream_t stream=nullptr);
}
