#pragma once
namespace ninfer::exl3 {
// Each verifier row has its own chronological endpoint. A future tiled reader
// may share represented history loads, but must not share this row's mask.
struct Exl3AttentionCausalRows {
    int position=0,rows=0,capacity=0;
#if defined(__CUDACC__)
    __host__ __device__
#endif
    constexpr int count(int query) const noexcept {
        if(position<0 || rows<1 || capacity<rows || position>capacity-rows || query<0 || query>=rows)return 0;
        return position+query+1;
    }
#if defined(__CUDACC__)
    __host__ __device__
#endif
    constexpr bool attends(int query,int key) const noexcept {
        return key>=0 && key<count(query);
    }
};
}
