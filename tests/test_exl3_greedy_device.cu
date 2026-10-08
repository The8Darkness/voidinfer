#include "exl3/greedy_packet.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
using namespace ninfer::exl3;
static void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
// Independent mathematical conversion of represented binary16, no production
// converter or reduction tree. Fixtures below are authored bit patterns.
static float value(std::uint16_t h) {
    const int exp=(h>>10)&31, mant=h&1023;
    if(exp==31) return mant?std::numeric_limits<float>::quiet_NaN():
        (h&32768?-INFINITY:INFINITY);
    const float magnitude=exp?std::ldexp(float(1024+mant),exp-25):std::ldexp(float(mant),-24);
    return h&32768?-magnitude:magnitude;
}
static void compare_route(const std::vector<std::uint16_t>& input,int rows,int vocab,int stride,bool warp) {
    std::uint16_t* device=nullptr;Exl3GreedyRow* output=nullptr;void* split=nullptr;
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&device),input.size()*2));
        check(cudaMalloc(reinterpret_cast<void**>(&output),rows*sizeof(Exl3GreedyRow)));
        check(cudaMemcpy(device,input.data(),input.size()*2,cudaMemcpyHostToDevice));
        check(cudaMalloc(&split,Exl3GreedySplitScratch::bytes()));
        check(cudaMemset(split,0,Exl3GreedySplitScratch::bytes()));
        exl3_launch_greedy_packet(warp,rows,nullptr,device,vocab,stride,91,output,Exl3GreedySplitScratch::carve(split));
        check(cudaGetLastError());
        std::vector<Exl3GreedyRow> actual(rows);
        check(cudaMemcpy(actual.data(),output,rows*sizeof(Exl3GreedyRow),cudaMemcpyDeviceToHost));
        for(int row=0;row<rows;++row) {
            std::vector<float> logical(vocab);
            for(int i=0;i<vocab;++i) logical[i]=value(input[row*stride+i]);
            const bool bad=std::any_of(logical.begin(),logical.end(),[](float x){return !std::isfinite(x);});
            const int expected=static_cast<int>(std::max_element(logical.begin(),logical.end())-logical.begin());
            if(actual[row].serial!=91 || bool(actual[row].nonfinite)!=bad || (!bad && actual[row].token!=expected))
                throw std::runtime_error("represented greedy disagreement");
        }
    } catch(...) {cudaFree(output);cudaFree(device);throw;}
    check(cudaFree(output));check(cudaFree(device));
}
static void compare(const std::vector<std::uint16_t>& input,int rows,int vocab,int stride) {
    compare_route(input,rows,vocab,stride,false);
    compare_route(input,rows,vocab,stride,true);
}
int main() {
    try {
        // Distinct B8 request slices require private winners and nonfinite flags.
        {
            constexpr int vocab=248320,stride=vocab+7;
            std::vector<std::uint16_t> paired(16ULL*stride,0x7e00);
            for(int row=0;row<16;++row) {
                std::fill_n(paired.begin()+row*stride,vocab,0xbc00);
                paired[row*stride+(row<8?0:256)]=0x3c00;
                paired[row*stride+vocab-1]=0x3c00;
            }
            compare(paired,16,vocab,stride);
            paired[7ULL*stride+vocab-1]=0x7e01;
            compare(paired,16,vocab,stride);
            for(int row=0;row<8;++row)
                std::swap_ranges(paired.begin()+row*stride,paired.begin()+(row+1)*stride,
                    paired.begin()+(row+8)*stride);
            compare(paired,16,vocab,stride);
        }
        for(int rows:{1,8,16}) for(int vocab:{1,255,257,1025,248320}) {
            const int stride=vocab+7;
            std::vector<std::uint16_t> input(rows*stride,0x7e00); // masked padding NaNs
            for(int row=0;row<rows;++row)
                std::fill_n(input.begin()+row*stride,vocab,0xbc00); // all negative, ties
            compare(input,rows,vocab,stride);
            // Adjacent represented values, subnormal boundary, and finite
            // extremes. Authored bits distinguish ranking from rounded FP32
            // fixture generation; lowest-index duplicates cross warp lanes.
            for(auto pair:{std::pair{0x3c00,0x3c01},std::pair{0x0001,0x0002},
                           std::pair{0x03ff,0x0400},std::pair{0xfbff,0xfbfe},
                           std::pair{0x7bfe,0x7bff}}) {
                auto near=input;
                for(int row=0;row<rows;++row) {
                    std::fill_n(near.begin()+row*stride,vocab,static_cast<std::uint16_t>(pair.first));
                    near[row*stride+vocab-1]=static_cast<std::uint16_t>(pair.second);
                    if(vocab>256)near[row*stride+256]=static_cast<std::uint16_t>(pair.second);
                }
                compare(near,rows,vocab,stride);
            }
            for(int row=0;row<rows;++row) {
                input[row*stride+vocab-1]=0x0000;
                input[row*stride]=0x8000; // signed-zero tie, lowest index
            }
            compare(input,rows,vocab,stride);
            for(int row=0;row<rows;++row) {
                input[row*stride+vocab-1]=0x3c00;
                if(vocab>256) input[row*stride+256]=0x3c00;
            }
            compare(input,rows,vocab,stride);
            for(auto bad:{0x7c00,0xfc00,0x7e01}) {
                auto invalid=input;invalid[(rows-1)*stride+vocab-1]=static_cast<std::uint16_t>(bad);
                compare(invalid,rows,vocab,stride);
            }
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
