#pragma once
#include "exl3/gdn_layer.h"

inline void run_residual_norm_operator_comparison() {
    constexpr std::size_t width=5120,guard=128;
    // Explicit FP16 halfway ties and signed-zero addition expectations.
    constexpr std::uint16_t cases[][3]={{0,0x8000,0},{0x8000,0x8000,0x8000},
        {0x3c00,0x1000,0x3c00},{0x3c01,0x1000,0x3c02},{0xbc00,0x9000,0xbc00}};
    for(int rows:{1,7,8,15,16,255,256,512,1024}) {
        const std::size_t count=std::size_t(rows)*width;
        std::vector<std::uint16_t> left(count),right(count),expected(count),weights(width,0);
        constexpr std::uint16_t weight_bits[]={0,0xbc00,0x3c00,0x3800,0xb800};
        for(std::size_t i=0;i<width;++i)weights[i]=weight_bits[i%5];
        for(std::size_t i=0;i<count;++i) {
            const auto& pattern=cases[(i+i/width)%5];
            left[i]=pattern[0];right[i]=pattern[1];expected[i]=pattern[2];
        }
        DeviceBuffer a(count*2),b(count*2),w(width*2),r((count+2*guard)*2),n((count+2*guard)*2);
        cuda_check(cudaMemcpy(a.get(),left.data(),count*2,cudaMemcpyHostToDevice),"residual fixture left");
        cuda_check(cudaMemcpy(b.get(),right.data(),count*2,cudaMemcpyHostToDevice),"residual fixture right");
        cuda_check(cudaMemcpy(w.get(),weights.data(),width*2,cudaMemcpyHostToDevice),"residual fixture weight");
        std::vector<std::uint16_t> canonical_norm,got(count+2*guard);
        for(bool fused:{false,true}) {
            cuda_check(cudaMemset(r.get(),0x5a,got.size()*2),"residual fixture guard");
            cuda_check(cudaMemset(n.get(),0x5a,got.size()*2),"norm fixture guard");
            ninfer::exl3::exl3_gdn_residual_norm(static_cast<const std::uint16_t*>(a.get()),
                static_cast<const std::uint16_t*>(b.get()),static_cast<const std::uint16_t*>(w.get()),
                static_cast<std::uint16_t*>(r.get())+guard,static_cast<std::uint16_t*>(n.get())+guard,rows,fused);
            for(bool normalized:{false,true}) {
                cuda_check(cudaMemcpy(got.data(),normalized?n.get():r.get(),got.size()*2,cudaMemcpyDeviceToHost),"residual norm represented readback");
                require(std::all_of(got.begin(),got.begin()+guard,[](auto x){return x==0x5a5a;}) &&
                    std::all_of(got.end()-guard,got.end(),[](auto x){return x==0x5a5a;}),"residual norm output guard");
                if(!normalized)require(std::equal(expected.begin(),expected.end(),got.begin()+guard),"residual halfway or signed-zero mismatch");
                else if(!fused)canonical_norm.assign(got.begin()+guard,got.end()-guard);
                else require(std::equal(canonical_norm.begin(),canonical_norm.end(),got.begin()+guard),"fused normalization changed represented bits");
            }
            const auto preserved=[&](const void* device,const auto& original) {
                std::vector<std::uint16_t> readback(original.size());
                cuda_check(cudaMemcpy(readback.data(),device,readback.size()*2,cudaMemcpyDeviceToHost),"residual norm input preservation");
                require(readback==original,"residual norm modified input or weights");
            };
            preserved(a.get(),left);preserved(b.get(),right);preserved(w.get(),weights);
            for(int overlap=0;overlap<4;++overlap) {
                auto* residual=static_cast<std::uint16_t*>(r.get())+guard;
                auto* normalized=static_cast<std::uint16_t*>(n.get())+guard;
                if(overlap==0)residual=static_cast<std::uint16_t*>(a.get())+1;
                if(overlap==1)residual=static_cast<std::uint16_t*>(b.get())+1;
                if(overlap==2)normalized=residual+1;
                if(overlap==3)normalized=static_cast<std::uint16_t*>(w.get())+1;
                cuda_check(cudaMemset(r.get(),0x5a,got.size()*2),"residual refusal guard");
                cuda_check(cudaMemset(n.get(),0x5a,got.size()*2),"norm refusal guard");
                bool refused=false;
                try{ninfer::exl3::exl3_gdn_residual_norm(static_cast<const std::uint16_t*>(a.get()),
                    static_cast<const std::uint16_t*>(b.get()),static_cast<const std::uint16_t*>(w.get()),
                    residual,normalized,rows,fused);}
                catch(const std::invalid_argument&){refused=true;}
                require(refused,"residual norm submitted overlapping outputs");
                preserved(a.get(),left);preserved(b.get(),right);preserved(w.get(),weights);
                for(const void* output:{static_cast<const void*>(r.get()),static_cast<const void*>(n.get())}) {
                    cuda_check(cudaMemcpy(got.data(),output,got.size()*2,cudaMemcpyDeviceToHost),"residual refusal output readback");
                    require(std::all_of(got.begin(),got.end(),[](auto x){return x==0x5a5a;}),"residual refusal modified output");
                }
            }
        }
    }
}
