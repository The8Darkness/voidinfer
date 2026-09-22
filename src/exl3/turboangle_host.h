#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::exl3 {
// Bounded CPU TurboAngle donor implementation (paper arXiv:2603.27467,
// Algorithm 1 + quantized pair norms). Fixed d256 seeded sign rotation.
// K: 7 angle + 8 linear norm bits/pair; V: 6 angle + 4 log norm bits/pair.
// Every record includes two FP32 norm parameters; total K248/V168 bytes.
// Approximate only; cannot construct or modify an authoritative L2 image.
struct TurboAngle256 {
    static constexpr std::uint64_t seed = 0x74616e676c653235ULL;
    static double sign(int i) {
        std::uint64_t x = seed + (static_cast<std::uint64_t>(i) + 1) * 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return ((x ^ (x >> 31)) & 1) ? 1.0 : -1.0;
    }
    static void hadamard(std::array<double,256>& x) {
        for (int stride = 1; stride < 256; stride *= 2)
            for (int base = 0; base < 256; base += stride * 2)
                for (int j = 0; j < stride; ++j) {
                    const auto a = x[base+j], b = x[base+j+stride];
                    x[base+j] = a+b; x[base+j+stride] = a-b;
                }
        for (auto& value : x) value /= 16.0;
    }
    static std::size_t bytes(bool key) { return key ? 248 : 168; }
    static std::vector<std::uint8_t> encode(std::span<const float,256> input, bool key) {
        constexpr double pi = 3.1415926535897932384626433832795;
        std::array<double,256> x;
        for (int i=0;i<256;++i) {
            if (!std::isfinite(input[i])) throw std::invalid_argument("TurboAngle nonfinite input");
            x[i] = input[i] * sign(i);
        }
        hadamard(x);
        std::array<double,128> norms;
        for (int i=0;i<128;++i) {
            const double radius = std::hypot(x[2*i], x[2*i+1]);
            norms[i] = key ? radius : std::log(std::max(radius, 1e-30));
        }
        const int angle_bits=key?7:6, norm_bits=key?8:4;
        const int bins=1<<angle_bits, levels=(1<<norm_bits)-1;
        const float low=static_cast<float>(*std::min_element(norms.begin(),norms.end()));
        const float scale=static_cast<float>(std::max(0.0,*std::max_element(norms.begin(),norms.end())-low)/levels);
        std::vector<std::uint8_t> output(bytes(key),0);
        std::memcpy(output.data(),&low,4); std::memcpy(output.data()+4,&scale,4);
        int cursor=64;
        const auto put=[&](unsigned value,int bits) {
            for(int bit=0;bit<bits;++bit,++cursor) output[cursor/8]|=((value>>bit)&1U)<<(cursor%8);
        };
        for(int i=0;i<128;++i) {
            const int angle=static_cast<int>(std::lround(bins*std::atan2(x[2*i+1],x[2*i])/(2*pi)))&(bins-1);
            const int norm=scale>0 ? std::clamp(static_cast<int>(std::lround((norms[i]-low)/scale)),0,levels):0;
            put(angle,angle_bits); put(norm,norm_bits);
        }
        return output;
    }
    static std::array<float,256> decode(std::span<const std::uint8_t> record, bool key) {
        constexpr double pi=3.1415926535897932384626433832795;
        if(record.size()!=bytes(key)) throw std::invalid_argument("TurboAngle record extent");
        float low=0,scale=0;
        std::memcpy(&low,record.data(),4); std::memcpy(&scale,record.data()+4,4);
        if(!std::isfinite(low)||!std::isfinite(scale)||scale<0)
            throw std::invalid_argument("TurboAngle norm parameters");
        const int angle_bits=key?7:6,norm_bits=key?8:4;
        int cursor=64;
        const auto get=[&](int bits) {
            unsigned value=0;
            for(int bit=0;bit<bits;++bit,++cursor) value|=((record[cursor/8]>>(cursor%8))&1U)<<bit;
            return value;
        };
        std::array<double,256> x;
        for(int i=0;i<128;++i) {
            const auto angle=get(angle_bits),norm=get(norm_bits);
            double radius=low+static_cast<double>(norm)*scale;
            if(!key) radius=std::exp(radius);
            const double theta=2*pi*angle/(1<<angle_bits);
            x[2*i]=radius*std::cos(theta); x[2*i+1]=radius*std::sin(theta);
        }
        hadamard(x);
        std::array<float,256> result;
        for(int i=0;i<256;++i) result[i]=static_cast<float>(x[i]*sign(i));
        return result;
    }
};
} // namespace ninfer::exl3
