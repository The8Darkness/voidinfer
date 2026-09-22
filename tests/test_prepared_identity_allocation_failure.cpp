#include "exl3/prepared_identity.h"
#include <cstdlib>
#include <new>
#include <iostream>
#include <malloc.h>

// Isolated executable: allocation interception never enters Engine or any
// numerical test process. Arm only around the operation being qualified.
static thread_local long allocation_countdown=-1;
static void allocation_point() {
    if(allocation_countdown==0) throw std::bad_alloc();
    if(allocation_countdown>0) --allocation_countdown;
}
void* operator new(std::size_t bytes) {
    allocation_point();if(auto* p=std::malloc(bytes?bytes:1))return p;throw std::bad_alloc();
}
void* operator new[](std::size_t bytes){return ::operator new(bytes);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
void* operator new(std::size_t bytes,std::align_val_t align) {
    allocation_point();if(auto* p=_aligned_malloc(bytes?bytes:1,static_cast<std::size_t>(align)))return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes,std::align_val_t align){return ::operator new(bytes,align);}
void operator delete(void* p,std::align_val_t) noexcept {_aligned_free(p);}
void operator delete[](void* p,std::align_val_t) noexcept {_aligned_free(p);}
void operator delete(void* p,std::size_t,std::align_val_t) noexcept {_aligned_free(p);}
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept {_aligned_free(p);}

namespace family=ninfer::targets::qwen3_6;
using family::detail::ResidentPrefixIdentity;
using family::detail::PrefixShortlistDigests;
static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main() {
    try {
        family::PreparedPromptData prompt;
        prompt.token_ids={7,8};prompt.token_types={0,0};prompt.positions={0,1,0,1,0,1};
        ResidentPrefixIdentity original;original.assign(prompt);
        auto expected=original;expected.append_generated(32,0);
        int failures=0;bool finished=false;
        for(long fail=0;fail<32;++fail) {
            auto candidate=original;bool failed=false;
            allocation_countdown=fail;
            try {candidate.append_generated(32,0);}catch(const std::bad_alloc&){failed=true;}
            allocation_countdown=-1;
            if(!failed){need(candidate.equals(expected),"successful append differs");finished=true;break;}
            ++failures;need(candidate.equals(original),"failed reserve changed logical identity");
            candidate.append_generated(32,0);need(candidate.equals(expected),"retry after reserve failure differs");
        }
        need(finished && failures>=4,"identity allocation failure coverage missing");
        PrefixShortlistDigests initial;initial.assign(prompt);
        const std::array<ninfer::TokenId,32> suffix{};
        auto digest_expected=initial;digest_expected.append_generated(suffix,0);
        finished=false;failures=0;
        for(long fail=0;fail<16;++fail) {
            auto candidate=initial;bool failed=false;allocation_countdown=fail;
            try {candidate.append_generated(suffix,0);}catch(const std::bad_alloc&){failed=true;}
            allocation_countdown=-1;
            if(failed) {
                ++failures;need(candidate.size()==initial.size(),"failed digest size changed");
                for(std::size_t i=0;i<=initial.size();++i)need(candidate.at(i)==initial.at(i),"failed digest content changed");
                candidate.append_generated(suffix,0);
            }
            need(candidate.size()==digest_expected.size(),"digest retry size");
            for(std::size_t i=0;i<=candidate.size();++i)need(candidate.at(i)==digest_expected.at(i),"digest retry content");
            if(!failed){finished=true;break;}
        }
        need(finished && failures>0,"digest allocation failure coverage missing");
        return 0;
    } catch(const std::exception& e) {allocation_countdown=-1;std::cerr<<e.what()<<'\n';return 1;}
}
