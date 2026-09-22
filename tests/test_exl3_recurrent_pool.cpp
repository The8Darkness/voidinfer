#include "exl3/recurrent_export_pool.h"
#include "exl3/host_resident_set.h"
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
#include "exl3_host_residency.h"
#include <iostream>
#include <cstring>
using namespace ninfer::exl3;
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(){try{
    constexpr std::size_t bytes=144ULL<<20;
    std::array<std::size_t,48> sizes;sizes.fill(bytes/48/4);
    void* source=nullptr;checked(cudaMalloc(&source,bytes));checked(cudaMemset(source,0x3f,bytes));
    std::shared_ptr<Exl3RecurrentSlab> survivor;
    {
        Exl3RecurrentExportPool pool;
        auto a=pool.acquire(sizes),b=pool.acquire(sizes),c=pool.acquire(sizes);
        need(a && b && c && !pool.acquire(sizes),"three-slot exhaustion failed");
        checked(cudaMemcpyAsync(a->data,source,bytes,cudaMemcpyDeviceToHost));
        checked(cudaEventRecord(a->completed));a->recorded=true;checked(cudaEventSynchronize(a->completed));
        need(reinterpret_cast<std::uint32_t*>(a->data)[bytes/4-1]==0x3f3f3f3f,"direct population failed");
        Exl3HostResidentSet registry(2ULL<<30,8ULL<<30);
        Exl3HostResidentSet::Snapshot snapshot;snapshot.owners.push_back(a);snapshot.add(a->data,bytes);
        registry.replace(std::move(snapshot));
        Exl3HostResidencyProbe probe;probe.add(a->data,bytes);auto measured=probe.measure();
        need(measured.locked_tensor_bytes==bytes && measured.resident_tensor_bytes==bytes,"CUDA registration + VirtualLock coexistence failed");
        registry.close();
        auto* address=b->data;b.reset();auto recycled=pool.acquire(sizes);
        need(recycled && recycled->data==address,"retired completed slot not reused");
        recycled->poisoned=true;recycled.reset();
        need(!pool.acquire(sizes),"poisoned slot reused");
        survivor=a;
        std::cout<<"pool allocated="<<pool.stats.allocations<<" hits="<<pool.stats.pool_hits<<" fallback="<<pool.stats.fallbacks
            <<" registration_ms="<<pool.stats.register_ms<<" allocation_ms="<<pool.stats.allocate_ms<<" lock_bytes="<<measured.locked_tensor_bytes<<"\n";
    }
    need(survivor->data[0]==survivor->data[bytes/4-1],"root lost storage after context destruction");
    std::vector<std::unique_ptr<Exl3RecurrentExportPool>> pools;
    std::vector<std::shared_ptr<Exl3RecurrentSlab>> owners;
    for(int i=0;i<3;++i){auto pool=std::make_unique<Exl3RecurrentExportPool>();for(int n=0;n<3;++n){auto slab=pool->acquire(sizes);if(slab)owners.push_back(std::move(slab));}pools.push_back(std::move(pool));}
    const auto peak=Exl3RecurrentPinBudget::snapshot();
    need(owners.size()==6 && peak[0]==7*bytes && peak[0]<=Exl3RecurrentPinBudget::cap,"process cap excludes survivors");
    owners.clear();pools.clear();survivor.reset();checked(cudaFree(source));
    need(Exl3RecurrentPinBudget::snapshot()[0]==0,"registration remains after all owners retire");
    std::cout<<"PASS_RECURRENT_POOL cap_peak="<<peak[1]<<" final_live=0\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
