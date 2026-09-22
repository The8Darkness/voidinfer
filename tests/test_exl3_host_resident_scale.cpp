#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "exl3/host_resident_set.h"
#include <cstring>
#include <iostream>

using Set=ninfer::exl3::Exl3HostResidentSet;
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
struct Block {
    static constexpr std::size_t bytes=128*1024;
    unsigned char* data;
    explicit Block(std::uint64_t first_page) {
        data=static_cast<unsigned char*>(VirtualAlloc(nullptr,bytes,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
        require(data!=nullptr,"scale allocation");std::memset(data,37,bytes);
        // Every physical page has a distinct cookie, preventing identical-page
        // combining from disguising the physical allocation used by this test.
        for(std::size_t p=0;p<bytes/4096;++p) *reinterpret_cast<std::uint64_t*>(data+p*4096)=first_page+p+1;
    }
    ~Block(){VirtualFree(data,0,MEM_RELEASE);}
};
int main(int argc,char** argv) {
    try {
        require(argc==2,"scale needs explicit GiB");const auto gib=std::stoull(argv[1]);
        require(gib>=1 && gib<=72,"bounded scale GiB");const auto bytes=gib*(1ULL<<30);
        MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
        require(GlobalMemoryStatusEx(&memory)!=0 && memory.ullAvailPhys>bytes+(8ULL<<30)+(512ULL<<20),"scale initial8GiB reserve");
        const auto now=[] {return std::chrono::steady_clock::now();};const auto start=now();
        const auto ms=[&] {return std::chrono::duration<double,std::milli>(now()-start).count();};
        Set registry(bytes,8ULL<<30);std::vector<std::shared_ptr<Block>> blocks;blocks.reserve(bytes/Block::bytes);
        double lock_ms=0;
        for(std::uint64_t offset=0;offset<bytes;offset+=Block::bytes) {
            blocks.push_back(std::make_shared<Block>(offset/4096));
            if((offset+Block::bytes)%(4ULL<<30)==0 || offset+Block::bytes==bytes) {
                Set::Snapshot snapshot;snapshot.owners.reserve(blocks.size());snapshot.regions.reserve(blocks.size());
                for(const auto& block:blocks) {snapshot.owners.push_back(block);snapshot.add(block->data,Block::bytes);}
                const auto stats=registry.replace(std::move(snapshot));lock_ms+=stats.elapsed_ms;
                std::cout << "HOST_RESIDENT_SCALE_PROGRESS bytes=" << offset+Block::bytes << " locked=" << stats.page_bytes
                    << " update_ms=" << stats.elapsed_ms << " wall_ms=" << ms() << std::endl;
            }
        }
        require(registry.locked_page_bytes()==bytes,"scale lock extent");
        require(EmptyWorkingSet(GetCurrentProcess())!=0,"scale own trim");
        std::array<PSAPI_WORKING_SET_EX_INFORMATION,4096> query{};std::array<std::uint64_t,4096> expected{};
        std::size_t count=0;std::uint64_t witnessed=0,page_id=0;
        const auto flush=[&] {
            require(QueryWorkingSetEx(GetCurrentProcess(),query.data(),static_cast<DWORD>(count*sizeof(query[0])))!=0,"scale query");
            for(std::size_t i=0;i<count;++i) {
                require(query[i].VirtualAttributes.Valid && query[i].VirtualAttributes.Locked,"scale trim lost resident lock");
                require(*static_cast<const std::uint64_t*>(query[i].VirtualAddress)==expected[i],"scale distinct page data changed");
                witnessed+=4096;
            }
            count=0;
        };
        for(const auto& block:blocks) for(std::size_t offset=0;offset<Block::bytes;offset+=4096) {
            query[count]={};query[count].VirtualAddress=block->data+offset;expected[count]=++page_id;
            if(++count==query.size()) flush();
        }
        if(count) flush();require(witnessed==bytes,"scale witness extent");
        PROCESS_MEMORY_COUNTERS_EX process{};process.cb=sizeof(process);
        require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&process),sizeof(process))!=0,"scale process resident bytes");
        require(process.WorkingSetSize>=bytes,"scale physical working set below unique payload");
        require(GlobalMemoryStatusEx(&memory)!=0 && memory.ullAvailPhys>=(8ULL<<30),"scale final8GiB reserve");
        const auto physical=process.WorkingSetSize,available=memory.ullAvailPhys;
        const auto close_start=now();registry.close();blocks.clear();
        const auto close_ms=std::chrono::duration<double,std::milli>(now()-close_start).count();
        std::cout << "HOST_RESIDENT_SCALE PASS bytes=" << bytes << " unique_pages=" << page_id << " witnessed_locked_bytes=" << witnessed
            << " process_working_set=" << physical << " available_host=" << available << " lock_update_ms=" << lock_ms
            << " close_ms=" << close_ms << " wall_ms=" << ms() << " inference_capacity_claim=0\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << "HOST_RESIDENT_SCALE FAIL " << e.what() << '\n';return 1;}
}
