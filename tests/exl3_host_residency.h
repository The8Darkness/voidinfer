#pragma once

// Windows read-only page probe. No touching, locking, trimming or prefaulting:
// QueryWorkingSetEx observes residency at sample time, not a permanent guarantee.
// https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-queryworkingsetex
struct Exl3HostResidency {
    std::uint64_t allocated_union_bytes=0,resident_tensor_bytes=0,locked_tensor_bytes=0;
    std::uint64_t page_coverage_bytes=0,resident_page_bytes=0,query_calls=0;
    double elapsed_ms=0;
};

class Exl3HostResidencyProbe {
    using Range=std::pair<std::uintptr_t,std::uintptr_t>;
    std::vector<Range> ranges_;
public:
    void add(const void* pointer,std::size_t bytes) {
        if(!bytes) return;
        const auto first=reinterpret_cast<std::uintptr_t>(pointer);
        require(first && bytes<=std::numeric_limits<std::uintptr_t>::max()-first,"residency address overflow");
        ranges_.emplace_back(first,first+bytes);
    }
    Exl3HostResidency measure() const {
        const auto start=std::chrono::steady_clock::now();
        Exl3HostResidency result;
        auto sorted=ranges_;std::sort(sorted.begin(),sorted.end());
        std::vector<Range> bytes;
        for(const auto& range:sorted) {
            if(!bytes.empty() && range.first<=bytes.back().second) bytes.back().second=std::max(bytes.back().second,range.second);
            else bytes.push_back(range);
        }
        SYSTEM_INFO info{};GetSystemInfo(&info);const std::uintptr_t page=info.dwPageSize;
        require(page && (page&(page-1))==0,"residency page geometry");
        std::vector<Range> pages;
        for(const auto& range:bytes) {
            result.allocated_union_bytes+=range.second-range.first;
            require(range.second<=std::numeric_limits<std::uintptr_t>::max()-(page-1),"residency rounded address overflow");
            const Range rounded{range.first&~(page-1),(range.second+page-1)&~(page-1)};
            if(!pages.empty() && rounded.first<=pages.back().second) pages.back().second=std::max(pages.back().second,rounded.second);
            else pages.push_back(rounded);
        }
        std::array<PSAPI_WORKING_SET_EX_INFORMATION,4096> query{};
        std::array<std::uint64_t,4096> covered{};
        std::size_t count=0,cursor=0;
        const auto flush=[&] {
            if(!count) return;
            require(QueryWorkingSetEx(GetCurrentProcess(),query.data(),static_cast<DWORD>(count*sizeof(query[0])))!=0,
                "QueryWorkingSetEx failed");
            ++result.query_calls;
            for(std::size_t i=0;i<count;++i) {
                result.page_coverage_bytes+=page;
                if(query[i].VirtualAttributes.Valid) {
                    result.resident_page_bytes+=page;result.resident_tensor_bytes+=covered[i];
                    if(query[i].VirtualAttributes.Locked) result.locked_tensor_bytes+=covered[i];
                }
            }
            count=0;
        };
        for(const auto& range:pages) for(auto address=range.first;address<range.second;address+=page) {
            while(cursor<bytes.size() && bytes[cursor].second<=address) ++cursor;
            std::uint64_t payload=0;
            for(auto j=cursor;j<bytes.size() && bytes[j].first<address+page;++j)
                payload+=std::min(bytes[j].second,address+page)-std::max(bytes[j].first,address);
            query[count]={};query[count].VirtualAddress=reinterpret_cast<void*>(address);covered[count]=payload;
            if(++count==query.size()) flush();
        }
        flush();
        result.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        return result;
    }
};

void qualify_residency_page_probe() {
    SYSTEM_INFO info{};GetSystemInfo(&info);const std::size_t page=info.dwPageSize;
    auto* allocation=static_cast<unsigned char*>(VirtualAlloc(nullptr,3*page,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
    require(allocation!=nullptr,"residency fixture allocation");
    try {
        std::memset(allocation,37,3*page);
        require(VirtualFree(allocation+page,page,MEM_DECOMMIT)!=0,"residency fixture decommit");
        Exl3HostResidencyProbe probe;
        probe.add(allocation+2*page+7,19);probe.add(allocation+7,3*page-14);probe.add(allocation+8,2);
        const auto result=probe.measure();
        require(result.allocated_union_bytes==3*page-14 && result.page_coverage_bytes==3*page &&
            result.resident_tensor_bytes==2*page-14 && result.resident_page_bytes==2*page,
            "residency overlap/nonresident oracle");
    } catch(...) {VirtualFree(allocation,0,MEM_RELEASE);throw;}
    require(VirtualFree(allocation,0,MEM_RELEASE)!=0,"residency fixture release");
}
