#pragma once

// Included inside the acceptance harness's anonymous namespace. Test-only.
// Payload words retain their represented bits. Windows/x64 little endian is
// part of V1; headers use explicit little-endian integers, never native structs.
struct V1Clock {
    using Clock = std::chrono::steady_clock;
    enum Phase { local, reference_work, reference_download, candidate_work,
        candidate_download, record_write, record_read, comparison, ring, target_only, count };
    std::array<double,count> seconds{};
    Phase active=local;
    Clock::time_point last=Clock::now(), start=last;
    void select(Phase p) { auto now=Clock::now(); seconds[active]+=std::chrono::duration<double>(now-last).count();last=now;active=p; }
    struct Scope { V1Clock& c; Phase previous; Scope(V1Clock& c,Phase p):c(c),previous(c.active){c.select(p);} ~Scope(){c.select(previous);} };
    void save(const std::filesystem::path& path) {
        select(active);
        const char* names[]={"local_checks","reference_work","reference_download","candidate_work","candidate_download","record_write","record_read","byte_comparison","ring_replay","target_only"};
        std::ofstream f(path); f<<"phase,seconds\n"<<std::setprecision(17);
        for(int i=0;i<count;++i)f<<names[i]<<','<<seconds[i]<<'\n';
        f<<"state_total,"<<std::chrono::duration<double>(Clock::now()-start).count()<<'\n';
        require(f.good(),"V1 phase report write");
    }
};

struct V1Oracle {
#if defined(NINFER_V1_EXPORT)
    static constexpr bool exporting=true;
#else
    static constexpr bool exporting=false;
#endif
    V1Clock clock;
    std::fstream file;
    std::ofstream index;
    std::filesystem::path cache,report;
    std::uint64_t ordinal=0, payload_bytes=0, comparisons=0;
    bool finished=false;
    std::string event="initial";
    static std::uint64_t checksum(const void* data,std::size_t bytes) {
        auto p=static_cast<const unsigned char*>(data);std::uint64_t h=14695981039346656037ULL;
        for(std::size_t i=0;i<bytes;++i){h^=p[i];h*=1099511628211ULL;}return h;
    }
    void put64(std::uint64_t v){for(int i=0;i<8;++i)file.put(static_cast<char>((v>>(8*i))&255));require(file.good(),"V1 write header");}
    std::uint64_t get64(){std::uint64_t v=0;for(int i=0;i<8;++i){int c=file.get();require(c!=EOF,"V1 truncated header");v|=std::uint64_t(static_cast<unsigned char>(c))<<(8*i);}return v;}
    void text(const std::string& s){put64(s.size());file.write(s.data(),s.size());require(file.good(),"V1 write label");}
    std::string text(){auto n=get64();require(n<=4096,"V1 oversized label");std::string s(static_cast<std::size_t>(n),'\0');file.read(s.data(),s.size());require(file.good(),"V1 truncated label");return s;}
    explicit V1Oracle(const std::filesystem::path& report):cache(env("NINFER_V1_CACHE")),report(report) {
        require(!cache.empty() && std::filesystem::is_directory(cache),"V1 wrapper-owned cache required");
        const auto identity=env("NINFER_V1_IDENTITY");
        require(identity.size()==64 && identity.find_first_not_of("0123456789abcdefABCDEF")==std::string::npos,"V1 identity SHA256 required");
        const auto payload=cache/"records.bin";
        if(exporting){
            require(!std::filesystem::exists(payload) && !std::filesystem::exists(cache/"complete.json"),"V1 cannot overwrite payload");
            file.open(payload,std::ios::binary|std::ios::out);
            require(file.is_open(),"V1 open export");text("NINFER_V1_EXACT_LE_1");text(identity);
        }else{
            require(std::filesystem::exists(cache/"complete.json"),"V1 incomplete cache");
            file.open(payload,std::ios::binary|std::ios::in);require(file.is_open(),"V1 open consume");
            require(text()=="NINFER_V1_EXACT_LE_1" && text()==identity,"V1 schema/identity mismatch");
        }
        index.open(report/"records.csv");index<<"ordinal,event,field,layer,position,rows,columns,dtype,bytes,checksum\n";
    }
    ~V1Oracle(){try{
        clock.save(report/"phases.csv");
        PROCESS_MEMORY_COUNTERS_EX counters{};counters.cb=sizeof(counters);
        const bool available=GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),sizeof(counters))!=0;
        std::ofstream f(report/"process-memory.json");
        f<<"{\"available\":"<<(available?"true":"false")<<",\"peak_working_set_bytes\":"<<counters.PeakWorkingSetSize<<",\"peak_private_commit_pagefile_bytes\":"<<counters.PeakPagefileUsage<<"}\n";
    }catch(...){} }
    template<class T,class Fn> std::vector<T> value(const std::string& field,int layer,int position,
            std::uint64_t rows,std::uint64_t columns,const std::string& dtype,Fn source) {
        std::vector<T> v;
        if(exporting){V1Clock::Scope s(clock,V1Clock::reference_download);v=source();}
        V1Clock::Scope s(clock,exporting?V1Clock::record_write:V1Clock::record_read);
        // A zero columns value is permitted only for variable-length OSCAR bytes.
        require(columns!=0 || dtype=="OSCAR_LOGICAL_BF16_INT2_V1","V1 invalid dynamic extent");
        std::uint64_t n=exporting?v.size():0, h=exporting?checksum(v.data(),v.size()*sizeof(T)):0;
        if(exporting){
            require(columns==0 || n==rows*columns,"V1 export dimensions");
            put64(ordinal);text(event);text(field);put64(static_cast<std::uint64_t>(layer));put64(position);put64(rows);put64(columns);text(dtype);put64(sizeof(T));put64(n);put64(h);
            file.write(reinterpret_cast<const char*>(v.data()),n*sizeof(T));require(file.good(),"V1 payload write");
        }else{
            require(get64()==ordinal && text()==event && text()==field && get64()==static_cast<std::uint64_t>(layer) && get64()==static_cast<std::uint64_t>(position) && get64()==rows && get64()==columns && text()==dtype && get64()==sizeof(T),"V1 reordered/incorrect record "+event+"/"+field);
            n=get64();h=get64();require(n<=512ULL*1024*1024/sizeof(T) && (columns==0 || n==rows*columns),"V1 invalid payload extent "+field);
            v.resize(static_cast<std::size_t>(n));file.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));require(file.good(),"V1 truncated payload "+field);
            require(checksum(v.data(),n*sizeof(T))==h,"V1 corrupt payload "+field);
        }
        index<<ordinal<<','<<event<<','<<field<<','<<layer<<','<<position<<','<<rows<<','<<columns<<','<<dtype<<','<<n*sizeof(T)<<','<<h<<'\n';require(index.good(),"V1 record index write");
        ++ordinal;payload_bytes+=n*sizeof(T);return v;
    }
    template<class A,class B> void equal(const A& actual,const B& expected,const std::string& label) {
        V1Clock::Scope s(clock,V1Clock::comparison);
        require(actual.size()*sizeof(actual[0])==expected.size()*sizeof(expected[0]),"V1 length mismatch "+label);
        const auto* a=reinterpret_cast<const unsigned char*>(actual.data());const auto* b=reinterpret_cast<const unsigned char*>(expected.data());
        const auto bytes=actual.size()*sizeof(actual[0]);
        if(bytes && std::memcmp(a,b,bytes)!=0){std::size_t i=0;while(i<bytes && a[i]==b[i])++i;throw std::runtime_error("V1 mismatch "+label+" byte="+std::to_string(i)+" actual="+std::to_string(a[i])+" expected="+std::to_string(b[i]));}
        ++comparisons;
    }
    void finish(int total,int boundaries){
        require(boundaries==(total==512?31:27),"V1 boundary coverage");
        {V1Clock::Scope s(clock,exporting?V1Clock::record_write:V1Clock::record_read);
            if(exporting){text("END");put64(ordinal);file.flush();require(file.good(),"V1 flush");}
            else {require(text()=="END" && get64()==ordinal && file.peek()==EOF,"V1 extra/missing records");}
            file.close();index.close();
        }
        std::ofstream f(report/"oracle-result.json");f<<"{\"status\":\"PASS\",\"records\":"<<ordinal<<",\"comparisons\":"<<comparisons<<",\"payload_bytes\":"<<payload_bytes<<",\"boundaries\":"<<boundaries<<"}\n";require(f.good(),"V1 result write");finished=true;
    }
};

// Adapter exposes only the reference observations used by the unmodified live
// schedule. Consumer has no reference context and never calls the source lambdas.
struct V1Reference {
    V1Oracle& oracle;
    std::unique_ptr<Exl3TextContext> context;
    int position_=0;
    V1Reference(Exl3TextModel& target,V1Oracle& o):oracle(o){if(V1Oracle::exporting){V1Clock::Scope s(o.clock,V1Clock::reference_work);context=target.create_context(true);}}
    bool try_enable_oscar_from_environment(){return !context || context->try_enable_oscar_from_environment();}
    void prefill(std::span<const std::int64_t> ids){if(context){V1Clock::Scope s(oracle.clock,V1Clock::reference_work);context->prefill(ids);cuda_check(cudaDeviceSynchronize(),"V1 reference prefill");}position_=static_cast<int>(ids.size());}
    void decode(std::int64_t token){if(context){V1Clock::Scope s(oracle.clock,V1Clock::reference_work);context->decode(token);cuda_check(cudaDeviceSynchronize(),"V1 reference decode");}++position_;}
    int position(){auto v=oracle.value<std::int32_t>("host_position",-1,position_,1,1,"I32",[&]{return std::vector<std::int32_t>{context->position()};});require(v[0]==position_,"V1 position schedule");return v[0];}
    int device_position_host(){return oracle.value<std::int32_t>("device_position",-1,position_,1,1,"I32",[&]{return std::vector<std::int32_t>{context->device_position_host()};})[0];}
    std::vector<float> gdn_state_host(int layer){return oracle.value<float>("gdn_state",layer,position_,48,128*128,"FP32_BITS",[&]{return context->gdn_state_host(layer);});}
    std::vector<std::uint16_t> gdn_physical_conv_host(int layer){return oracle.value<std::uint16_t>("physical_conv",layer,position_,10240,4,"BF16_BITS",[&]{return context->gdn_physical_conv_host(layer);});}
    std::vector<std::byte> oscar_live_state_host_for_test(){return oracle.value<std::byte>("oscar_live",-1,position_,1,0,"OSCAR_LOGICAL_BF16_INT2_V1",[&]{return context->oscar_live_state_host_for_test();});}
    std::vector<std::uint16_t> embedding_bits_host_for_test(){return oracle.value<std::uint16_t>("embedding",-1,position_,1,kHidden,"F16_BITS",[&]{return context->embedding_bits_host_for_test();});}
    struct Logits { V1Reference* reference; };
    Logits logits_device(){return {this};}
    ninfer::exl3::Exl3FullAttentionQKVHost full_attention_qkv_host(int layer){
        ninfer::exl3::Exl3FullAttentionQKVHost source,result;
        if(context){V1Clock::Scope s(oracle.clock,V1Clock::reference_download);source=context->full_attention_qkv_host(layer);require(source.rows==1,"V1 reference QKV rows");}
        result.layer=layer;result.rows=1;
        result.q_rope=oracle.value<std::uint16_t>("q_rope",layer,position_,1,6144,"F16_BITS",[&]{return source.q_rope;});
        result.k_rope=oracle.value<std::uint16_t>("k_rope",layer,position_,1,1024,"F16_BITS",[&]{return source.k_rope;});
        result.v_projection=oracle.value<std::uint16_t>("v_projection",layer,position_,1,1024,"F16_BITS",[&]{return source.v_projection;});return result;
    }
};
std::vector<std::uint16_t> target_continue_tap_bits(V1Reference& r,int layer,int rows){return r.oracle.value<std::uint16_t>("tap",layer,r.position_,rows,kHidden,"F16_BITS",[&]{return target_continue_tap_bits(*r.context,layer,rows);});}
std::vector<std::uint16_t> target_continue_device_bits(V1Reference::Logits handle,std::size_t count,const std::string&){auto& r=*handle.reference;return r.oracle.value<std::uint16_t>("logits",-1,r.position_,1,count,"F16_BITS",[&]{return target_continue_device_bits(r.context->logits_device(),count,"V1 reference logits");});}
void require_target_continue_state_equal(Exl3TextContext& a,V1Reference& b,const std::string& label,bool compare_taps){
    require(!compare_taps,"V1 state taps are separate records");
    require(a.position()==b.position() && a.device_position_host()==b.device_position_host(),label+" position mismatch");
    b.oracle.equal(target_continue_device_bits(a.logits_device(),kVocab,label),target_continue_device_bits(b.logits_device(),kVocab,label),label+" logits");
    for(int layer=0;layer<64;++layer)if((layer+1)%4!=0)b.oracle.equal(a.gdn_state_host(layer),b.gdn_state_host(layer),label+" gdn "+std::to_string(layer));
}
