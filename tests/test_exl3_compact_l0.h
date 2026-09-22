#pragma once

void run_compact_l0_qualification(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    require(target.max_context()==1024 && source.size()>600,"compact L0 fixture extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    auto legacy=target.create_context(true);
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    auto compact=target.create_context(true);
    require(compact->oscar_only_context() && compact->cache_staging_bytes()==256*4096,
        "compact L0 bounded allocation");
    bool uninitialized=false;
    try {compact->prefill(std::span<const std::int64_t>(source.data(),16));}
    catch(const std::exception&) {uninitialized=true;}
    require(uninitialized && compact->position()==0,"compact L0 silently ran ordinary fallback");
    require(legacy->try_enable_oscar_from_environment() && compact->try_enable_oscar_from_environment(),"compact OSCAR enable");
    for(auto* ctx:{legacy.get(),compact.get()}) {ctx->prepare_transaction();ctx->prepare_continuation(8);}
    int cases=0;
    for(int prefix:{63,319,575}) {
        const auto request=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),prefix));
        const auto warm=Exl3TextContext::make_turboangle_warm_pages(request->state(),std::min(64,prefix));
        for(int arm=0;arm<2;++arm) {
            for(auto* ctx:{legacy.get(),compact.get()}) ctx->restore_oscar_host_state(*request->state(),arm?warm.get():nullptr);
            require(prefix_retention_snapshot(*legacy)==prefix_retention_snapshot(*compact),"compact L0 restore full state");
            for(auto* ctx:{legacy.get(),compact.get()}) {
                ctx->begin_transaction();
                ctx->continue_rows(std::span<const std::int64_t>(source.data()+prefix,8));
            }
            require(legacy->continuation_logits_host()==compact->continuation_logits_host() &&
                prefix_retention_snapshot(*legacy)==prefix_retention_snapshot(*compact),"compact L0 attempted state");
            legacy->rollback_transaction();compact->rollback_transaction();
            require(prefix_retention_snapshot(*legacy)==prefix_retention_snapshot(*compact),"compact L0 rollback");
            for(int i=0;i<3;++i) {
                const auto token=sample_target(*legacy);
                legacy->decode(token);compact->decode(token);
                require(prefix_retention_snapshot(*legacy)==prefix_retention_snapshot(*compact),"compact L0 continuation");
            }
            bool authority=false;
            try {compact->export_exact_host_state();} catch(const std::exception&) {authority=true;}
            require(authority,"compact approximate state exported as authoritative");
            ++cases;
        }
    }
    std::cout << "COMPACT_L0 PASS cases=" << cases << " staging_bytes=" << compact->cache_staging_bytes()
        << " legacy_context_bytes=" << legacy->persistent_bytes() << " compact_context_bytes=" << compact->persistent_bytes()
        << " append_chunk=256 authoritative_export=forbidden\n";
}
