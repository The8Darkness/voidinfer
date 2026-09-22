#pragma once

void run_device_prefix_metadata_gate(const std::filesystem::path& output){
    using namespace ninfer::exl3;
    require(Exl3DevicePrefixCache::budget_snapshot()[0]==0,"device prefix initial budget");
    std::ofstream out(output);out<<"case,pass\n";
    const auto gate=[&](const char* name,bool pass){out<<name<<','<<pass<<'\n';out.flush();require(pass,std::string("device prefix ")+name);};
    std::array<std::unique_ptr<Exl3DevicePrefixCache>,5> owners;
    for(int i=0;i<5;++i)owners[i]=std::make_unique<Exl3DevicePrefixCache>();
    gate("four_admitted_fifth_exact_fallback",owners[0]->admitted()&&owners[1]->admitted()&&owners[2]->admitted()&&owners[3]->admitted()&&!owners[4]->admitted());
    gate("one_GiB_process_cap",Exl3DevicePrefixCache::budget_snapshot()[0]==Exl3DevicePrefixCache::cap);
    auto a=std::make_shared<Exl3ExactKVPage>();a->rows=64;
    auto b=std::make_shared<Exl3ExactKVPage>();b->first=64;b->rows=63;
    std::vector<std::shared_ptr<const Exl3ExactKVPage>> pages{a,b};
    const auto owner_count=a.use_count();owners[0]->publish_tags(pages,128);
    gate("weak_tags_do_not_change_COW_owners",a.use_count()==owner_count);
    gate("complete_only_partial_tail_excluded",owners[0]->matched_rows(pages,128)==64&&owners[0]->matched_rows(pages,63)==0);
    auto fork=std::make_shared<Exl3ExactKVPage>();fork->rows=64;
    gate("same_position_different_owner_refused",owners[0]->matched_rows({fork,b},128)==0);
    std::weak_ptr<const Exl3ExactKVPage> old=a;a.reset();pages[0]=fork;
    gate("expired_owner_not_kept_or_reused",old.expired()&&owners[0]->matched_rows(pages,128)==0);
    owners[0]->publish_tags(pages,64);owners[0]->invalidate();
    gate("invalidation_before_refill_or_reset",owners[0]->matched_rows(pages,128)==0);
    owners[0].reset();owners[4]=std::make_unique<Exl3DevicePrefixCache>();
    gate("released_budget_reacquired",owners[4]->admitted()&&Exl3DevicePrefixCache::budget_snapshot()[0]==Exl3DevicePrefixCache::cap);
    for(auto& owner:owners)owner.reset();
    gate("all_device_cache_bytes_retired",Exl3DevicePrefixCache::budget_snapshot()[0]==0&&Exl3DevicePrefixCache::budget_snapshot()[1]==0);
}

void run_device_prefix_16k_metadata_gate(const std::filesystem::path& output){
    using namespace ninfer::exl3;std::ofstream out(output);out<<"case,pass\n";
    const auto gate=[&](const char* name,bool pass){out<<name<<','<<pass<<'\n';out.flush();require(pass,std::string("device prefix16K ")+name);};
    gate("initial_empty",Exl3DevicePrefixCache::budget_snapshot()[0]==0);
    auto owner=std::make_unique<Exl3DevicePrefixCache>(16384);auto denied=std::make_unique<Exl3DevicePrefixCache>();
    gate("one_full_GiB_second_reserve_denied",owner->admitted()&&owner->allocation_bytes()==Exl3DevicePrefixCache::cap&&owner->capacity_tokens()==16384&&!denied->admitted()&&Exl3DevicePrefixCache::budget_snapshot()[0]==Exl3DevicePrefixCache::cap);
    constexpr std::uint64_t plane_bytes=16384ULL*1024*2;
    for(int bank=0;bank<16;++bank)for(int key=0;key<2;++key)for(int tail=0;tail<2;++tail){std::uint32_t value=1000+bank*4+key*2+tail;auto* p=static_cast<std::byte*>(owner->plane(bank,key!=0))+(tail?plane_bytes-4:0);cuda_check(cudaMemcpy(p,&value,4,cudaMemcpyHostToDevice),"16K plane boundary write");}
    for(int bank=0;bank<16;++bank)for(int key=0;key<2;++key)for(int tail=0;tail<2;++tail){std::uint32_t value=0;auto* p=static_cast<std::byte*>(owner->plane(bank,key!=0))+(tail?plane_bytes-4:0);cuda_check(cudaMemcpy(&value,p,4,cudaMemcpyDeviceToHost),"16K plane boundary read");gate("all_bank_extent_boundaries_distinct",value==1000+bank*4+key*2+tail);}
    std::vector<std::shared_ptr<const Exl3ExactKVPage>> pages;for(int i=0;i<256;++i){auto page=std::make_shared<Exl3ExactKVPage>();page->first=i*64;page->rows=64;pages.push_back(page);}
    const auto count=pages.back().use_count();owner->publish_tags(pages,16384);gate("all256_complete_weak_tags",owner->matched_rows(pages,16384)==16384&&pages.back().use_count()==count);
    std::weak_ptr<const Exl3ExactKVPage> old=pages.back();auto partial=std::make_shared<Exl3ExactKVPage>();partial->first=16320;partial->rows=63;pages.back()=partial;
    gate("last_page_expired_or_partial_refused",old.expired()&&owner->matched_rows(pages,16384)==16320);owner->publish_tags(pages,16384);gate("partial_tail_not_published",owner->matched_rows(pages,16384)==16320);
    owner->publish_tags(pages,16383,true);
    gate("partial_tail_identity_matches_exact_extent",
        owner->matched_rows(pages,16383,true)==16383);
    partial->rows=64;
    gate("in_place_extension_keeps_only_published_rows",
        owner->matched_rows(pages,16383,true)==16383 &&
        owner->matched_rows(pages,16384,true)==16320);
    owner->publish_tags(pages,16384,true);
    gate("completed_forward_published_page_matches",
        owner->matched_rows(pages,16384,true)==16384);
    partial->rows=62;
    gate("rollback_truncation_invalidates_longer_tag",
        owner->matched_rows(pages,16382,true)==16320);
    owner->invalidate();gate("reset_refill_invalidation",owner->matched_rows(pages,16384)==0);
    owner.reset();owner=std::make_unique<Exl3DevicePrefixCache>(16384);gate("full_reserve_reacquired",owner->admitted()&&Exl3DevicePrefixCache::budget_snapshot()[0]==Exl3DevicePrefixCache::cap);
    owner.reset();denied.reset();gate("all_retired",Exl3DevicePrefixCache::budget_snapshot()[0]==0&&Exl3DevicePrefixCache::budget_snapshot()[1]==0);
}
