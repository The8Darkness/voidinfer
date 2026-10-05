#include "exl3/resource_inventory.h"
#include "exl3/merge_resident_ranges.h"
#include "exl3/retained_descriptor_ledger.h"
#include "exl3/bounded_shared_owner.h"
#include "exl3/embedded_shared_control.h"
#include <optional>
#include "exl3/host_kv_transfer_requirements.h"
#include "exl3/transfer_descriptor_storage.h"
#include "exl3/scatter_piece_storage.h"
#include "exl3/engine_scratch_requirements.h"
#include "exl3/resource_availability.h"
#include "exl3/linear_workspace_requirements.h"
#include "exl3/fixed_allocation_owners.h"
#include "exl3/exact_kv_extent.h"
#include "exl3/packed_cost_policy.h"
#include "exl3/output_delivery.h"
#include "exl3/engine_round_tokens.h"
#include "exl3/pending_payload_extent.h"
#include "exl3/reserved_host_payload_union.h"
#include "exl3/reserved_request_queue.h"
#include "exl3/result_control_allocator.h"
#include "exl3/request_option_storage.h"
#include "exl3/activation_lifetime.h"
#include "exl3/packed_gather_signature.h"
#include "exl3/draft_private_segment.h"
#include "exl3/kv_transfer_lease.h"
#include "exl3/device_page_key.h"
#include "exl3/device_page_fill.h"
#include "exl3/device_page_reader.h"
#include "exl3/attention_page_ranges.h"
#include "exl3/attention_history_coverage.h"
#include "exl3/attention_position_contract.h"
#include "exl3/attention_input_view.h"
#include "exl3/attention_causal_rows.h"
#include "exl3/attention_profile.h"
#include "exl3/attention_stage.h"
#include "exl3/attention_stage_history.h"
#include "attention_segment_reference.h"
#include "exl3/attention_scratch_lifetime.h"
#include "exl3/projection_lifetime.h"
#include "exl3/reconstruction_config.h"
#include "exl3/reconstruction_stream.h"
#include "exl3/reconstruction_device_retirement.h"
#include "exl3/reconstruction_control_allocator.h"
#include "exl3/paired_transform_extent.h"
#include "exl3/residual_norm_extent.h"
#include "exl3/quant_descriptor.h"
#include "exl3/borrowed_descriptor.h"
#include "exl3/prefill_chunk_contract.h"
#include "exl3/projection_graph_binding.h"
#include "exl3/head_consumer_plan.h"
#include <iostream>
using Inventory=ninfer::exl3::Exl3ResourceInventory;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){const char* phase="initial ownership/storage contracts";try{
    {
        using namespace ninfer::exl3;
        struct Payload {std::vector<int> values;explicit Payload(unsigned count):values(count,7){}};
        auto owner=make_bounded_shared<Payload>(8);
        std::weak_ptr<Payload> weak=owner;
        RetainedDescriptorLedger ledger;
        const auto dynamic=owner->values.capacity()*sizeof(int);
        const auto block=bounded_shared_allocation_bytes<Payload>();
        if(attach_bounded_split_retirement_credit<Payload>(owner,ledger.acquire(block+dynamic-1),dynamic) ||
            !attach_bounded_split_retirement_credit<Payload>(owner,ledger.acquire(block+dynamic),dynamic) ||
            attach_bounded_split_retirement_credit<Payload>(owner,ledger.acquire(block+dynamic),dynamic))
            throw std::runtime_error("bounded split exact/duplicate credit contract");
        auto copied=owner->values;
        owner.reset();
        if(!weak.expired() || ledger.bytes()!=block || copied.size()!=8 || copied.front()!=7)
            throw std::runtime_error("bounded split payload release lost weak allocation charge");
        weak.reset();
        if(ledger.bytes())throw std::runtime_error("bounded split final weak release leaked charge");
    }
    {
        auto owner=std::make_shared<int>();
        const auto attach=+[](const std::shared_ptr<const void>&,ninfer::exl3::RetainedCudaRegistrationLedger::Ticket) noexcept {return true;};
        Inventory::Allocation allocation{owner,0,Inventory::Domain::cuda_registered_host,64,{},nullptr,nullptr,attach};
        Inventory inventory;inventory.add(allocation);inventory.add(allocation);
        require(inventory.first_lifetime_retirement()->registration_lifetime_retire==attach,
            "registration lifetime hook omitted from retirement discovery");
        auto changed=allocation;changed.registration_lifetime_retire=nullptr;
        bool refused=false;try{inventory.add(changed);}catch(const std::invalid_argument&){refused=true;}
        require(refused,"registration hook change accepted for existing owner slot");
        refused=false;try{inventory.without_allocation(changed);}catch(const std::invalid_argument&){refused=true;}
        require(refused,"registration removal ignored hook identity");
        changed=allocation;changed.domain=Inventory::Domain::device;
        refused=false;try{Inventory invalid;invalid.add(changed);}catch(const std::invalid_argument&){refused=true;}
        require(refused,"registration credit attached to device domain");
        require(!inventory.without_allocation(allocation).first_lifetime_retirement(),"registration lifetime removal retained owner");
    }
    {
        struct Piece {std::shared_ptr<int> owner;std::size_t offset=0;};
        using Storage=ninfer::exl3::Exl3ScatterPieceStorage<Piece,2>;
        static_assert(!std::is_copy_constructible_v<Storage> && !std::is_move_constructible_v<Storage>);
        Storage pieces;auto first=std::make_shared<int>(1),second=std::make_shared<int>(2);
        std::weak_ptr<int> first_weak=first,second_weak=second;
        pieces.push_back({first,17});pieces.push_back({second,29});
        first.reset();second.reset();
        require(!first_weak.expired() && !second_weak.expired(),"scatter slot lost retained owners");
        const auto* address=pieces.begin();
        bool refused=false;
        try{pieces.push_back({std::make_shared<int>(3),41});}catch(const std::length_error&){refused=true;}
        require(refused && pieces.size()==2 && pieces.begin()==address &&
            pieces.begin()[0].offset==17 && pieces.begin()[1].offset==29,"scatter saturation changed active records");
        pieces.clear();
        require(pieces.empty() && pieces.begin()==pieces.end() && first_weak.expired() && second_weak.expired(),
            "scatter clear retained inactive page owners");
        pieces.push_back({std::make_shared<int>(4),53});
        require(pieces.begin()==address && pieces.size()==1 && pieces.begin()->offset==53,
            "scatter slot reuse moved inline records");
        pieces.clear();pieces.clear();
        require(pieces.empty(),"repeated scatter clear changed empty state");
        ninfer::exl3::Exl3ScatterPieceStorage<Piece,512> full_slot;
        auto page=std::make_shared<int>(5);std::weak_ptr<int> page_weak=page;
        for(std::size_t row=0;row<512;++row)full_slot.push_back({page,row*2048});
        page.reset();
        std::size_t represented_bytes=0;
        for(const auto& piece:full_slot) {
            require(piece.offset==represented_bytes && !page_weak.expired(),"full scatter slot reordered or lost owner");
            represented_bytes+=2048;
        }
        require(represented_bytes==(1u<<20),"one-row scatter pieces do not cover full chunk");
        refused=false;
        try{full_slot.push_back({std::make_shared<int>(6),1u<<20});}
        catch(const std::length_error&){refused=true;}
        require(refused && full_slot.size()==512,"full scatter chunk exceeded prepared piece bound");
        full_slot.clear();require(page_weak.expired(),"full scatter slot clear leaked repeated page owner");
    }
    {
        using Storage=ninfer::exl3::Exl3TransferDescriptorStorage<std::size_t>;
        static_assert(!std::is_move_constructible_v<Storage> && !std::is_copy_constructible_v<Storage>);
        Storage values;
        values.reserve(0);
        require(values.empty() && values.capacity()==0 && !values.data(),"empty descriptor storage allocated");
        bool refused=false;
        try{values.push_back(9);}catch(const std::length_error&){refused=true;}
        require(refused && values.empty(),"unreserved descriptor append succeeded");
        values.reserve(2);auto* address=values.data();
        values.push_back(17);values.push_back(29);
        refused=false;try{values.push_back(41);}catch(const std::length_error&){refused=true;}
        require(refused && values.size()==2 && values.data()==address &&
            address[0]==17 && address[1]==29,"saturated descriptor append mutated storage");
        refused=false;try{values.reserve(3);}catch(const std::logic_error&){refused=true;}
        require(refused && values.capacity()==2 && values.data()==address,"descriptor growth relocated storage");
        using Use=ninfer::exl3::Exl3TransferDescriptorUse;
        static_assert(!std::is_copy_constructible_v<Use> && !std::is_move_constructible_v<Use>);
        Use use;unsigned fences=0;
        require(!use.retire_before_reuse([&]{++fences;}) && fences==0,
            "idle descriptor reuse manufactured a completion fence");
        use.begin_submission();
        refused=false;try{use.begin_submission();}catch(const std::logic_error&){refused=true;}
        require(refused && use.pending(),"in-flight descriptor table accepted a second submission");
        for(unsigned attempt=0;attempt<2;++attempt) {
            bool failed=false;
            try {
                use.retire_before_reuse([&]{++fences;throw std::runtime_error("prepared drain failure");});
                values.clear();
            } catch(const std::runtime_error&){failed=true;}
            require(failed && use.pending() && values.size()==2 && values.data()==address &&
                address[0]==17 && address[1]==29,
                "failed descriptor drain exposed storage for mutation");
        }
        require(use.retire_before_reuse([&]{++fences;}) && !use.pending() && fences==3,
            "successful descriptor drain did not release pending use");
        require(!use.retire_before_reuse([&]{++fences;}) && fences==3,
            "completed descriptor use repeated its fence");
        use.begin_submission();
        require(use.retire_before_reuse([&]{++fences;}) && fences==4,
            "second physical descriptor use inherited prior completion");
        values.clear();values.reserve(1);values.reserve(2);values.push_back(53);
        require(values.size()==1 && values.capacity()==2 && values.data()==address && address[0]==53,
            "descriptor clear/reuse changed address or capacity");
        Storage oversized;refused=false;
        try{oversized.reserve(Storage::max_size()+1);}catch(const std::length_error&){refused=true;}
        require(refused && oversized.empty() && !oversized.data(),"oversized descriptor plan allocated");
    }
    {
        using Plan=ninfer::exl3::Exl3HostKVTransferRequirements;
        {
            using ninfer::exl3::exl3_host_kv_append_download_interval;
            using ninfer::exl3::exl3_host_kv_append_download_page;
            int page_end=-1;
            require(exl3_host_kv_append_download_page(64,64,80,256,65536,65536,page_end) && page_end==128,
                "download page preflight refused represented partial tail");
            for(bool key_plane:{false,true}) {
                require(!exl3_host_kv_append_download_page(128,64,80,256,
                    key_plane?65535:65536,key_plane?65536:65535,page_end) && page_end==128,
                    "truncated download plane advanced accepted sequence");
            }
            require(!exl3_host_kv_append_download_page(129,64,80,256,65536,65536,page_end) && page_end==128,
                "download page gap advanced accepted sequence");
            require(exl3_host_kv_append_download_page(128,64,80,256,65536,65536,page_end) && page_end==192,
                "valid download page could not follow refused metadata");
            int end=-1;
            int suffix_end=80;
            require(!exl3_host_kv_append_download_page(81,47,80,256,47*1024,47*1024,suffix_end) && suffix_end==80,
                "download suffix accepted missing leading row");
            require(exl3_host_kv_append_download_page(0,64,80,256,65536,65536,suffix_end) && suffix_end==80,
                "old empty page changed required suffix start");
            require(exl3_host_kv_append_download_page(64,64,80,256,65536,65536,suffix_end) && suffix_end==128,
                "download suffix failed to consume partial first page");
            require(!exl3_host_kv_append_download_page(127,2,80,256,2048,2048,suffix_end) && suffix_end==128,
                "download suffix accepted repeated row across page boundary");
            require(exl3_host_kv_append_download_page(128,1,80,256,1024,1024,suffix_end) && suffix_end==129,
                "download suffix lost one-row final page");
            require(exl3_host_kv_append_download_interval(0,64,80,256,end) && end==-1,
                "empty download prefix advanced sequence");
            require(exl3_host_kv_append_download_interval(64,64,80,256,end) && end==128,
                "partial download page suffix refused");
            require(!exl3_host_kv_append_download_interval(129,32,80,256,end) && end==128,
                "download gap advanced sequence");
            require(!exl3_host_kv_append_download_interval(127,32,80,256,end) && end==128,
                "download overlap advanced sequence");
            require(exl3_host_kv_append_download_interval(128,64,80,256,end) && end==192,
                "download sequence could not retry after rejected interval");
            require(!exl3_host_kv_append_download_interval(192,65,80,256,end) && end==192,
                "oversized download page accepted");
            const int limit=std::numeric_limits<int>::max();
            end=-1;
            require(exl3_host_kv_append_download_interval(limit-64,64,limit-32,limit,end) && end==limit,
                "representable maximum download end refused");
            for(const auto& invalid:std::array<std::array<int,4>,6>{{
                {limit-63,64,limit-32,limit},{-1,1,0,256},{0,1,-1,256},
                {0,1,257,256},{0,0,0,256},{0,1,0,0}}}) {
                require(!exl3_host_kv_append_download_interval(invalid[0],invalid[1],invalid[2],invalid[3],end) &&
                    end==limit,"invalid download geometry changed prior end");
            }
        }
        {
            std::array<int,3> pieces{1,2,-1},destination{7,7,7};
            std::size_t writes=0;bool refused=false;
            const auto validate=[](int value){if(value<0)throw std::invalid_argument("late scatter descriptor");};
            const auto apply=[&](int value) noexcept {destination[writes++]=value;};
            try{ninfer::exl3::exl3_host_kv_scatter_batch(pieces,validate,apply);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused && writes==0 && destination==std::array<int,3>{7,7,7},
                "late scatter refusal modified earlier destinations");
            pieces[2]=3;
            ninfer::exl3::exl3_host_kv_scatter_batch(pieces,validate,apply);
            require(writes==3 && destination==std::array<int,3>{1,2,3},"scatter retry changed copy order");
        }
        using ninfer::exl3::exl3_host_kv_scatter_fits;
        require(exl3_host_kv_scatter_fits(128,64,64,0,64),"exact scatter tail refused");
        require(!exl3_host_kv_scatter_fits(128,65,64,0,64),"scatter destination tail overflow admitted");
        require(!exl3_host_kv_scatter_fits(128,64,64,1,64),"scatter staging tail overflow admitted");
        require(!exl3_host_kv_scatter_fits(128,0,64,0,0),"empty scatter admitted");
        const auto extreme=std::numeric_limits<std::size_t>::max();
        require(!exl3_host_kv_scatter_fits(128,extreme,64,0,2) &&
            !exl3_host_kv_scatter_fits(128,0,64,extreme,2),"wrapped scatter offset admitted");
        require(exl3_host_kv_scatter_fits(extreme,extreme-1,extreme,extreme-1,1) &&
            !exl3_host_kv_scatter_fits(extreme,extreme-1,extreme,extreme-1,2),
            "maximum representable scatter boundary misclassified");
        const auto metadata=static_cast<unsigned>(Inventory::Domain::host_metadata);
        const auto pinned=static_cast<unsigned>(Inventory::Domain::cuda_registered_host);
        for(const auto extent:std::array<std::pair<int,std::size_t>,5>{{{1,2},{63,2},{64,2},{65,4},{65536,2048}}})
            for(bool chunks:{false,true})for(bool d2h:{false,true}) {
                if(d2h && !chunks)continue;
                const auto plan=Plan::derive(extent.first,true,chunks,d2h,1u<<20);
                const auto count=chunks && d2h?0:extent.second;
                require(plan.descriptors==count,"HostKV tail descriptor count");
                require(plan.pinned_owners==(chunks?2u:0u),"HostKV pinned owner count");
                require(plan.resources.units[metadata]==count*(sizeof(void*)+sizeof(const void*)+sizeof(std::size_t)),"HostKV metadata requirement");
                require(plan.resources.units[pinned]==(chunks?2u<<20:0),"HostKV pinned domain requirement");
            }
        require(Plan::derive(64,false,false,false,0).resources.units==Inventory::Requirement{}.units,"disabled HostKV plan allocated");
        require(Plan::derive(64,false,true,true,1u<<20).pinned_owners==0,
            "disabled batch path charged unused pinned owners");
        const auto banked=Plan::derive(4352,true,true,true,1u<<20,true,512u<<10);
        require(banked.descriptors==0 && banked.pinned_owners==3 &&
            banked.resources.units[pinned]==(2u<<20)+(512u<<10),
            "HostKV banked D2H owner/registered-host requirement");
        const auto deep=Plan::derive(4352,true,true,true,1u<<20,true,512u<<10,32);
        require(deep.descriptors==0 && deep.pinned_owners==33 &&
            deep.resources.units[pinned]==(32u<<20)+(512u<<10),
            "HostKV deep H2D ring owner/registered-host requirement");
        bool refused=false;
        try{(void)Plan::derive(64,true,true,true,0);}catch(const std::invalid_argument&){refused=true;}
        require(refused,"HostKV empty pinned extent accepted");
        refused=false;
        try{(void)Plan::derive(64,true,true,true,1u<<20,true,0);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"HostKV empty banked D2H extent accepted");
    }
    {
        int borrowed=0;
        Inventory inventory;
        auto nonowning=std::shared_ptr<const void>(std::shared_ptr<const void>{},&borrowed);
        bool refused=false;
        try{inventory.add({nonowning,0,Inventory::Domain::host_metadata,sizeof(borrowed)});}
        catch(const std::invalid_argument&){refused=true;}
        require(refused && inventory.totals()==Inventory::Totals{},"inventory accepted non-owning allocation backing");
    }
    {
        using namespace ninfer::exl3;
        {
            int position=16;
            std::size_t remaining=1041;
            for(int rows:{1024,16,1}) {
                const auto step=exl3_prefill_chunk_step(1024,remaining,position,1057);
                require(step.first==position && step.rows==rows && step.next_position==position+rows,
                    "prefill tail skipped or duplicated absolute rows");
                position=step.next_position;remaining-=rows;
            }
            require(position==1057 && remaining==0,"prefill tail failed exact capacity boundary");
            require(exl3_prefill_chunk_step(8,0,1057,1057).rows==0,"empty completed suffix planned work");
            for(const auto& bad:std::array<std::array<int,3>,4>{{{0,16,1},{17,16,0},{16,16,1},{1,16,16}}}) {
                bool refused=false;
                try{(void)exl3_prefill_chunk_step(8,bad[2],bad[0],bad[1]);}
                catch(const std::invalid_argument&){refused=true;}
                require(refused,"invalid prefill position/capacity admitted");
            }
        }
        for(const auto& boundary:std::array<std::array<int,2>,8>{{{0,0},{1,1},{9,8},{17,16},{33,32},{129,128},{1023,128},{1024,1024}}})
            require(exl3_next_prefill_rows(1024,boundary[0])==boundary[1],"prefill partition boundary changed");
        require(exl3_prefill_partition_relation(128,1024,1023)==Exl3PrefillPartitionRelation::identical,
            "identical bounded suffix partitions misclassified");
        require(exl3_prefill_partition_relation(128,1024,1024)==Exl3PrefillPartitionRelation::requires_numerical_qualification,
            "changed recurrent/projection partition declared exact");
        for(int width:{0,1,7,64,256,512,-1}) {
            bool refused=false;
            try{(void)exl3_next_prefill_rows(width,1024);}catch(const std::invalid_argument&){refused=true;}
            require(refused,"unsupported production prefill menu admitted");
        }
    }
    {
        int collection_a=0,collection_b=0;
        const int descriptor=7;
        ninfer::exl3::Exl3BorrowedDescriptor<int> borrowed(&collection_a,descriptor);
        const auto copy=borrowed;
        require(&borrowed.get(&collection_a)==&descriptor && &copy.get(&collection_a)==&descriptor,
            "borrowed descriptor duplicated retained metadata");
        for(const void* owner:{static_cast<const void*>(&collection_b),static_cast<const void*>(nullptr)}) {
            bool refused=false;
            try{(void)borrowed.get(owner);}catch(const std::invalid_argument&){refused=true;}
            require(refused,"foreign descriptor collection admitted");
        }
    }
    for(int bits:{5,6,7,8})
        require(ninfer::exl3::exl3_native_bits_from_tile_extent(std::uint64_t(bits)*16)==bits,
            "canonical trellis descriptor bits");
    for(std::uint64_t extent:std::array<std::uint64_t,8>{0,64,79,81,129,144,
        ((std::uint64_t{1}<<32)+5)*16,std::numeric_limits<std::uint64_t>::max()}) {
        bool refused=false;
        try{(void)ninfer::exl3::exl3_native_bits_from_tile_extent(extent);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"invalid or narrowing-overflow quant descriptor admitted");
    }
    {
        auto old_owner=std::make_shared<int>(1),tracker=std::make_shared<int>(2);
        std::weak_ptr<int> retired=old_owner;
        Inventory initial;
        Inventory::Allocation metadata{old_owner,1,Inventory::Domain::host_metadata,128};
        const Inventory::Allocation tracking{tracker,3,Inventory::Domain::host_metadata,128};
        const Inventory::Allocation device{old_owner,0,Inventory::Domain::device,4096};
        initial.add(metadata);initial.add(device);
        bool premature=false;try{initial.transfer_metadata(metadata,tracking);}
        catch(const std::invalid_argument&){premature=true;}
        require(premature,"metadata tracking dropped owner before device retirement");
        auto metadata_only=initial.without_allocation(device);
        auto transferred=metadata_only.transfer_metadata(metadata,tracking);
        require(transferred.totals()==metadata_only.totals(),"metadata tracking released allocation credit");
        const auto active_attribution=metadata_only.attribution_snapshot();
        const auto retired_attribution=transferred.attribution_snapshot();
        require(active_attribution.size()==1 &&
            active_attribution.front().reason==Inventory::AttributionReason::live_owner &&
            !active_attribution.front().owner.expired() &&
            retired_attribution.size()==1 &&
            retired_attribution.front().reason==Inventory::AttributionReason::retired_owner_pending &&
            !retired_attribution.front().owner.expired() &&
            !retired_attribution.front().retired_owner.expired(),
            "inventory attribution lost live/retired owner identity or lifetime reason");
        const auto released=transferred.without_allocation(tracking).attribution_snapshot();
        require(released.empty(),"released allocation remained in attribution snapshot");
        auto wrong=tracking;++wrong.units;
        bool mismatch=false;try{metadata_only.transfer_metadata(metadata,wrong);}
        catch(const std::invalid_argument&){mismatch=true;}
        require(mismatch,"metadata tracking changed physical charge");
        Inventory duplicate=metadata_only;duplicate.add(tracking);
        bool merged=false;try{duplicate.transfer_metadata(metadata,tracking);}
        catch(const std::invalid_argument&){merged=true;}
        require(merged,"metadata tracking merged independent retired charges");
        require(!retired.expired(),"live metadata owner unexpectedly expired");
        require(!transferred.metadata_owner_expired(tracking,retired),"live retired metadata reclaimed");
        std::weak_ptr<const void> unrelated;
        {auto other=std::make_shared<int>(9);unrelated=other;}
        bool foreign=false;try{(void)transferred.metadata_owner_expired(tracking,unrelated);}
        catch(const std::invalid_argument&){foreign=true;}
        require(foreign,"expired unrelated owner released live metadata credit");
        auto replacement=tracking;replacement.slot=4;
        bool rebound=false;try{(void)transferred.transfer_metadata(tracking,replacement);}
        catch(const std::invalid_argument&){rebound=true;}
        require(rebound,"retired identity rebound to tracking owner");
        Inventory expired_inventory;
        std::weak_ptr<const void> expected_expired;
        std::shared_ptr<const void> survivor;
        {
            auto owner=std::make_shared<int>(7);survivor=owner;expected_expired=owner;
            Inventory source;const Inventory::Allocation payload{owner,0,Inventory::Domain::host_metadata,128};
            source.add(payload);expired_inventory=source.transfer_metadata(payload,replacement);
        }
        require(!expired_inventory.metadata_owner_expired(replacement,expected_expired),"surviving borrower ignored");
        survivor.reset();
        require(expired_inventory.metadata_owner_expired(replacement,expected_expired),"matching expired owner not collectible");
    }
    {
        auto retired_owner=std::make_shared<int>(1);
        auto tracker=std::make_shared<int>(2);
        auto quarantined_owner=std::make_shared<int>(3);
        const Inventory::Allocation retired{retired_owner,7,Inventory::Domain::host_metadata,128};
        const Inventory::Allocation tracking{tracker,41,Inventory::Domain::host_metadata,128};
        Inventory live;live.add(retired);
        auto pending=live.transfer_metadata(retired,tracking);
        Inventory proposed;proposed.add(retired);
        proposed.add({quarantined_owner,9,Inventory::Domain::device,64});
        Inventory quarantined;quarantined.add({quarantined_owner,9,Inventory::Domain::device,64});
        const auto mixed=Inventory::transition_union({},proposed,pending,quarantined);
        require(mixed[static_cast<unsigned>(Inventory::Domain::host_metadata)]==128 &&
            mixed[static_cast<unsigned>(Inventory::Domain::device)]==64,
            "mixed live/pending/quarantined aliases were charged more than once");
        auto limits=Inventory::unlimited();
        limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=128;
        limits[static_cast<unsigned>(Inventory::Domain::device)]=64;
        require(Inventory::peak({},proposed,pending,quarantined,limits)==mixed,
            "transition peak diverged from complete owner union");
        Inventory mismatched;
        auto changed=tracking;changed.slot=42;changed.units=129;
        changed.retired_owner=retired_owner;changed.retired_slot=retired.slot;
        mismatched.add(changed);
        bool refused=false;
        try{(void)Inventory::transition_union({},proposed,mismatched,{});}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"retired-owner alias changed physical extent without refusal");
        proposed={};retired_owner.reset();
        require(Inventory::transition_union({}, {},pending,{})[
            static_cast<unsigned>(Inventory::Domain::host_metadata)]==128,
            "expired retired identity released pending tracking charge");
        require(Inventory::transition_union({}, {},pending.without_allocation(tracking),{})==Inventory::Totals{},
            "explicit final-use release retained pending transition charge");
    }
    using Segment=ninfer::exl3::Exl3DraftPrivateSegment;
    const std::array<std::int64_t,8> ids{42,99,99,99,99,99,99,99};
    const auto segment=Segment::make(3,7,512,8,520,ids);
    auto mutable_ids=ids;
    const auto snapshot=Segment::make(3,7,512,8,520,mutable_ids);
    mutable_ids[3]=(mutable_ids[3]+1)%248320;
    require(snapshot.tokens==ids && snapshot.tokens!=mutable_ids,
        "proposal conditioning snapshot borrowed mutable token storage");
    require(segment.matches(3,7,512,8,520,ids[0]),"private segment rejected actual scope");
    require(segment.matches_tokens(ids),"private segment rejected unchanged full proposal");
    for(std::size_t row=1;row<ids.size();++row) {
        auto changed=ids;changed[row]=(changed[row]+1)%248320;
        require(segment.matches(3,7,512,8,520,changed[0]) && !segment.matches_tokens(changed),
            "unchanged seed concealed changed private proposal row");
    }
    require(!segment.matches_tokens(std::span(ids).first(7)) && !segment.matches_tokens({}),
        "private proposal token comparison accepted truncated extent");
    require(!segment.matches(4,7,512,8,520,ids[0]) && !segment.matches(3,8,512,8,520,ids[0]) &&
        !segment.matches(3,7,511,8,520,ids[0]) && !segment.matches(3,7,512,7,520,ids[0]) &&
        !segment.matches(3,7,512,8,521,ids[0]) && !segment.matches(3,7,512,8,520,(ids[0]+1)%248320),
        "private segment accepted changed conditioning scope");
    require(segment.tokens==ids && segment.positions[0]==520 && segment.positions[7]==527,
        "private draft row descriptor lost seed/masks/positions");
    const auto peer=Segment::make(4,9,1000,16,1016,ids);
    require(segment.valid() && peer.valid(),"actual B8 segments invalid");
    auto malformed=segment;std::swap(malformed.positions[1],malformed.positions[7]);
    require(!malformed.valid(),"draft segment accepted swapped position rows");
    malformed=segment;malformed.tokens[4]=-1;
    require(!malformed.valid(),"draft segment checked seed but ignored middle token");
    malformed=segment;malformed.ring_base=std::numeric_limits<std::int64_t>::max();
    require(!malformed.valid(),"draft ring geometry overflow accepted");
    malformed=segment;malformed.positions[0]=std::numeric_limits<std::int32_t>::max();
    require(!malformed.valid(),"draft row position overflow accepted");
    require(peer.positions[0]==1016 && segment.positions[0]==520,"draft segments became one causal sequence");
    for(const int count:{0,1,2047}) {
        const auto start=std::numeric_limits<std::int32_t>::max()-7;
        const auto edge=Segment::make(3,7,static_cast<std::int64_t>(start)-count,count,start,ids);
        require(edge.valid() && edge.positions.back()==std::numeric_limits<std::int32_t>::max() &&
            edge.matches(3,7,static_cast<std::int64_t>(start)-count,count,start,ids[0]),
            "legal draft ring/position boundary rejected or truncated");
        for(std::size_t row=0;row<edge.positions.size();++row) {
            auto changed=edge;--changed.positions[row];
            require(!changed.valid(),"private segment ignored a changed row position at integer boundary");
        }
        bool next_refused=false;
        try{(void)Segment::make(3,7,static_cast<std::int64_t>(start)+1-count,count,start+1,ids);}
        catch(const std::invalid_argument&){next_refused=true;}
        require(next_refused && edge.valid(),"overflowing next proposal changed retained legal segment");
    }
    {
        const auto empty_ring=Segment::make(3,7,0,0,0,ids);
        require(empty_ring.valid() && empty_ring.positions.back()==7,
            "zero-position private segment lost legal empty-ring boundary");
        bool overfull=false;
        try{(void)Segment::make(3,7,0,2048,2048,ids);}catch(const std::invalid_argument&){overfull=true;}
        require(overfull && empty_ring.valid(),"overfull ring admitted or changed prior segment");
    }
    for(int fault=0;fault<4;++fault) {
        bool refused=false;auto invalid=ids;if(fault==3)invalid[7]=248320;
        try{Segment::make(fault==0?0:3,7,512,8,fault==1?521:520,
            std::span<const std::int64_t>(invalid.data(),fault==2?7:8));}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"invalid private draft scope admitted");
    }
    using Lifetime=ninfer::exl3::Exl3ActivationLifetime;
    Lifetime activation;
    Lifetime::Witness old_activation;
    {
        Lifetime::Scope scope(activation);old_activation=scope.witness();
        require(old_activation.current(),"active normalized value lost lifetime");
        bool overlap=false;try{Lifetime::Scope invalid(activation);}catch(const std::logic_error&){overlap=true;}
        require(overlap && old_activation.current(),"overlapping producer changed activation generation");
    }
    require(!old_activation.current(),"residual boundary retained activation witness");
    {
        Lifetime::Scope next(activation);
        require(next.witness().current() && !old_activation.current(),"same storage revived old activation");
    }
    Lifetime::Witness exceptional;
    try{Lifetime::Scope scope(activation);exceptional=scope.witness();throw 1;}catch(int){}
    require(!exceptional.current(),"exception retained immutable activation interval");
    using Gather=ninfer::exl3::Exl3PackedGatherSignature;
    Lifetime first_activation,second_activation;
    auto gather_model=std::make_shared<int>(1);
    auto first_owner=std::make_shared<int>(2),second_owner=std::make_shared<int>(3);
    std::array<std::uint16_t,16> packed_storage{},first_storage{},second_storage{};
    Gather gathered;
    {
        Lifetime::Scope first(first_activation),second(second_activation);
        Gather::Pair pair{{{1,1,1,gather_model,first_owner,first_owner,first_storage.data(),3,512,4,2,2,first.witness()},
                          {2,1,1,gather_model,second_owner,second_owner,second_storage.data(),3,520,4,2,2,second.witness()}}};
        for(auto& lane:pair)lane.contract="Engine/text/fp16/shared-target/M16";
        require(gathered.remember(pair,packed_storage.data(),16),"contract fixture gather install");
        auto changed_contract=pair;
        for(auto& lane:changed_contract)lane.contract="Engine/prepared-media/fp16/shared-target/M16";
        require(!gathered.consume(changed_contract,packed_storage.data(),16),
            "gather reused inputs after contract change with unchanged addresses and epochs");
        require(!gathered.consume(pair,packed_storage.data(),16),"contract mismatch retained old gather");
        auto mixed_contract=pair;mixed_contract[1].contract=changed_contract[1].contract;
        require(!gathered.remember(mixed_contract,packed_storage.data(),16),"gather installed mixed modality contracts");
        auto missing_contract=pair;missing_contract[0].contract={};
        require(!gathered.remember(missing_contract,packed_storage.data(),16),"gather installed missing contract");
        require(gathered.remember(changed_contract,packed_storage.data(),16) &&
            gathered.consume(changed_contract,packed_storage.data(),16),"same prepared contract lost gather reuse");
        require(gathered.remember(pair,packed_storage.data(),16),"live gather signature refused");
        auto reversed=pair;std::swap(reversed[0],reversed[1]);
        require(!gathered.consume(reversed,packed_storage.data(),16),"gather reused reversed packed order");
        require(!gathered.consume(pair,packed_storage.data(),16),"mismatch retained reusable signature");
        require(gathered.remember(pair,packed_storage.data(),16),"gather signature reverse-order reinstall");
        require(gathered.consume_order(reversed,packed_storage.data(),16)==-1,
            "Engine reverse-order gather was not identified");
        require(gathered.remember(pair,packed_storage.data(),16),"gather signature reinstall");
        require(gathered.consume(pair,packed_storage.data(),16),"matching immutable gather was lost");
        require(!gathered.consume(pair,packed_storage.data(),16),"gather signature reused twice");
        gathered.remember(pair,packed_storage.data(),16);
        auto stale=pair;++stale[0].execution;
        require(!gathered.consume(stale,packed_storage.data(),16),"gather crossed execution epoch");
        require(!gathered.remember(pair,packed_storage.data(),15),"gather accepted short packed extent");
        gathered.remember(pair,packed_storage.data(),16);
        auto changed_root=pair;changed_root[0].root=std::make_shared<int>(9);
        require(!gathered.consume(changed_root,packed_storage.data(),16),"gather crossed published root");
        gathered.remember(pair,packed_storage.data(),16);
        auto reloaded=pair;reloaded[0].model=std::make_shared<int>(1);
        require(!gathered.consume(reloaded,packed_storage.data(),16),"same-labelled reload reused old model gather");
        for(int owner_slot=0;owner_slot<3;++owner_slot) {
            require(gathered.remember(pair,packed_storage.data(),16),"owner identity fixture install");
            auto rebound=pair;
            auto& identity=owner_slot==0?rebound[0].model:owner_slot==1?rebound[0].root:rebound[0].input_owner;
            auto unrelated_owner=std::make_shared<int>(42);
            identity=std::shared_ptr<const void>(unrelated_owner,identity.get());
            require(!gathered.consume(rebound,packed_storage.data(),16),
                "same address with foreign ownership reused gathered rows");
        }
        auto aliased=pair;
        aliased[0].input_owner=std::shared_ptr<const void>(pair[0].input_owner,pair[0].input_owner.get());
        require(gathered.remember(pair,packed_storage.data(),16) &&
            gathered.consume(aliased,packed_storage.data(),16),"same-control-block alias lost gather reuse");
        auto foreign_model=pair;
        foreign_model[1].model=std::shared_ptr<const void>(std::make_shared<int>(42),gather_model.get());
        require(!gathered.remember(foreign_model,packed_storage.data(),16),
            "gather installed two model addresses with unrelated ownership");
        gathered.remember(pair,packed_storage.data(),16);
        auto controlled=pair;++controlled[0].position;
        require(!gathered.consume(controlled,packed_storage.data(),16),"control append reused previous position");
        gathered.remember(pair,packed_storage.data(),16);
        auto reacquired=pair;++reacquired[0].acquisition;
        require(!gathered.consume(reacquired,packed_storage.data(),16),"reacquired request reused old gather");
        gathered.remember(pair,packed_storage.data(),16);
        require(!gathered.consume(pair,first_storage.data(),16),"gather reused a conflicting physical buffer");
        gathered.remember(pair,packed_storage.data(),16);
        gathered.clear(); // intervening family owns and may overwrite packed input
        require(!gathered.consume(pair,packed_storage.data(),16),"intervening gather retained signature");
        {
            bool deleter_observed_empty=false;
            bool observe_retirement=false;
            Gather retiring_source;
            auto retiring_model=std::shared_ptr<int>(new int(17),[&](int* value) noexcept {
                delete value;
                if(observe_retirement)
                    deleter_observed_empty=retiring_source.consume_order(pair,packed_storage.data(),16)==0;
            });
            std::weak_ptr<int> retiring_weak=retiring_model;
            auto retiring_pair=pair;
            for(auto& lane:retiring_pair)lane.model=retiring_model;
            require(retiring_source.remember(retiring_pair,packed_storage.data(),16),"gather retirement fixture install");
            retiring_pair={};retiring_model.reset();
            {
                auto detached=std::exchange(retiring_source,Gather{});
                observe_retirement=true;
                require(!retiring_weak.expired() && !deleter_observed_empty,
                    "detaching gather destroyed its final model owner prematurely");
                require(retiring_source.consume_order(pair,packed_storage.data(),16)==0,
                    "detached gather remained visible to a new consumer");
            }
            require(retiring_weak.expired() && deleter_observed_empty,
                "detached gather release did not expose empty source to final deleter");
        }
        gathered.remember(pair,packed_storage.data(),16);
    }
    // A later scope cannot revive metadata recorded before a residual update.
    {
        Lifetime::Scope first(first_activation),second(second_activation);
        Gather::Pair next{{{1,1,1,gather_model,first_owner,first_owner,first_storage.data(),3,512,4,2,2,first.witness()},
                          {2,1,1,gather_model,second_owner,second_owner,second_storage.data(),3,520,4,2,2,second.witness()}}};
        for(auto& lane:next)lane.contract="Engine/text/fp16/shared-target/M16";
        require(!gathered.consume(next,packed_storage.data(),16),"gather survived activation mutation");
    }
    Lifetime canceled_activation;
    Lifetime::Scope surviving_scope(first_activation);
    Gather::Pair canceled_pair;
    {
        Lifetime::Scope canceled_scope(canceled_activation);
        canceled_pair={{{1,1,1,gather_model,first_owner,first_owner,first_storage.data(),3,512,4,2,2,surviving_scope.witness()},
                        {2,1,1,gather_model,second_owner,second_owner,second_storage.data(),3,520,4,2,2,canceled_scope.witness()}}};
        for(auto& lane:canceled_pair)lane.contract="Engine/text/fp16/shared-target/M16";
        require(gathered.remember(canceled_pair,packed_storage.data(),16),"cancellation fixture signature");
    }
    require(!gathered.consume(canceled_pair,packed_storage.data(),16),"one canceled lane allowed gather reuse");
    using Cost=ninfer::exl3::Exl3PackedCostPolicy;
    Cost missing;
    require(!Cost::calibrated,"supplied cost estimates claimed calibration");
    require(!missing.permits(0,5120,12288,6,3,5,0),"missing underfill table did not fall back");
    require(!missing.may_wait(0,5120,12288,6,3),"missing underfill estimate waited for a peer");
    require(!missing.permits(Cost::family_count,5120,12288,6,8,8,0),"full batch bypassed family bounds");
    for(unsigned family:{13U,14U}) {
        require(missing.permits(family,5120,17408,6,8,8,0) &&
            !missing.permits(family,5120,17408,6,3,5,0),
            "draft gate/up cost policy lost full pair or invented underfill estimate");
        const std::array<Cost::Entry,1> draft_entry{{{family,5120,17408,6,3,5,100,40,30}}};
        const Cost draft_cost(draft_entry);
        require(draft_cost.permits(family,5120,17408,6,3,5,30) &&
            !draft_cost.permits(family==13?14:13,5120,17408,6,3,5,0),
            "draft gate/up estimates crossed distinct operator families");
    }
    require(missing.permits(0,5120,12288,6,8,8,0),"full M16 opt-in lost its route");
    const std::array<Cost::Entry,1> entries{{{7,17408,5120,6,3,5,100,40,30}}};
    Cost supplied(entries);
    require(supplied.may_wait(7,17408,5120,6,3),"supplied underfill entry cannot rendezvous");
    require(supplied.permits(7,17408,5120,6,3,5,30),"matching supplied cost not selected");
    require(!supplied.permits(7,17408,5120,6,3,5,31),"cost age limit exceeded");
    require(!supplied.permits(7,17408,5120,7,3,5,0),"cost reused across bitwidth");
    require(!supplied.permits(6,17408,5120,6,3,5,0),"cost reused across family");
    require(!supplied.permits(7,17408,5120,6,5,3,0),"cost assumed symmetric row ordering");
    require(!supplied.permits(7,17408,5120,6,8,8,51),"full batch ignored age bound");
    require(supplied.wait_budget_us(7,17408,5120,6,3)==30,"supplied age did not bound actual wait");
    require(missing.wait_budget_us(0,5120,12288,6,3)==0,"missing estimate budgeted a wait");
    require(missing.wait_budget_us(0,5120,12288,6,8)==50,"full pair wait budget changed");
    const std::string multibyte="a\xc3\xa9\xf0\x9f\x8c\x89z";
    using ninfer::exl3::output_result_slots_available;
    {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Inventory inventory;
        Inventory::Allocation allocation{std::make_shared<int>(5),0,Inventory::Domain::host_metadata,4,{},
            +[](const std::shared_ptr<const void>&,ninfer::exl3::RetainedDescriptorLedger::Ticket) noexcept {return false;}};
        inventory.add(allocation);const auto before=inventory.totals();
        auto changed=allocation;changed.lifetime_retire=nullptr;
        bool rebound=false,removed=false,domain=false;
        try{inventory.add(changed);}catch(const std::invalid_argument&){rebound=true;}
        try{(void)inventory.without_allocation(changed);}catch(const std::invalid_argument&){removed=true;}
        changed=allocation;changed.domain=Inventory::Domain::device;
        try{Inventory invalid;invalid.add(changed);}catch(const std::invalid_argument&){domain=true;}
        require(rebound && removed && domain && inventory.totals()==before &&
            inventory.first_lifetime_retirement()->lifetime_retire==allocation.lifetime_retire,
            "retirement hook identity/domain mutation changed retained inventory");
    }
    {
        struct Owner {
            ninfer::exl3::EmbeddedSharedControl arena;
            int value=37;
            int* retired;
            explicit Owner(int& count):retired(&count) {}
            ~Owner(){++*retired;}
        };
        int retired=0,destroyed=0;
        auto backing=ninfer::exl3::make_bounded_shared<Owner>(retired);
        std::weak_ptr<Owner> weak_backing=backing;
        std::weak_ptr<int> weak_request;
        {
            std::shared_ptr<int> request(&backing->value,[&](int*) noexcept {++destroyed;},
                ninfer::exl3::EmbeddedControlAllocator<int>(backing,backing->arena));
            weak_request=request;
            bool reuse_refused=false;
            try{(void)backing->arena.allocate<unsigned char>(1);}catch(const std::bad_alloc&){reuse_refused=true;}
            require(reuse_refused && *request==37,"embedded control arena reuse damaged live request");
            backing.reset();
        }
        require(destroyed==1 && retired==0 && weak_request.expired() && !weak_backing.expired(),
            "embedded request weak tail lost backing owner after strong retirement");
        weak_request.reset();
        require(retired==1 && weak_backing.expired(),"embedded control retirement retained backing payload");
        weak_backing.reset();
        require(ninfer::exl3::bounded_shared_live_blocks_for_test<Owner>()==0,
            "embedded control retirement leaked bounded backing allocation");
        backing=ninfer::exl3::make_bounded_shared<Owner>(retired);
        bool refused=false;
        try{std::shared_ptr<int> failed(&backing->value,[&](int*) noexcept {++destroyed;},
            ninfer::exl3::EmbeddedControlAllocator<int>(backing,backing->arena,true));}
        catch(const std::bad_alloc&){refused=true;}
        require(refused && destroyed==2 && !backing->arena.used && retired==1,
            "embedded control injection consumed arena or mishandled payload cleanup");
        backing.reset();require(retired==2 && ninfer::exl3::bounded_shared_live_blocks_for_test<Owner>()==0,
            "embedded control refusal retained backing owner");
    }
    {
        struct alignas(64) Payload {
            int* destroyed;
            explicit Payload(int& value,bool fail=false):destroyed(&value) {
                if(fail)throw std::runtime_error("prepared bounded payload construction failure");
            }
            ~Payload(){++*destroyed;}
        };
        int destroyed=0;
        std::weak_ptr<Payload> weak;
        {
            auto owner=ninfer::exl3::make_bounded_shared<Payload>(destroyed);
            require(reinterpret_cast<std::uintptr_t>(owner.get())%alignof(Payload)==0 &&
                ninfer::exl3::bounded_shared_live_blocks_for_test<Payload>()==1 &&
                ninfer::exl3::bounded_shared_allocation_bytes<Payload>()>=sizeof(Payload)+256,
                "bounded owner lost payload alignment or control storage extent");
            weak=owner;
        }
        require(destroyed==1 && weak.expired() && ninfer::exl3::bounded_shared_live_blocks_for_test<Payload>()==1,
            "bounded shared owner freed control backing before last weak reference");
        weak.reset();
        require(ninfer::exl3::bounded_shared_live_blocks_for_test<Payload>()==0,
            "bounded shared owner did not release last weak backing");
        bool payload_failed=false;
        try{(void)ninfer::exl3::make_bounded_shared<Payload>(destroyed,true);}
        catch(const std::runtime_error& error){payload_failed=std::string_view(error.what())==
            "prepared bounded payload construction failure";}
        require(payload_failed && destroyed==1 && ninfer::exl3::bounded_shared_live_blocks_for_test<Payload>()==0,
            "bounded payload constructor failure leaked backing or destroyed incomplete payload");
        bool control_failed=false;
        try{(void)ninfer::exl3::make_bounded_shared<Payload,1>(destroyed);}
        catch(const std::bad_alloc&){control_failed=true;}
        require(control_failed && destroyed==2 && ninfer::exl3::bounded_shared_live_blocks_for_test<Payload,1>()==0,
            "bounded control allocation refusal leaked or double-destroyed payload");
    }
    {
        using Ledger=ninfer::exl3::RetainedDescriptorLedger;
        {
            using PayloadLedger=ninfer::exl3::RetainedHostAllocationLedger;
            using Counter=PayloadLedger::State;
            Ledger metadata;
            std::optional<PayloadLedger::Ticket> payload;
            std::weak_ptr<Counter> weak;
            const auto bytes=ninfer::exl3::bounded_shared_allocation_bytes<Counter>();
            {
                auto credit=metadata.acquire(bytes);
                auto counter=ninfer::exl3::make_bounded_shared<Counter>();
                require(ninfer::exl3::attach_bounded_retirement_credit<Counter>(counter,std::move(credit)),
                    "reserved payload counter refused metadata attachment");
                weak=counter;
                PayloadLedger ledger(counter);
                payload.emplace(ledger.acquire(31));
                counter.reset();
                require(metadata.bytes()==bytes && ledger.bytes()==31,
                    "reserved counter lost metadata or payload credit");
            }
            require(!weak.expired() && metadata.bytes()==bytes,
                "payload ticket failed to retain reserved counter after registry retirement");
            payload.reset();
            require(weak.expired() && metadata.bytes()==bytes,
                "final strong counter retirement released final-weak metadata early");
            weak.reset();
            require(metadata.bytes()==0,"final weak counter retirement retained metadata credit");
            bool missing=false;
            try{PayloadLedger absent{std::shared_ptr<Counter>{}};}catch(const std::invalid_argument&){missing=true;}
            require(missing,"payload ledger accepted an absent reserved counter");
        }
        std::optional<Ledger::Ticket> survivor;
        std::weak_ptr<const void> state;
        {
            Ledger ledger;state=ledger.lifetime_for_test();
            survivor.emplace(ledger.acquire(17));
            {
                auto ticket=ledger.acquire(23);auto moved=std::move(ticket);
                require(ticket.bytes()==0 && moved.bytes()==23 && ledger.bytes()==40,
                    "descriptor credit move released or duplicated outstanding bytes");
            }
            require(ledger.bytes()==17,"descriptor credit destruction did not release exact bytes");
            bool overflow=false;
            try{(void)ledger.acquire(std::numeric_limits<std::uint64_t>::max());}
            catch(const std::overflow_error&){overflow=true;}
            require(overflow && ledger.bytes()==17,"descriptor ledger overflow changed live credit");
            {
                auto parent=ledger.acquire(29);
                require(!parent.split(0) && !parent.split(30) && parent.bytes()==29 && ledger.bytes()==46,
                    "invalid descriptor split consumed or expanded credit");
                auto child=parent.split(11);
                require(child && child->bytes()==11 && parent.bytes()==18 && ledger.bytes()==46,
                    "descriptor split did not preserve total live credit");
                auto tail=parent.split(18);
                require(tail && parent.bytes()==0 && !parent.split(1) && ledger.bytes()==46,
                    "full descriptor split left duplicate parent credit");
                child.reset();require(ledger.bytes()==35,"child retirement released another owner's credit");
                auto moved=std::move(*tail);tail.reset();
                require(moved.bytes()==18 && ledger.bytes()==35,"split ticket move changed retained credit");
            }
            require(ledger.bytes()==17,"split descendants failed to retire exact credit");
        }
        require(!state.expired() && survivor->bytes()==17,"descriptor credit depended on destroyed registry ledger");
        survivor.reset();require(state.expired(),"last descriptor ticket retained ledger state");
    }
    {
        using Range=ninfer::exl3::ResidentRange;
        const std::array<Range,3> ranges{{{0,32},{40,41},{64,81}}};
        require(ninfer::exl3::resident_lock_chunk_count(ranges,16)==5 &&
            ninfer::exl3::resident_lock_chunk_count({},16)==0,
            "resident completed-lock count lost exact or partial chunks");
        const auto top=std::numeric_limits<std::uintptr_t>::max();
        const std::array<Range,1> maximal{{{0,top}}};
        require(ninfer::exl3::resident_lock_chunk_count(maximal,top)==1,
            "resident lock chunk rounding overflowed maximum endpoint");
        bool zero_refused=false;
        try{(void)ninfer::exl3::resident_lock_chunk_count(ranges,0);}
        catch(const std::invalid_argument&){zero_refused=true;}
        const std::array<Range,1> empty_range{{{7,7}}};bool extent_refused=false;
        try{(void)ninfer::exl3::resident_lock_chunk_count(empty_range,16);}
        catch(const std::invalid_argument&){extent_refused=true;}
        const std::array<Range,2> excess{{{0,top},{0,1}}};bool count_refused=false;
        try{(void)ninfer::exl3::resident_lock_chunk_count(excess,1);}
        catch(const std::overflow_error&){count_refused=true;}
        require(zero_refused && extent_refused && count_refused,
            "resident completed-lock planning accepted invalid or overflowing demand");
    }
    {
        using Range=ninfer::exl3::ResidentRange;
        const auto top=std::numeric_limits<std::uintptr_t>::max();
        const std::array<Range,2> source{{{10,50},{70,100}}};
        const std::array<Range,4> subtract{{{0,12},{20,30},{40,80},{90,top}}};
        const std::vector<Range> expected{{12,20},{30,40},{80,90}};
        std::vector<Range> combined;combined.reserve(source.size()+subtract.size());
        const auto* union_storage=combined.data();const auto union_capacity=combined.capacity();
        ninfer::exl3::merge_sorted_resident_ranges(source,subtract,combined);
        require(combined==std::vector<Range>{{0,top}} && combined.data()==union_storage &&
            combined.capacity()==union_capacity,"linear resident union changed coverage or allocation");
        bool occupied_refused=false;
        try{ninfer::exl3::merge_sorted_resident_ranges({},source,combined);}
        catch(const std::length_error&){occupied_refused=true;}
        require(occupied_refused && combined==std::vector<Range>{{0,top}},
            "linear resident union overwrote retained output");
        combined.clear();
        ninfer::exl3::merge_sorted_resident_ranges({},source,combined);
        require(combined==std::vector<Range>(source.begin(),source.end()),"linear resident empty union changed coverage");
        std::vector<Range> short_output;short_output.reserve(1);
        bool short_refused=false;
        try{ninfer::exl3::merge_sorted_resident_ranges(source,subtract,short_output);}
        catch(const std::length_error&){short_refused=true;}
        require(short_refused && short_output.empty(),"linear resident union exceeded admitted output");
        const std::array<Range,2> separated{{{0,4},{110,120}}};
        const std::array<Range,2> seams{{{0,10},{50,70}}};
        for(bool reverse:{false,true}) {
            combined.clear();
            ninfer::exl3::merge_sorted_resident_ranges(reverse?std::span<const Range>(source):std::span<const Range>(separated),
                reverse?std::span<const Range>(separated):std::span<const Range>(source),combined);
            require(combined==std::vector<Range>{{0,4},{10,50},{70,100},{110,120}},
                "linear resident union filled a disjoint gap or depended on input order");
            combined.clear();
            ninfer::exl3::merge_sorted_resident_ranges(reverse?std::span<const Range>(source):std::span<const Range>(seams),
                reverse?std::span<const Range>(seams):std::span<const Range>(source),combined);
            require(combined==std::vector<Range>{{0,100}},"linear resident union lost touching boundary");
        }
        combined.clear();ninfer::exl3::merge_sorted_resident_ranges(source,source,combined);
        require(combined==std::vector<Range>(source.begin(),source.end()),"linear resident union duplicated identical input");
        combined.clear();ninfer::exl3::merge_sorted_resident_ranges({}, {},combined);
        require(combined.empty() && combined.data()==union_storage && combined.capacity()==union_capacity,
            "empty linear resident union changed reusable allocation");
        require(ninfer::exl3::resident_difference_precounted(source,subtract,3)==expected &&
            ninfer::exl3::resident_difference_precounted(source,source,0).empty(),
            "admitted difference count changed exact coverage");
        for(std::size_t count:{0U,2U,4U}) {
            bool refused=false;
            try{(void)ninfer::exl3::resident_difference_precounted(source,subtract,count);}
            catch(const std::length_error&){refused=true;}
            require(refused,"difference materialization accepted changed admitted count");
        }
        require(ninfer::exl3::resident_difference_count(source,subtract)==expected.size() &&
            ninfer::exl3::resident_difference(source,subtract)==expected,
            "resident difference lost split fragments or miscounted descriptor demand");
        require(ninfer::exl3::resident_difference(subtract,source)==
            std::vector<Range>{{0,10},{50,70},{100,top}},
            "resident difference reversed coverage or overflowed maximum endpoint");
        require(ninfer::exl3::resident_difference_count(source,source)==0 &&
            ninfer::exl3::resident_difference(source,source).empty(),
            "equal resident unions retained difference descriptors");
        require(ninfer::exl3::resident_difference({},source).empty() &&
            ninfer::exl3::resident_difference(source,{})==std::vector<Range>(source.begin(),source.end()),
            "empty resident union subtraction changed source coverage");
        const std::array<Range,2> touching{{{0,10},{100,top}}};
        require(ninfer::exl3::resident_difference(source,touching)==
            std::vector<Range>(source.begin(),source.end()),
            "half-open touching resident ranges removed live bytes");
    }
    {
        using Range=std::pair<std::uintptr_t,std::uintptr_t>;
        const auto top=std::numeric_limits<std::uintptr_t>::max();
        std::vector<Range> ranges{{90,100},{12,18},{10,20},{20,30},{10,20},{top-4,top},{80,95}};
        ranges.reserve(19);
        const auto* backing=ranges.data();const auto capacity=ranges.capacity();
        ninfer::exl3::merge_resident_ranges_in_place(ranges);
        require(ranges==std::vector<Range>{{10,30},{80,100},{top-4,top}} &&
            ranges.data()==backing && ranges.capacity()==capacity,
            "resident interval union changed coverage or allocated replacement backing");
        ninfer::exl3::merge_resident_ranges_in_place(ranges);
        require(ranges.size()==3 && ranges.data()==backing && ranges.capacity()==capacity,
            "resident union was not idempotent with stable backing");
        ranges.clear();ninfer::exl3::merge_resident_ranges_in_place(ranges);
        require(ranges.empty() && ranges.data()==backing && ranges.capacity()==capacity,
            "empty resident union lost retained storage");
        ranges.emplace_back(1,top);
        ninfer::exl3::merge_resident_ranges_in_place(ranges);
        require(ranges==std::vector<Range>{{1,top}} && ranges.data()==backing,
            "resident union reuse overflowed or replaced backing");
    }
    {
        using Queue=ninfer::exl3::ReservedRequestQueue<int>;
        Queue queue;require(queue.empty(),"unbound startup queue is not empty");
        auto backing=ninfer::exl3::make_bounded_shared<Queue::Storage>();backing->reserve(2);
        queue.bind(backing);const auto capacity=queue.capacity();
        require(capacity==2,"reserved queue fixture requires exact startup capacity");
        auto first=std::make_shared<int>(1),second=std::make_shared<int>(2);
        std::weak_ptr<int> retired=first;
        queue.push_back(first);queue.push_back(second);first.reset();
        bool refused=false;try{queue.push_back(second);}catch(const std::length_error&){refused=true;}
        require(refused && queue.size()==2 && *queue.front()==1,"full reserved queue changed FIFO or grew");
        queue.erase(queue.begin());
        require(retired.expired() && *queue.front()==2 && queue.capacity()==capacity,
            "queue cancellation retained removed owner or changed capacity/order");
        queue.pop_front();queue.push_back(second);
        require(queue.size()==1 && queue.capacity()==capacity,"reserved queue failed slot reuse");
        queue.pop_front();
        bool late_reserve=false;
        try{backing->reserve(capacity+1);}catch(const std::logic_error&){late_reserve=true;}
        require(late_reserve && backing->capacity()==capacity && queue.empty(),
            "bound queue backing allowed unaccounted growth");
        Queue alias;
        bool alias_refused=false;
        try{alias.bind(backing);}catch(const std::logic_error&){alias_refused=true;}
        require(alias_refused && alias.empty() && alias.capacity()==0,
            "inventory alias acquired a second mutable queue");
        auto replacement=ninfer::exl3::make_bounded_shared<Queue::Storage>();replacement->reserve(1);
        bool rebind_refused=false;
        try{queue.bind(replacement);}catch(const std::logic_error&){rebind_refused=true;}
        require(rebind_refused && queue.capacity()==capacity,
            "failed rebind replaced reserved queue storage");
        alias.bind(replacement);alias.push_back(second);
        queue.push_back(second);
        require(alias.size()==1 && queue.size()==1,
            "failed binding consumed replacement or damaged original queue");
        auto retained=ninfer::exl3::make_bounded_shared<Queue::Storage>();retained->reserve(1);
        std::weak_ptr<int> inventory_retained;
        {
            Queue temporary;temporary.bind(retained);
            auto value=std::make_shared<int>(3);inventory_retained=value;
            temporary.push_back(value);
        }
        require(!inventory_retained.expired(),"inventory lost queued owner with wrapper destruction");
        Queue late_alias;bool retired_binding_refused=false;
        try{late_alias.bind(retained);}catch(const std::logic_error&){retired_binding_refused=true;}
        require(retired_binding_refused && late_alias.empty(),
            "retained inventory backing was rebound after wrapper destruction");
        retained.reset();
        require(inventory_retained.expired(),"retired inventory backing retained queued request");
    }
    {
        ninfer::exl3::EngineRoundTokenStorage round;
        {
            alignas(4096) std::array<std::byte,8192> bytes{};
            auto owner=std::make_shared<int>(1);
            ninfer::exl3::ReservedHostPayloadUnion<2> pages;
            pages.add(owner,0,bytes.data(),16);pages.add(owner,0,bytes.data(),16);
            pages.add(owner,1,bytes.data()+32,16);
            require(pages.allocations()==2 && pages.page_bytes(4096)==4096,
                "credited shared payload duplicated allocation or page charge");
            bool changed=false;try{pages.add(owner,0,bytes.data(),32);}catch(const std::invalid_argument&){changed=true;}
            require(changed && pages.allocations()==2,"changed credited payload replaced admitted extent");
            bool full=false;try{pages.add(owner,2,bytes.data()+4096,16);}catch(const std::length_error&){full=true;}
            require(full && pages.page_bytes(4096)==4096,"full reconciliation changed page union");
            const auto begin=reinterpret_cast<std::uintptr_t>(bytes.data());
            const std::array<std::pair<std::uintptr_t,std::uintptr_t>,1> resident{{{begin,begin+4096}}};
            require(pages.bytes_outside({})==32 && pages.bytes_outside(resident)==0,
                "credited payload/resident overlap double counted raw allocation");
            const std::array<std::pair<std::uintptr_t,std::uintptr_t>,1> partial{{{begin,begin+8}}};
            require(pages.bytes_outside(partial)==24,"partial credited allocation coverage lost bytes");
            const std::array<std::pair<std::uintptr_t,std::uintptr_t>,1> incoming{{{begin+4,begin+48}}};
            require(pages.bytes_outside(partial,incoming)==0 && pages.bytes_outside(incoming,partial)==0,
                "old/new resident overlap discounted twice or depended on input order");
            require(pages.bytes_outside(resident,resident)==0,
                "identical old/new resident union underflowed uncovered payload");
            const std::array<std::pair<std::uintptr_t,std::uintptr_t>,2> overlap{{{begin,begin+16},{begin+8,begin+48}}};
            bool invalid=false;try{(void)pages.bytes_outside(overlap);}catch(const std::invalid_argument&){invalid=true;}
            require(invalid,"overlapping resident ranges discounted credit twice");
            ninfer::exl3::ReservedHostPayloadTracker<1> tracker;
            auto transient=std::make_shared<int>(2);
            tracker.track(transient,0,bytes.data()+4096,16);
            tracker.track(transient,0,bytes.data()+4096,16);
            std::weak_ptr<int> witness=transient;
            {
                const auto snapshot=tracker.snapshot();transient.reset();
                require(!witness.expired() && snapshot.allocations()==1,
                    "payload tracker snapshot failed to retain live allocation");
                bool refused=false;
                try{tracker.track(owner,1,bytes.data(),16);}catch(const std::length_error&){refused=true;}
                require(refused,"payload tracker evicted a snapshot-held owner");
            }
            require(witness.expired() && tracker.snapshot().allocations()==0,
                "weak payload tracker kept retired allocation alive");
            tracker.track(owner,1,bytes.data(),16);
            require(tracker.snapshot().allocations()==1,"payload tracker could not reuse expired slot");
            {
                using Ledger=ninfer::exl3::RetainedHostAllocationLedger;
                struct CreditedOwner {
                    Ledger::Ticket credit;
                    explicit CreditedOwner(Ledger::Ticket value):credit(std::move(value)) {}
                };
                Ledger ledger;
                ninfer::exl3::ReservedHostPayloadTracker<2> credited;
                auto first=std::make_shared<CreditedOwner>(ledger.acquire(16));
                auto survivor=std::make_shared<CreditedOwner>(ledger.acquire(16));
                credited.track(first,0,bytes.data(),16);
                credited.track(survivor,0,bytes.data()+4096,16);
                {
                    auto unbound=ledger.acquire(7);
                    require(credited.additional_bytes(ledger,{})==39,
                        "host admission omitted unbound or nonresident payload credit");
                    require(credited.additional_bytes(ledger,resident)==23,
                        "host admission discounted a survivor outside resident roots");
                    require(credited.additional_bytes(ledger,partial)==31,
                        "host admission discounted page padding instead of raw overlap");
                    const std::array<std::pair<std::uintptr_t,std::uintptr_t>,1> both{{{begin,begin+8192}}};
                    require(credited.additional_bytes(ledger,both)==7,
                        "old/new resident union lost unbound payload promise");
                    Ledger foreign;
                    bool refused=false;
                    try{(void)credited.additional_bytes(foreign,{});}catch(const std::logic_error&){refused=true;}
                    require(refused,"underfunded host ledger accepted tracked payloads");
                    bool malformed=false;
                    try{(void)credited.additional_bytes(ledger,overlap);}catch(const std::invalid_argument&){malformed=true;}
                    require(malformed && ledger.bytes()==39,
                        "malformed resident union consumed payload reservations");
                }
                require(credited.additional_bytes(ledger,resident)==16,
                    "released unbound promise remained charged");
                {
                    const auto held=credited.snapshot();survivor.reset();
                    require(credited.additional_bytes(ledger,resident)==16,
                        "snapshot-held survivor retired its credit prematurely");
                }
                require(credited.additional_bytes(ledger,resident)==0 && ledger.bytes()==16,
                    "retired survivor retained payload charge");
                first.reset();
                require(credited.additional_bytes(ledger,{})==0 && ledger.bytes()==0,
                    "expired tracker records retained payload charge");
            }
        }
        const std::array<std::size_t,5> capacities{1,2,3,4,5};
        const auto half=std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t);
        require(!ninfer::exl3::pending_payload_extent(0,{half+1,0,0,0,0}),
            "tap-only payload accepted overflowing plane capacity");
        require(!ninfer::exl3::pending_payload_extent(0,{half,1,0,0,0}),
            "tap-only payload accepted overflowing combined capacity");
        require(ninfer::exl3::pending_payload_extent(0,capacities)==30,
            "tap-only payload included token or wrapper charge");
        require(ninfer::exl3::pending_payload_extent(8,capacities,11)==105,
            "Pending capacity extent omitted wrapper or tap capacity");
        require(!ninfer::exl3::pending_payload_extent(std::numeric_limits<std::size_t>::max(),capacities),
            "Pending token capacity overflow accepted");
        require(!ninfer::exl3::pending_payload_extent(0,capacities,std::numeric_limits<std::size_t>::max()-1),
            "Pending cumulative tap extent overflow accepted");
        const std::array<std::int64_t,8> full{0,1,2,3,4,5,6,std::numeric_limits<ninfer::TokenId>::max()};
        auto tokens=round.assign(full);
        require(tokens.size()==8 && tokens.back()==full.back(),"fixed round lost complete token range");
        const auto* storage=tokens.data();
        const auto stale_full=tokens;
        const auto stale_repair=round.repair();
        tokens=round.truncate(3);
        require(tokens.data()==storage && round.repair().size()==3 && round.repair().back()==2,
            "round truncation lost stable storage or repair prefix");
        bool stale_output_refused=false,stale_repair_refused=false;
        try{(void)stale_full.size();}catch(const std::logic_error& error){
            stale_output_refused=std::string_view(error.what())=="stale Engine round token view";
        }
        try{(void)stale_repair.data();}catch(const std::logic_error& error){
            stale_repair_refused=std::string_view(error.what())=="stale Engine round token view";
        }
        require(stale_output_refused && stale_repair_refused,
            "round overwrite left an earlier borrowed host view readable");
        bool refused=false;try{round.truncate(4);}catch(const std::invalid_argument&){refused=true;}
        require(refused && round.output().size()==3,"expanded decoder prefix mutated fixed round");
        for(const auto bad:{std::int64_t{-1},std::int64_t{std::numeric_limits<ninfer::TokenId>::max()}+1}) {
            const std::array<std::int64_t,2> invalid{7,bad};refused=false;
            try{round.assign(invalid);}catch(const std::invalid_argument&){refused=true;}
            require(refused && round.output().size()==3 && round.output().front()==0,
                "invalid round token partially overwrote retained prefix");
        }
        const std::array<std::int64_t,17> excess{};refused=false;
        try{round.assign(excess);}catch(const std::invalid_argument&){refused=true;}
        require(refused && round.output().size()==3,"oversized round changed retained extent");
        require(round.truncate(0).empty() && round.repair().empty(),"zero accepted prefix retained repair tokens");
        require(round.assign(full).data()==storage,"next round replaced fixed conversion storage");
        const auto retained_round=round.output();
        const std::array<std::int64_t,2> replacement{17,18};
        require(round.assign(replacement).data()==storage && round.output().front()==17,
            "next round failed to reuse the bounded host arena");
        bool next_round_refused=false;
        try{(void)retained_round.front();}catch(const std::logic_error& error){
            next_round_refused=std::string_view(error.what())=="stale Engine round token view";
        }
        require(next_round_refused,"retained round view observed next-round overwrite");
        const std::array<std::int64_t,16> two_blocks{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        for(const std::size_t accepted:{0u,1u,7u,8u,9u,15u,16u}) {
            require(round.assign(two_blocks).data()==storage,"two-block round changed fixed storage");
            const auto prefix=round.truncate(accepted);
            require(prefix.size()==accepted && round.repair().size()==accepted &&
                std::equal(round.repair().begin(),round.repair().end(),two_blocks.begin()) &&
                std::equal(prefix.begin(),prefix.end(),two_blocks.begin()),
                "two-block stop boundary changed repair or publication prefix");
        }
        auto invalid_second=two_blocks;invalid_second[15]=-1;refused=false;
        try{round.assign(invalid_second);}catch(const std::invalid_argument&){refused=true;}
        require(refused && round.output().size()==16 && round.output().back()==15 &&
            round.repair().back()==15,"invalid second block partially replaced retained round");
    }
    {
        ninfer::StopPolicy stop;
        require(ninfer::exl3::request_stop_storage_bytes(stop)==0,"empty stop policy charged dynamic storage");
        stop.token_ids.reserve(17);stop.token_ids.push_back(1);
        stop.strings.reserve(5);stop.strings.push_back({"x",ninfer::OutputChannel::Content,false});
        stop.strings.front().text.reserve(257);
        const auto expected=stop.token_ids.capacity()*sizeof(ninfer::TokenId)+
            stop.strings.capacity()*sizeof(ninfer::StopString)+stop.strings.front().text.capacity()+1;
        require(ninfer::exl3::request_stop_storage_bytes(stop)==expected,
            "stop policy charge omitted unused vector or text capacity");
        auto moved=std::move(stop);
        require(ninfer::exl3::request_stop_storage_bytes(moved)==expected,
            "stop policy ownership transfer changed retained charge");
        moved.strings.clear();moved.token_ids.clear();
        require(ninfer::exl3::request_stop_storage_bytes(moved)==
            moved.token_ids.capacity()*sizeof(ninfer::TokenId)+moved.strings.capacity()*sizeof(ninfer::StopString),
            "cleared stop policy lost retained vector charge or retained destroyed text charge");
    }
    {
        unsigned destroyed=0;
        struct Value {unsigned* count;~Value(){++*count;}};
        // This fixture uses an ordinary separately allocated Value; Engine uses
        // aligned reserved backing. The same allocator exercises shared_ptr's
        // failure/deleter contract without depending on a simulated throw site.
        bool refused=false;
        try {
            auto owner=std::shared_ptr<Value>(new Value{&destroyed},
                [](Value* value) noexcept {delete value;},ninfer::exl3::ResultControlAllocator<Value>(true));
        } catch(const std::bad_alloc&){refused=true;}
        require(refused && destroyed==1,"control-block failure did not destroy constructed value exactly once");
        {
            auto retry=std::shared_ptr<Value>(new Value{&destroyed},
                [](Value* value) noexcept {delete value;},ninfer::exl3::ResultControlAllocator<Value>(false));
            require(destroyed==1,"control-block retry destroyed live value");
        }
        require(destroyed==2,"control-block retry failed final value destruction");
    }
    require(output_result_slots_available(0,8,8,8) && output_result_slots_available(8,8,8,0),
        "exact reserved output extent refused");
    require(!output_result_slots_available(0,8,7,8) && !output_result_slots_available(7,8,8,2) &&
        !output_result_slots_available(9,8,16,0) && !output_result_slots_available(9,16,8,0),
        "output reservation allowed insufficient capacity or invalid used extent");
    const auto largest_size=std::numeric_limits<std::size_t>::max();
    using ninfer::exl3::output_text_capacity_ceiling;
    require(output_text_capacity_ceiling(0)==32 && output_text_capacity_ceiling(64)==160,
        "text admission capacity policy changed");
    const auto largest_text=(largest_size-32)/2;
    require(output_text_capacity_ceiling(largest_text)==largest_text*2+32,
        "largest representable text ceiling refused");
    bool text_overflow=false;
    try{(void)output_text_capacity_ceiling(largest_text+1);}catch(const std::overflow_error&){text_overflow=true;}
    require(text_overflow,"text capacity ceiling overflow accepted");
    using ninfer::exl3::output_result_capacity_accepted;
    require(output_result_capacity_accepted(8,8,64,160,64,160) &&
        output_result_capacity_accepted(8,8,64,160,160,64) &&
        output_result_capacity_accepted(0,0,3,38,15,15),
        "result capacity policy refused exact, rounded, or zero-token storage");
    require(!output_result_capacity_accepted(8,7,64,160,64,64) &&
        !output_result_capacity_accepted(8,9,64,160,64,64) &&
        !output_result_capacity_accepted(8,8,64,160,63,64) &&
        !output_result_capacity_accepted(8,8,64,160,64,63) &&
        !output_result_capacity_accepted(8,8,64,160,161,64) &&
        !output_result_capacity_accepted(8,8,64,160,64,161) &&
        !output_result_capacity_accepted(8,8,64,63,64,64),
        "result capacity policy accepted underallocation, excess, or invalid bound");
    require(output_result_slots_available(largest_size-1,largest_size,largest_size,1) &&
        !output_result_slots_available(largest_size-1,largest_size,largest_size,2),
        "output reservation overflowed additive extent");
    const std::array<std::size_t,9> expected_prefix{0,1,1,3,3,3,3,7,8};
    for(std::size_t credit=0;credit<expected_prefix.size();++credit)
        require(ninfer::exl3::output_delivery_prefix(multibyte,credit)==expected_prefix[credit],
            "delivery credit split UTF-8 code point");
    const std::string large(ninfer::exl3::output_delivery_byte_limit+17,'x');
    require(ninfer::exl3::output_delivery_prefix(large,ninfer::exl3::output_delivery_byte_limit)==
        ninfer::exl3::output_delivery_byte_limit,"delivery materialization exceeded byte cap");
    const auto offered=std::chrono::steady_clock::time_point{}+std::chrono::seconds(1);
    const auto claim_budget=std::chrono::microseconds(1);
    require(Cost::claim_before_deadline(offered,offered,claim_budget),"fresh bounded offer refused");
    require(Cost::claim_before_deadline(offered,offered+std::chrono::nanoseconds(999),claim_budget),
        "peer before deadline refused");
    require(!Cost::claim_before_deadline(offered,offered+claim_budget,claim_budget),
        "peer at timeout claimed expired offer");
    require(!Cost::claim_before_deadline(offered,offered+std::chrono::nanoseconds(1001),claim_budget),
        "sub-microsecond late peer survived rounded cost age");
    require(!Cost::claim_before_deadline(offered,offered-std::chrono::nanoseconds(1),claim_budget) &&
        !Cost::claim_before_deadline(offered,offered,std::chrono::microseconds(0)),
        "invalid age or zero wait budget admitted peer");
    auto narrow_savings=entries;narrow_savings[0].independent_us=45;
    require(Cost(narrow_savings).wait_budget_us(7,17408,5120,6,3)==4,
        "wait budget included break-even age");
    narrow_savings[0].independent_us=41;
    require(Cost(narrow_savings).wait_budget_us(7,17408,5120,6,3)==0,
        "zero-age profitable pair authorized positive wait");
    const std::array<Cost::Entry,2> peer_choices{{entries[0],{7,17408,5120,6,3,4,100,40,45}}};
    require(Cost(peer_choices).wait_budget_us(7,17408,5120,6,3)==45,
        "wait budget discarded longer-lived eligible peer");
    require(Cost(peer_choices).wait_budget_us(7,17408,5120,7,3)==0,
        "wait budget crossed arithmetic identity");
    auto huge_costs=entries;
    huge_costs[0].independent_us=~std::uint64_t{0};
    huge_costs[0].shared_us=huge_costs[0].independent_us-3;
    require(Cost(huge_costs).wait_budget_us(7,17408,5120,6,3)==2,
        "large absolute costs overflowed wait savings");
    const auto bypass=Cost::unqualified_test_permit_underfilled();
    require(bypass.wait_budget_us(7,17408,5120,6,3)==5000 &&
        bypass.wait_budget_us(Cost::family_count,17408,5120,6,3)==0,
        "test wait bypass lost bounded identity admission");
    auto slower=entries;slower[0].shared_us=101;
    require(!Cost(slower).permits(7,17408,5120,6,3,5,0),"slower estimate selected packing");
    auto duplicate=std::array{entries[0],entries[0]};bool duplicate_refused=false;
    try{Cost invalid(duplicate);}catch(const std::invalid_argument&){duplicate_refused=true;}
    require(duplicate_refused,"ambiguous cost estimates accepted");
    using Page=ninfer::exl3::Exl3ExactKVPage;
    using Extent=ninfer::exl3::Exl3ExactKVExtent;
    {
        using History=ninfer::exl3::Exl3AttentionStageHistory;
        using Accounting=ninfer::exl3::Exl3SharedControlAccounting;
        const auto baseline=Accounting::live_bytes.load();
        std::weak_ptr<const History::Snapshot> weak;
        {
            History history;
            auto snapshot=history.bind({},0,0);weak=snapshot;
            require(Accounting::live_bytes.load()==baseline+512,
                "history snapshot control allocation missing from physical accounting");
            snapshot.reset();
            require(!weak.expired(),"history owner did not retain reusable snapshot");
        }
        require(weak.expired() && Accounting::live_bytes.load()==baseline+512,
            "history weak survivor lost physical control-block accounting");
        weak.reset();
        require(Accounting::live_bytes.load()==baseline,"history last weak release retained control bytes");
    }
    {
        ninfer::exl3::Exl3AttentionStageHistory history;
        auto source=std::make_shared<Page>();source->rows=1;
        source->k[0].resize(1024);source->v[0].resize(1024);
        std::weak_ptr<Page> lifetime=source;
        std::array<std::shared_ptr<const Page>,1> pages{source};
        auto snapshot=history.bind(pages,1,0);
        bool clear_refused=false,rebind_refused=false;
        try{history.clear();}catch(const std::logic_error&){clear_refused=true;}
        try{history.bind({},0,0);}catch(const std::logic_error&){rebind_refused=true;}
        require(clear_refused && rebind_refused && snapshot->size()==1 && snapshot->front()==source,
            "leased history allowed replacement or clear");
        snapshot.reset();pages[0].reset();source.reset();
        require(!lifetime.expired(),"idle reusable history lost retained source before replacement");
        bool invalid=false;
        try{history.bind({},1,0);}catch(const std::invalid_argument&){invalid=true;}
        require(invalid && !lifetime.expired(),"invalid history replacement released prior snapshot");
        auto empty=history.bind({},0,0);
        require(empty->size()==0 && lifetime.expired(),"valid empty history retained old page slots");
        empty.reset();history.clear();
    }
    {
        using Transfer=ninfer::exl3::Exl3KVTransferLease;
        auto shared_page=std::make_shared<Page>();shared_page->rows=2;
        shared_page->k[0].resize(2*1024,0x3555);
        auto registration=std::make_shared<int>(1);
        auto first_output=std::make_shared<int>(2),second_output=std::make_shared<int>(3);
        std::weak_ptr<Page> page_lifetime=shared_page;
        std::weak_ptr<int> registration_lifetime=registration,first_lifetime=first_output,second_lifetime=second_output;
        Transfer first(Extent::view(shared_page,0,Extent::Plane::key,2,0),first_output,registration);
        Transfer second(Extent::view(shared_page,0,Extent::Plane::key,2,0),second_output,registration);
        const auto first_generation=first.begin(11,21,101),second_generation=second.begin(12,22,102);
        shared_page.reset();registration.reset();first_output.reset();second_output.reset();
        require(first.finish(first_generation,0),"first concurrent extent completion");
        first.clear_completed();
        require(first_lifetime.expired() && !second_lifetime.expired() && !page_lifetime.expired() &&
            !registration_lifetime.expired() && second.uncertain(),"first reader released peer dependencies");
        bool pending_clear=false;try{second.clear_completed();}catch(const std::logic_error&){pending_clear=true;}
        require(pending_clear && !second_lifetime.expired() && !registration_lifetime.expired(),
            "pending reader clear released shared registration");
        require(!second.finish(second_generation+1,0) && second.uncertain(),"stale peer event completed transfer");
        require(second.finish(second_generation,0),"second concurrent extent completion");
        second.clear_completed();
        require(second_lifetime.expired() && page_lifetime.expired() && registration_lifetime.expired(),
            "last completed reader retained transfer dependencies");
    }
    auto page=std::make_shared<Page>();page->first=64;page->rows=16;
    page->k[3].resize(16*1024,0x3555);page->v[3].resize(16*1024,0x1555);
    const auto* original=page->k[3].data();std::weak_ptr<Page> retained_page=page;
    {
        auto key=Extent::view(page,3,Extent::Plane::key,73,68);
        auto value=Extent::view(page,3,Extent::Plane::value,73,68);
        require(key.first()==68 && key.rows()==5 && key.bytes()==5ULL*1024*2 &&
            key.data()==original+4*1024,"partial authoritative extent gathered or included unpublished tail");
        require(key.destination_offset(73)==68ULL*1024 && value.data()!=key.data(),"extent plane/destination identity");
        bool short_destination=false;try{key.destination_offset(72);}catch(const std::invalid_argument&){short_destination=true;}
        require(short_destination,"extent destination overflow");
        page.reset();require(!retained_page.expired() && key.data()[0]==0x3555,"extent lost backing owner");
    }
    require(retained_page.expired(),"extent retained unused backing");
    bool ownerless=false;try{Extent::view({},0,Extent::Plane::key,1);}catch(const std::invalid_argument&){ownerless=true;}
    require(ownerless,"ownerless extent accepted");
    auto short_plane=std::make_shared<Page>();short_plane->rows=2;short_plane->k[0].resize(1024);
    auto mutable_page=std::make_shared<Page>();mutable_page->rows=1;mutable_page->k[0].resize(1024);
    auto captured_extent=Extent::view(mutable_page,0,Extent::Plane::key,1);
    const auto backing=captured_extent.backing();
    require(backing.owner==mutable_page && backing.data==mutable_page->k[0].data() && backing.bytes==2048,
        "extent backing identity does not describe actual plane storage");
    mutable_page->k[0].resize(2048);
    require(!captured_extent.backing_current(),"resized backing retained transfer eligibility");
    bool stale_extent=false;try{captured_extent.data();}catch(const std::invalid_argument&){stale_extent=true;}
    require(stale_extent,"extent exposed stale transfer pointer");
    auto captured_rows=Extent::view(mutable_page,0,Extent::Plane::key,1);
    {
        using Transfer=ninfer::exl3::Exl3KVTransferLease;
        auto destination=std::make_shared<int>(7),registration=std::make_shared<int>(8);
        std::weak_ptr<int> output_lifetime=destination,registration_lifetime=registration;
        {
            Transfer transfer(captured_rows,destination,registration);
            destination.reset();registration.reset();
            const auto generation=transfer.begin(1,3,99);
            require(!transfer.finish(generation+1,0) && !transfer.complete(),"stale event released transfer owners");
            require(!output_lifetime.expired() && !registration_lifetime.expired(),"pending transfer lost owners");
            bool pending_rebind=false;
            try{transfer.rebind(captured_rows,std::make_shared<int>(10));}
            catch(const std::logic_error&){pending_rebind=true;}
            require(pending_rebind && !output_lifetime.expired() && !registration_lifetime.expired(),
                "pending rebind replaced transfer owners");
            require(transfer.finish(generation,0) && transfer.complete(),"matching transfer completion lost");
            bool invalid_rebind=false;
            try{transfer.rebind(captured_rows,{});}
            catch(const std::invalid_argument&){invalid_rebind=true;}
            require(invalid_rebind && transfer.complete() && !output_lifetime.expired(),
                "invalid rebind changed completed transfer");
            int borrowed_output=0;
            invalid_rebind=false;
            try{transfer.rebind(captured_rows,std::shared_ptr<const void>(std::shared_ptr<const void>{},&borrowed_output));}
            catch(const std::invalid_argument&){invalid_rebind=true;}
            require(invalid_rebind && transfer.complete() && !output_lifetime.expired(),
                "non-owning rebind replaced completed transfer backing");
            for(unsigned dependency=0;dependency<2;++dependency) {
                const auto nonowning=std::shared_ptr<const void>(std::shared_ptr<const void>{},&borrowed_output);
                invalid_rebind=false;
                try{transfer.rebind(captured_rows,std::make_shared<int>(12),
                    dependency==0?nonowning:std::shared_ptr<const void>{},
                    dependency==1?nonowning:std::shared_ptr<const void>{});}
                catch(const std::invalid_argument&){invalid_rebind=true;}
                require(invalid_rebind && transfer.complete() && !output_lifetime.expired() && !registration_lifetime.expired(),
                    "invalid optional dependency replaced completed transfer owners");
            }
            transfer.rebind(captured_rows,std::make_shared<int>(11));
            require(output_lifetime.expired() && registration_lifetime.expired() && !transfer.complete(),
                "completed rebind retained old owners or completion");
            require(!transfer.finish(generation,0),"old completion accepted between transfer bindings");
            const auto next_generation=transfer.begin(2,4,99);
            require(next_generation>generation && !transfer.finish(generation,0) && !transfer.complete(),
                "recycled event address accepted previous transfer generation");
            require(transfer.finish(next_generation,0) && transfer.complete(),"rebound transfer completion lost");
        }
        require(output_lifetime.expired() && registration_lifetime.expired(),"completed transfer retained owners");
        const auto before=Transfer::quarantined_records();
        {
            auto output=std::make_shared<int>(9);
            Transfer failed(captured_rows,output);
            const auto generation=failed.begin(2,4,100);
            require(failed.finish(generation,7) && !failed.complete(),"failed event certified completion");
            bool failed_rebind=false;
            try{failed.rebind(captured_rows,std::make_shared<int>(12));}
            catch(const std::logic_error&){failed_rebind=true;}
            require(failed_rebind,"failed transfer was recycled");
        }
        require(Transfer::quarantined_records()==before+1,"failed transfer record not retained");
    }
    phase="attention/page/graph contracts";
    {
        using Key=ninfer::exl3::Exl3DevicePageKey;
        auto model=std::make_shared<int>(1);
        auto published=std::make_shared<ninfer::exl3::Exl3ExactKVPage>();published->rows=64;
        for(int bank=0;bank<16;++bank){published->k[bank].resize(64*1024);published->v[bank].resize(64*1024);}
        Key original(model,published,{0,false},64),alias(model,published,{0,false},128);
        for(bool missing_model:{false,true}) {
            bool rejected=false;
            try{Key invalid(missing_model?std::shared_ptr<const void>(std::shared_ptr<const void>{},model.get()):model,
                missing_model?published:std::shared_ptr<const ninfer::exl3::Exl3ExactKVPage>(
                    std::shared_ptr<const ninfer::exl3::Exl3ExactKVPage>{},published.get()),{0,false},64);}
            catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"device page key accepted nonowning model or source alias");
        }
        require(original.same(alias),"same published device page failed identity match");
        Key shifted(model,published,{1,false},64);
        auto other_model=std::make_shared<int>(1);
        Key foreign(other_model,published,{0,false},64);
        auto copied_page=std::make_shared<ninfer::exl3::Exl3ExactKVPage>(*published);
        Key collision(model,copied_page,{0,false},64);
        require(original.shortlist()==collision.shortlist() && !original.same(collision) &&
            !original.same(shifted) && !original.same(foreign),
            "device page shortlist bypassed source/model/position identity");
        std::shared_ptr<const void> false_owner(model.get(),[](const void*){});
        Key same_address(false_owner,published,{0,false},64);
        require(!original.same(same_address),"device page key trusted address without ownership");
        bool media=false;try{Key invalid(model,published,{0,true},64);}
        catch(const std::invalid_argument&){media=true;}
        bool unpublished=false;try{Key invalid(model,published,{0,false},63);}
        catch(const std::invalid_argument&){unpublished=true;}
        require(media && unpublished,"device page accepted unsupported media or unpublished tail");
        using Fill=ninfer::exl3::Exl3DevicePageFill;
        phase="device page bounded metadata credit";
        {
            using namespace ninfer::exl3;
            auto storage=std::make_shared<std::vector<std::uint16_t>>(Fill::elements);
            auto owner=make_bounded_shared<Fill>(original,storage,*storage);
            RetainedDescriptorLedger metadata;
            require(!Fill::attach_own_metadata_credit(owner,metadata.acquire(Fill::metadata_bytes()-1)) && !metadata.bytes(),
                "page fill accepted short component metadata credit");
            require(Fill::attach_own_metadata_credit(owner,metadata.acquire(Fill::metadata_bytes())) &&
                !Fill::attach_own_metadata_credit(owner,metadata.acquire(Fill::metadata_bytes())) &&
                metadata.bytes()==Fill::metadata_bytes(),"page fill exact or duplicate metadata credit mismatch");
            const auto generation=owner->begin(81,82,601,model);
            for(int bank=0;bank<16;++bank)for(bool key_plane:{false,true})owner->plane_submitted(generation,bank,key_plane);
            require(owner->finish(generation,0),"page metadata fixture publication");
            auto shared=owner->ready_handle();std::weak_ptr<const Fill::View> weak_view=shared;
            std::weak_ptr<Fill> weak_fill=owner;std::optional<Fill::View> copy=owner->view();
            const auto fill_bytes=bounded_shared_allocation_bytes<Fill>();
            const auto view_bytes=bounded_shared_allocation_bytes<Fill::View>();
            const auto reader_bytes=bounded_shared_allocation_bytes<std::atomic<std::uint64_t>>();
            owner.reset();
            require(weak_fill.expired() && metadata.bytes()==fill_bytes+view_bytes+reader_bytes,
                "page state destruction released a surviving bounded control charge");
            shared.reset();
            require(weak_view.expired() && metadata.bytes()==fill_bytes+view_bytes+reader_bytes,
                "page weak view or copied counter released metadata early");
            weak_view.reset();require(metadata.bytes()==fill_bytes+reader_bytes,"page final weak view retained its block charge");
            weak_fill.reset();require(metadata.bytes()==reader_bytes,"page final weak fill released copied reader counter");
            copy.reset();require(metadata.bytes()==0,"page copied-view counter leaked final allocation credit");
        }
        std::optional<Fill::View> visible;
        std::weak_ptr<std::vector<std::uint16_t>> device_owner;
        phase="device page fill publication";
        {
            auto storage=std::make_shared<std::vector<std::uint16_t>>(Fill::elements);
            bool storage_refused=false;
            try{Fill invalid(original,std::shared_ptr<const void>(std::shared_ptr<const void>{},storage.get()),*storage);}
            catch(const std::invalid_argument&){storage_refused=true;}
            require(storage_refused,"device page fill accepted nonowning storage alias");
            device_owner=storage;
            bool event_deleter_saw_ready=false;
            Fill fill(original,storage,*storage);storage.reset();
            bool event_refused=false;
            try{fill.begin(2,3,99,std::shared_ptr<const void>(std::shared_ptr<const void>{},model.get()));}
            catch(const std::logic_error&){event_refused=true;}
            require(event_refused && !fill.failed(),"nonowning fill event changed producer state");
            require(!fill.fill_in_flight(),"unsubmitted allocation counted as in-flight fill");
            auto event_owner=std::shared_ptr<int>(new int(99),[&](int* event) noexcept {
                delete event;
                try {
                    const auto handle=fill.ready_handle();
                    event_deleter_saw_ready=handle && handle->generation()!=0;
                } catch(...) {event_deleter_saw_ready=false;}
            });
            std::weak_ptr<int> event_weak=event_owner;
            const auto generation=fill.begin(2,3,99,event_owner);
            event_owner.reset();
            require(fill.fill_in_flight(),"submitted fill omitted in-flight accounting");
            bool hidden=false;try{fill.view();}catch(const std::logic_error&){hidden=true;}
            require(hidden && !fill.ready_handle() && !fill.finish(generation,0),"partial device page became visible");
            require(!event_weak.expired(),"partial fill released its event owner");
            for(int bank=15;bank>=0;--bank) {
                fill.plane_submitted(generation,bank,false);
                fill.plane_submitted(generation,bank,true);
            }
            bool duplicate=false;try{fill.plane_submitted(generation,0,true);}
            catch(const std::logic_error&){duplicate=true;}
            require(duplicate && !fill.finish(generation+1,0) && !fill.ready(),
                "duplicate plane or stale event certified device page");
            require(fill.finish(generation,0) && fill.ready(),"all-plane page readiness lost");
            require(event_weak.expired() && event_deleter_saw_ready,
                "fill event final deleter could not inspect complete published handle");
            require(!fill.fill_in_flight(),"published fill retained in-flight accounting");
            const auto ready_handle=fill.ready_handle();
            require(ready_handle && ready_handle==fill.ready_handle() && ready_handle->generation()==generation,
                "ready handle allocated another identity or lost readiness generation");
            visible=fill.view();
            require(visible->plane(15,false).size()==Fill::plane_elements &&
                visible->key().same(original),"ready page geometry/key changed");
        }
        require(!device_owner.expired(),"immutable page handle lost storage owner");
        phase="device page reader identity";
        {
            using Reader=ninfer::exl3::Exl3DevicePageReader;
            auto view=std::make_shared<Fill::View>(*visible);
            auto lane=std::make_shared<int>(17);std::weak_ptr<int> lane_owner=lane;
            Reader first,second;
            first.bind(view,original,view->generation(),lane);
            second.bind(view,original,view->generation(),lane);
            const auto a=first.begin(11,12,201),b=second.begin(21,22,202);
            const auto saved_first=published->first;
            ++published->first;
            bool stale_key=false,stale_plane=false;
            try{first.input_key();}catch(const std::logic_error&){stale_key=true;}
            try{second.plane(0,true);}catch(const std::invalid_argument&){stale_plane=true;}
            require(stale_key && stale_plane && first.uncertain() && second.uncertain(),
                "active page reader exposed stale backing or released final-use ownership");
            published->first=saved_first;
            for(int bank=0;bank<16;++bank)for(bool key:{false,true}) {
                auto& plane=key?published->k[bank]:published->v[bank];
                auto replacement=plane;
                plane.swap(replacement); // preserve original allocation for exact restoration
                bool stale=false;
                try{first.input_key();}catch(const std::logic_error&){stale=true;}
                require(stale && !original.current() && first.uncertain() && second.uncertain(),
                    "reader accepted same-value replacement in one KV bank/plane");
                plane.swap(replacement);
                require(original.current() && first.input_key().same(original),
                    "restored physical page backing lost current identity");
            }
            {
                auto replacement=std::make_shared<int>(18);
                std::weak_ptr<int> replacement_owner=replacement;
                bool active_rebind=false;
                try{first.bind(view,original,view->generation(),replacement);}
                catch(const std::logic_error&){active_rebind=true;}
                replacement.reset();
                bool repeated_begin=false;
                try{first.begin(11,13,203);}catch(const std::logic_error&){repeated_begin=true;}
                bool nonowning_event=false;
                try{first.bind(view,original,view->generation(),
                    std::shared_ptr<const void>(std::shared_ptr<const void>{},lane.get()));}
                catch(const std::invalid_argument&){nonowning_event=true;}
                require(active_rebind && repeated_begin && nonowning_event && replacement_owner.expired() &&
                    first.uncertain() && first.input_key().same(original),
                    "rejected reader transition replaced active ownership or retained candidate");
            }
            visible.reset();view.reset();lane.reset();
            require(first.plane(15,false).size()==Fill::plane_elements && !device_owner.expired(),
                "active reader lost immutable device plane");
            require(!first.finish(a+1,0) && first.uncertain(),"stale reader event released owner");
            require(second.finish(b,0) && second.complete() && !device_owner.expired() && !lane_owner.expired(),
                "one reader completion released another consumer");
            require(first.finish(a,0) && first.complete() && device_owner.expired() && lane_owner.expired(),
                "last reader final use did not release storage/event owners");
            bool expired_access=false;try{first.plane(0,true);}catch(const std::logic_error&){expired_access=true;}
            require(expired_access,"completed reader exposed recycled device pointer");
        }
        visible.reset();require(device_owner.expired(),"idle immutable page handle leaked storage");
        phase="attention and projection host contracts";
        {
            auto storage=std::make_shared<std::vector<std::uint16_t>>(Fill::elements);
            Fill sealing(original,storage,*storage);storage.reset();
            {
                using Profile=ninfer::exl3::Exl3AttentionProfile;
                {
                    using Plan=ninfer::exl3::Exl3ExactDownLifetimePlan;
                    {
                        using ninfer::exl3::exl3_require_paired_transform_extents;
                        {
                            using ninfer::exl3::exl3_require_residual_norm_extents;
                            const std::array<std::uintptr_t,5> buffers{0x1000000,0x2000000,0x3000000,0x4000000,0x5000000};
                            for(int rows:{-1,0,1025,std::numeric_limits<int>::max()}) {
                                bool refused=false;
                                try{exl3_require_residual_norm_extents(rows,buffers);}
                                catch(const std::invalid_argument&){refused=true;}
                                require(refused,"residual norm admitted unsupported row extent");
                            }
                            for(int rows:{1,7,8,15,16,255,256,512,1024}) {
                                exl3_require_residual_norm_extents(rows,buffers);
                                auto shared_read=buffers;shared_read[1]=shared_read[0];
                                exl3_require_residual_norm_extents(rows,shared_read);
                                for(unsigned output:{3u,4u})for(unsigned other=0;other<output;++other) {
                                    auto overlap=buffers;overlap[output]=buffers[other]+2;
                                    bool refused=false;
                                    try{exl3_require_residual_norm_extents(rows,overlap);}
                                    catch(const std::invalid_argument&){refused=true;}
                                    require(refused,"fused residual norm admitted partial live overlap");
                                }
                            }
                        }
                        using Admission=ninfer::exl3::Exl3PairedTransformAdmission;
                        using ninfer::exl3::exl3_paired_transform_admission;
                        for(int rows:{1,8,15,16,256,1024}) {
                            require(exl3_paired_transform_admission(false,false,false,false,rows,true)==Admission::disabled,
                                "wide rows bypassed disabled paired transform");
                            require(exl3_paired_transform_admission(true,true,false,false,rows,true)==Admission::timing,
                                "paired transform bypassed per-projection timing");
                            require(exl3_paired_transform_admission(true,false,true,false,rows,true)==Admission::observer,
                                "paired transform bypassed canonical observer inputs");
                            require(exl3_paired_transform_admission(true,false,false,true,rows,true)==Admission::admitted,
                                "wide prefill failed paired admission");
                        }
                        for(int rows:{8,15,16})
                            require(exl3_paired_transform_admission(true,false,false,true,rows,false)==Admission::preserved_multirow,
                                "paired transform changed preserved multirow topology");
                        require(exl3_paired_transform_admission(true,false,false,true,1,false)==Admission::admitted,
                            "single-row canonical paired transform excluded");
                        for(int rows:{0,-1})
                            require(exl3_paired_transform_admission(true,false,false,false,rows,true)==Admission::invalid_rows,
                                "empty or negative paired work admitted");
                        using ninfer::exl3::exl3_equal_transform_reuse_enabled;
                        require(!exl3_equal_transform_reuse_enabled(nullptr) &&
                            !exl3_equal_transform_reuse_enabled("0") && exl3_equal_transform_reuse_enabled("1"),
                            "paired transform reuse option semantics");
                        for(const char* value:{"","2","true","01"," 1","1 "}) {
                            bool refused=false;
                            try{(void)exl3_equal_transform_reuse_enabled(value);}
                            catch(const std::invalid_argument&){refused=true;}
                            require(refused,"malformed paired transform option admitted");
                        }
                        const std::array<std::uintptr_t,5> base{0x10000,0x20000,0x20000,0x30000,0x40000};
                        for(unsigned slot=0;slot<base.size();++slot)for(bool wrapping:{false,true}) {
                            auto invalid=base;
                            invalid[slot]=wrapping?std::numeric_limits<std::uintptr_t>::max()-1:0;
                            bool refused=false;
                            try{exl3_require_paired_transform_extents(16,128,invalid);}
                            catch(const std::invalid_argument&){refused=true;}
                            require(refused,"paired transform null or wrapping extent admitted");
                        }
                        for(int rows:{1,8,15,16})exl3_require_paired_transform_extents(rows,128,base);
                        for(unsigned output:{3u,4u})for(unsigned other=0;other<output;++other) {
                            auto overlap=base;overlap[output]=base[other]+2;
                            bool refused=false;
                            try{exl3_require_paired_transform_extents(16,128,overlap);}
                            catch(const std::invalid_argument&){refused=true;}
                            require(refused,"paired transform partial write overlap admitted");
                        }
                        for(int rows:{0,-1,std::numeric_limits<int>::max()}) {
                            bool refused=false;
                            try{exl3_require_paired_transform_extents(rows,128,base);}
                            catch(const std::invalid_argument&){refused=true;}
                            require(refused,"paired transform invalid row indexing admitted");
                        }
                    }
                    require(!ninfer::exl3::exl3_reconstruction_budget_fallback(nullptr) &&
                        !ninfer::exl3::exl3_reconstruction_budget_fallback("0") &&
                        ninfer::exl3::exl3_reconstruction_budget_fallback("1"),"reconstruction budget fallback option");
                    for(const char* bad:{"","2","true","01"}) {
                        bool refused=false;
                        try{ninfer::exl3::exl3_reconstruction_budget_fallback(bad);}
                        catch(const std::invalid_argument&){refused=true;}
                        require(refused,"malformed reconstruction fallback admitted");
                    }
                    {
                        auto model=std::make_shared<int>(7),other=std::make_shared<int>(7);
                        std::weak_ptr<int> weak=model;
                        ninfer::exl3::Exl3ReconstructionStream binding;
                        binding.bind_model(model,4096,true);
                        binding.require_identity(model,4096,true);
                        for(int mismatch=0;mismatch<4;++mismatch) {
                            std::shared_ptr<const void> supplied=mismatch==0?other:model;
                            if(mismatch==3)supplied.reset();
                            bool refused=false;
                            try{binding.require_identity(supplied,mismatch==1?2048:4096,mismatch!=2);}
                            catch(const std::invalid_argument&){refused=true;}
                            require(refused,"reconstruction identity mismatch admitted");
                        }
                        model.reset();require(!weak.expired(),"reconstruction backing lost model owner");
                    }
                    {
                        using Alloc=ninfer::exl3::Exl3ReconstructionControlAllocator<std::byte>;
                        bool admitted=false;
                        Alloc allocator(&admitted);
                        bool oversized=false;
                        try{allocator.allocate(Alloc::capacity+1);}catch(const std::bad_alloc&){oversized=true;}
                        require(oversized && !admitted,"oversized control block consumed reservation");
                        auto* bytes=allocator.allocate(Alloc::payload_capacity);
                        bool duplicate=false;
                        try{allocator.allocate(1);}catch(const std::bad_alloc&){duplicate=true;}
                        require(duplicate && admitted,"multiple control allocations escaped reservation");
                        allocator.deallocate(bytes,Alloc::payload_capacity);
                        struct Owner {bool admitted=false;};
                        auto* raw=new Owner;
                        bool destroyed=false;
                        std::shared_ptr<Owner> owner(raw,[&](Owner* value){destroyed=true;delete value;},Alloc(&raw->admitted));
                        std::weak_ptr<Owner> weak=owner;
                        auto retained=owner;owner.reset();
                        require(!destroyed && !weak.expired(),"control allocator lost retained owner");
                        retained.reset();
                        require(destroyed && weak.expired(),"control allocator extended backing object lifetime");
                        weak.reset(); // Control deallocation must not dereference freed Owner::admitted.
                    }
                        {
                            ninfer::exl3::Exl3ProjectionGraphBinding binding;
                            using Options=ninfer::exl3::Exl3ProjectionGraphOptions;
                            const auto absent=Options::from_values({nullptr,nullptr,nullptr});
                            require(absent.matches(absent),"absent graph options mismatch");
                            for(const auto changed:{Options::from_values({"1",nullptr,nullptr}),
                                Options::from_values({nullptr,"2",nullptr}),
                                Options::from_values({nullptr,nullptr,"direct"}),
                                Options::from_values({nullptr,nullptr,nullptr,"1"}),
                                Options::from_values({"",nullptr,nullptr})})
                                require(!absent.matches(changed),"changed graph arithmetic identity accepted");
                            const std::string too_long(64,'1');
                            const auto oversized=Options::from_values({too_long.c_str(),nullptr,nullptr});
                            require(!oversized.matches(oversized),"truncated graph option identity reused");
                            auto model=std::make_shared<int>(1),scratch=std::make_shared<int>(2);
                            {
                                std::shared_ptr<const void> empty;
                                std::shared_ptr<const void> borrowed_model(empty,model.get());
                                std::shared_ptr<const void> borrowed_scratch(empty,scratch.get());
                                binding.bind(borrowed_model,scratch,absent);
                                require(!binding.matches(borrowed_model,scratch,absent),
                                    "graph accepted model pointer without retained ownership");
                                binding.bind(model,borrowed_scratch,absent);
                                require(!binding.matches(model,borrowed_scratch,absent),
                                    "graph accepted scratch pointer without retained ownership");
                                std::shared_ptr<const void> null_scratch(scratch,nullptr);
                                binding.bind(model,null_scratch,absent);
                                require(!binding.matches(model,null_scratch,absent),
                                    "graph accepted owned null scratch as an absent group");
                                std::shared_ptr<const void> model_alias(model,model.get());
                                std::shared_ptr<const void> scratch_alias(scratch,scratch.get());
                                binding.bind(model_alias,scratch_alias,absent);
                                require(binding.matches(model,scratch,absent),
                                    "graph rejected aliases retaining the same backing groups");
                            }
                            binding.bind(model,scratch,absent);
                            require(!binding.matches(model,scratch,Options::from_values({"2",nullptr,nullptr})),
                                "graph owner match ignored arithmetic change");
                            require(binding.matches(model,scratch,absent),"unchanged arithmetic graph refused");
                            binding={};
                            require(!binding.matches(model,scratch),"unbound graph reused");
                            binding.bind(model,scratch);
                            require(binding.matches(model,scratch),"stable graph binding rejected");
                            require(!binding.matches(model,std::make_shared<int>(2)),"replaced scratch graph reused");
                            std::shared_ptr<const void> foreign(scratch.get(),[](const void*){});
                            require(!binding.matches(model,foreign),"same-address foreign graph owner accepted");
                            require(!binding.matches(std::make_shared<int>(1),scratch),"replaced model graph reused");
                            std::weak_ptr<int> weak=scratch;scratch.reset();
                            binding.invalidate();
                            require(!binding.matches(model,weak.lock()),"invalidated graph binding reused");
                            require(!weak.expired(),"graph binding lost scratch owner");
                            binding.bind(model,{});
                            require(weak.expired() && binding.matches(model,{}),"direct graph binding retirement mismatch");
                        }
                    for(int columns:{256,512,1024,5120}) {
                        for(int remaining:{16,17,23,31,32}) {
                            const auto tail=ninfer::exl3::exl3_prefill_chunk_step(32,remaining,16,48,true);
                            const auto canonical=ninfer::exl3::exl3_prefill_chunk_step(32,remaining,16,48);
                            require(tail.rows==remaining && tail.next_position==16+remaining,
                                "actual tail lost represented rows");
                            require(canonical.rows==(remaining==32?32:16),"tail option changed canonical default");
                            const auto narrow=ninfer::exl3::exl3_prefill_chunk_step(16,remaining,16,48,true);
                            require(narrow.rows==16,"tail widened configured row capacity");
                        }
                        using Consumer=ninfer::exl3::Exl3HeadConsumer;
                        using Head=ninfer::exl3::Exl3HeadConsumerPlan;
                        require(Head::make(1,Consumer::verification,true).root_row==0,
                            "single-row head omission lost only row");
                        for(int bad:{0,-1}) {
                            bool refused=false;
                            try{Head::make(bad,Consumer::root_only,true);}catch(const std::invalid_argument&){refused=true;}
                            require(refused,"empty head plan admitted");
                        }
                        const auto root=Head::make(columns,Consumer::root_only,true);
                        require(root.rows==1 && root.first_row==columns-1 && root.root_row==columns-1,
                            "head omission lost final root");
                        for(auto consumer:{Consumer::verification,Consumer::sampled_rows,Consumer::diagnostic_rows}) {
                            const auto full=Head::make(columns,consumer,false);
                            require(full.first_row==0 && full.rows==columns && full.root_row==columns-1,
                                "head plan omitted consumed logits");
                            bool refused=false;
                            try{Head::make(columns,consumer,true);}catch(const std::invalid_argument&){refused=true;}
                            require(refused,"unsafe head omission admitted");
                        }
                        const auto bytes=std::size_t(17408)*columns*2;
                        require(ninfer::exl3::exl3_reconstruction_reservation_fits(bytes),
                            "implemented reconstruction reservation rejected");
                        require(!ninfer::exl3::exl3_reconstruction_reservation_fits(bytes-1) &&
                            !ninfer::exl3::exl3_reconstruction_reservation_fits(bytes+1),
                            "unsupported reconstruction reservation selected");
                        require(ninfer::exl3::exl3_down_slice_columns(bytes)==columns,
                            "reconstruction bounded slice capacity mismatch");
                        const auto plan=Plan::make(256,5,6,false,true,false,columns);
                        require(plan.buffers[1].bytes==bytes && plan.buffers[2].bytes==256ull*5120*5*4,
                            "slice altered parent accumulator topology");
                        bool refused=false;
                        try{ninfer::exl3::exl3_down_slice_columns(bytes-1);}
                        catch(const std::invalid_argument&){refused=true;}
                        require(refused,"short reconstruction slice admitted");
                    }
                    for(const char* invalid:{"","0","255","2048","0512","512 "}) {
                        bool refused=false;
                        try{ninfer::exl3::exl3_reconstruction_slice_columns(invalid);}
                        catch(const std::invalid_argument&){refused=true;}
                        require(refused,"invalid reconstruction slice option admitted");
                    }
                    for(bool invalid_owner:{false,true}) {
                        int queries=0,actions=0;
                        const auto result=ninfer::exl3::exl3_retire_reconstruction_device(invalid_owner?-1:3,
                            [&](int* device) noexcept {++queries;*device=-1;return 0;},
                            [&](int) noexcept {++actions;return 0;},
                            [&]() noexcept {++actions;return 0;},[&]() noexcept {++actions;});
                        require(result.error==-1 && !result.released && actions==0 && queries==(invalid_owner?0:1),
                            "reconstruction invalid device identity admitted retirement");
                    }
                    for(int failure=0;failure<=4;++failure) {
                        int current=7,drains=0,releases=0,selections=0;
                        bool correct_release_device=false;
                        const auto result=ninfer::exl3::exl3_retire_reconstruction_device(3,
                            [&](int* device) noexcept {*device=current;return failure==1?11:0;},
                            [&](int device) noexcept {
                                ++selections;
                                if((failure==2 && device==3) || (failure==4 && device==7))return 12;
                                current=device;return 0;
                            },
                            [&]() noexcept {++drains;return failure==3?13:0;},
                            [&]() noexcept {++releases;correct_release_device=current==3;});
                        const bool released=failure==0 || failure==4;
                        require(result.released==released && releases==(released?1:0),
                            "reconstruction device retirement released uncertain owner");
                        require(!released || correct_release_device,"reconstruction release on wrong device");
                        require(drains==((failure==1 || failure==2)?0:1),
                            "reconstruction drain after device admission failure");
                        require(failure==4 ? result.restore_error==12 : current==7,
                            "reconstruction prior device restoration contract");
                        require(selections==(failure==1?0:2),"reconstruction device selection sequence");
                    }
                    {
                        int releases=0,selections=0;
                        const auto result=ninfer::exl3::exl3_retire_reconstruction_device(3,
                            [](int* device) noexcept {*device=3;return 0;},
                            [&](int) noexcept {++selections;return 0;},
                            []() noexcept {return 0;},[&]() noexcept {++releases;},true);
                        require(!result.released && result.error==-2 && !releases && !selections,
                            "reconstruction injected retirement bypassed retention");
                    }
                    for(std::uintptr_t first:{std::uintptr_t(0),std::uintptr_t(17)}) {
                        ninfer::exl3::Exl3ReconstructionStream slab;
                        slab.require_ordered(first);slab.require_ordered(first);
                        bool refused=false;
                        try{slab.require_ordered(first==0?17:0);}
                        catch(const std::invalid_argument&){refused=true;}
                        require(refused,"reconstruction slab admitted unordered consumer stream");
                        slab.require_ordered(first);
                        {
                            auto submission=slab.acquire(first);
                            require(submission.owns_lock(),"reconstruction submission not serialized");
                            slab.fail();
                        }
                        require(slab.failed(),"reconstruction failure latch missing");
                        for(auto retry:{first,first==0?std::uintptr_t(17):std::uintptr_t(0)}) {
                            bool failed=false;
                            try{slab.require_ordered(retry);}
                            catch(const std::runtime_error&){failed=true;}
                            require(failed,"reconstruction failed slab admitted reuse");
                        }
                    }
                    using ninfer::exl3::exl3_reconstruction_k6_down_option;
                    for(bool slab:{false,true}) {
                        require(!exl3_reconstruction_k6_down_option(nullptr,slab) &&
                            !exl3_reconstruction_k6_down_option("0",slab),"K6 reconstruction default isolation");
                        for(const char* bad:{"","2","-1","01","true"," 1","1 "}) {
                            bool refused=false;
                            try{exl3_reconstruction_k6_down_option(bad,slab);}
                            catch(const std::invalid_argument&){refused=true;}
                            require(refused,"K6 reconstruction malformed option admitted");
                        }
                    }
                    bool missing_slab=false;
                    try{exl3_reconstruction_k6_down_option("1",false);}
                    catch(const std::invalid_argument&){missing_slab=true;}
                    require(missing_slab && exl3_reconstruction_k6_down_option("1",true),
                        "K6 reconstruction slab prerequisite");
                    for(int k:{6,7}) for(int rows:{256,512,1024}) for(int splits:{1,5}) {
                        const auto plan=Plan::make(rows,splits,k,false,true,false);
                        std::array<std::uintptr_t,4> addresses{0x1000,0x10000000,0x20000000,0x30000000};
                        std::array<std::size_t,4> capacities{};
                        for(std::size_t i=0;i<4;++i)capacities[i]=plan.buffers[i].bytes;
                        plan.require_binding(addresses,capacities);
                        // Output begins after input/decode consumers finish. This
                        // checks the interval contract, not permission to recycle
                        // a slab whose external asynchronous owners are still live.
                        for(int retired:{0,1}) {
                            auto reused=addresses;reused[3]=reused[retired];
                            plan.require_binding(reused,capacities);
                        }
                        for(std::size_t i=0;i<4;++i) {
                            for(bool overflow:{false,true}) {
                                auto invalid=addresses;
                                invalid[i]=overflow?std::numeric_limits<std::uintptr_t>::max()-plan.buffers[i].bytes+1:0;
                                bool refused=false;
                                try{plan.require_binding(invalid,capacities);}
                                catch(const std::invalid_argument&){refused=true;}
                                require(refused,"exact Down invalid address extent admitted");
                            }
                        }
                        for(std::size_t i=0;i<4;++i) {
                            auto short_capacity=capacities;--short_capacity[i];
                            bool refused=false;
                            try{plan.require_binding(addresses,short_capacity);}
                            catch(const std::invalid_argument&){refused=true;}
                            require(refused,"exact Down undersized lifetime buffer admitted");
                        }
                        for(auto pair:{std::pair{0,1},std::pair{0,2},std::pair{1,2},std::pair{2,3}}) {
                            auto overlap=addresses;overlap[pair.second]=overlap[pair.first];
                            bool refused=false;
                            try{plan.require_binding(overlap,capacities);}
                            catch(const std::invalid_argument&){refused=true;}
                            require(refused,"exact Down overlapping live consumers admitted");
                        }
                    }
                    for(int unsupported=0;unsupported<4;++unsupported) {
                        bool refused=false;
                        try{Plan::make(256,1,unsupported==0?5:7,unsupported==1,
                                       unsupported!=2,unsupported==3);}
                        catch(const std::invalid_argument&){refused=true;}
                        require(refused,"exact Down unsupported transform admitted");
                    }
                }
                {
                    using Step=ninfer::exl3::Exl3AttentionScratchStep;
                    using Life=ninfer::exl3::Exl3AttentionScratchLifetime;
                    const Life input{Step::input_norm,Step::qkv_projection,8192};
                    const Life mlp{Step::residual_norm,Step::mlp_projection,8192};
                    require(ninfer::exl3::exl3_attention_scratch_can_alias(input,mlp,true,false,false),
                        "disjoint input/MLP lifetimes refused");
                    require(!ninfer::exl3::exl3_attention_scratch_can_alias(input,mlp,true,true,false) &&
                        !ninfer::exl3::exl3_attention_scratch_can_alias(input,mlp,false,false,false) &&
                        !ninfer::exl3::exl3_attention_scratch_can_alias(input,mlp,true,false,true),
                        "diagnostic/cross-stream/failed-consumer lifetime admitted alias");
                    const Life overlapping{Step::qkv_projection,Step::attention,8192};
                    require(!ninfer::exl3::exl3_attention_scratch_can_alias(input,overlapping,true,false,false),
                        "shared producer-consumer endpoint admitted alias");
                }
                {
                    using namespace attention_reference;
                    Storage local(16,99);
                    auto a=std::make_shared<const Storage>(Storage{7,7,20,21,30,31,7,7});
                    auto b=std::make_shared<const Storage>(Storage{60,61});
                    std::array<Owner,2> registry{a,b};
                    std::array<attention_reference::Segment,2> segments{{{6,1,0,1,b},{2,2,1,0,a}}};
                    const auto expanded=expand(local,2,8,segments,registry);
                    require(expanded==Storage({99,99,99,99,20,21,30,31,99,99,99,99,60,61,99,99}),
                        "independent reference seam/private gap mapping");
                    for(int query=0;query<3;++query)for(int key=0;key<8;++key)
                        require(attention_reference::visible(3,3,query,key)==(key<4+query),"independent cross-seam causal mask");
                    auto wrong=segments;wrong[0].owner=std::make_shared<const Storage>(*b);
                    bool refused=false;
                    try{expand(local,2,8,wrong,registry);}catch(const std::invalid_argument&){refused=true;}
                    require(refused,"reference admitted equal bytes from wrong physical owner");
                    require(!attention_reference::visible(3,3,3,0) &&
                        !attention_reference::visible(3,3,-1,0),"reference admitted padded query");
                }
                {
                    using Route=ninfer::exl3::Exl3SegmentedAttentionRoute;
                    using ninfer::exl3::exl3_segmented_attention_route;
                    using ninfer::exl3::exl3_direct_staged_attention_supported;
                    for(int rows:{1,8,16})for(int score_rows:{1,8,16})
                        require(exl3_direct_staged_attention_supported(true,rows,64,64+rows,score_rows),
                            "direct staged native row menu rejected exact-fit consumer");
                    for(int score_rows:{0,17})
                        require(!exl3_direct_staged_attention_supported(true,8,64,128,score_rows),
                            "direct staged invalid score capacity admitted");
                    require(!exl3_direct_staged_attention_supported(false,8,64,128,8) &&
                        !exl3_direct_staged_attention_supported(true,0,64,128,8) &&
                        !exl3_direct_staged_attention_supported(true,17,64,128,8) &&
                        !exl3_direct_staged_attention_supported(true,8,0,128,8) &&
                        !exl3_direct_staged_attention_supported(true,8,64,71,8),
                        "direct staged unsupported profile or causal extent admitted");
                    for(int rows:{1,3,8,15,16}) {
                        require(exl3_segmented_attention_route(true,true,rows,64,0,80,24,4,256,true,true,true)==Route::query_pair,
                            "query-pair menu priority/native rows");
                        require(exl3_segmented_attention_route(true,true,rows,64,0,80,24,4,256,false,true,true)==Route::gqa_pair,
                            "GQA pair menu priority");
                        require(exl3_segmented_attention_route(true,true,rows,64,0,80,24,4,256,false,false,true)==Route::q_shared,
                            "Q-shared menu priority");
                        require(exl3_segmented_attention_route(true,true,rows,64,0,80,24,4,256,false,false,false)==Route::scalar,
                            "scalar segmented menu");
                    }
                    for(int rows:{0,17})require(exl3_segmented_attention_route(true,true,rows,64,0,128,24,4,256,true,false,false)==Route::inherited,
                        "unknown row menu did not identify inherited fallback");
                    require(exl3_segmented_attention_route(true,false,8,64,0,128,24,4,256,true,false,false)==Route::inherited &&
                        exl3_segmented_attention_route(false,true,8,64,0,128,24,4,256,true,false,false)==Route::inherited &&
                        exl3_segmented_attention_route(true,true,8,64,0,128,32,4,256,true,false,false)==Route::inherited &&
                        exl3_segmented_attention_route(true,true,8,64,57,128,24,4,256,true,false,false)==Route::inherited,
                        "precision/profile/head/offset exclusion lost");
                }
                for(int rows:{1,8,16})
                    require(ninfer::exl3::exl3_exact_attention_score_bytes(rows,65536)==
                        static_cast<std::size_t>(rows)*100*65536*4,"exact attention scratch extent");
                for(int rows:{-1,0,17}) {
                    bool refused=false;
                    try{ninfer::exl3::exl3_exact_attention_score_bytes(rows,64);}catch(const std::invalid_argument&){refused=true;}
                    require(refused,"unsupported score row menu admitted");
                }
                {
                    using Route=ninfer::exl3::Exl3AttentionStageRoute;
                    Route route;route.enabled=route.owners=route.host_kv=route.copy_stream=true;
                    route.position=64;route.rows=16;route.context_capacity=80;route.pages=1;
                    require(route.eligible(),"exact-fit staging route refused");
                    for(auto member:{&Route::private_prefix,&Route::shared_attention,&Route::shared_copy,
                        &Route::media,&Route::wide,&Route::profiling,&Route::observer,&Route::capture,&Route::graph,&Route::oscar}) {
                        auto excluded=route;excluded.*member=true;
                        require(!excluded.eligible(),"unsupported staging route admitted");
                    }
                    for(auto member:{&Route::enabled,&Route::owners,&Route::host_kv,&Route::copy_stream}) {
                        auto excluded=route;excluded.*member=false;
                        require(!excluded.eligible(),"staging missing prerequisite admitted");
                    }
                    for(int rows:{-1,0,17}) {auto excluded=route;excluded.rows=rows;require(!excluded.eligible(),"staging row menu exceeded");}
                    auto excluded=route;excluded.context_capacity=79;
                    require(!excluded.eligible(),"staging context tail overflow");
                    excluded=route;excluded.pages=1025;
                    require(!excluded.eligible(),"staging history pool overflow");
                    excluded.pages=0;require(!excluded.eligible(),"empty staging history admitted");
                }
                {
                    using Extent=ninfer::exl3::Exl3ExactKVExtent;
                    auto page=std::make_shared<ninfer::exl3::Exl3ExactKVPage>();
                    page->rows=1;page->k[0].resize(1024);page->v[0].resize(1024);
                    auto destination=std::make_shared<int>(0);
                    ninfer::exl3::Exl3AttentionStage stages;
                    const auto extent=Extent::view(page,0,Extent::Plane::key,1);
                    for(bool missing_acquisition:{false,true}) {
                        auto candidate=std::make_shared<int>(9);
                        std::weak_ptr<int> lifetime=candidate;
                        bool rejected=false;
                        try{stages.begin(0,extent,candidate,missing_acquisition?0:1,
                            missing_acquisition?1:0,101,102);}
                        catch(const std::invalid_argument&){rejected=true;}
                        candidate.reset();
                        require(rejected && lifetime.expired() && !stages.uncertain(),
                            "invalid attention stage scope retained candidate or began transfer");
                    }
                    const auto first=stages.begin(0,extent,destination,1,1,101,102);
                    {
                        ninfer::exl3::Exl3AttentionStage other;
                        const auto peer=other.begin(0,extent,destination,1,1,101,102);
                        require(peer.slot==first.slot && peer.generation==first.generation &&
                            peer.owner_identity!=first.owner_identity,
                            "independent stage ticket fixture did not isolate owner identity");
                        require(!other.producer_pending(first,101) && !other.producer_finished(first,101,0) &&
                            !other.consumer_pending(first,102) && !other.consumer_finished(first,102,0) &&
                            !other.cancel(first) && other.producer_pending(peer,101) && stages.producer_pending(first,101),
                            "foreign stage ticket changed producer/consumer ownership");
                        bool foreign=false;
                        try{other.consume(first,0,Extent::Plane::key);}catch(const std::invalid_argument&){foreign=true;}
                        require(foreign && other.producer_finished(peer,101,0),
                            "foreign stage consume bypassed owner identity or damaged valid ticket");
                        other.consume(peer,0,Extent::Plane::key);
                        require(other.consumer_finished(peer,102,0),"valid peer stage could not complete after foreign refusal");
                    }
                    bool early=false;
                    try{stages.consume(first,0,Extent::Plane::key);}catch(const std::logic_error&){early=true;}
                    require(early,"attention stage exposed unfinished producer");
                    require(!stages.producer_finished(first,102,0),"consumer event accepted as producer proof");
                    require(stages.producer_finished(first,101,0),"stage producer completion refused");
                    stages.consume(first,0,Extent::Plane::key);
                    bool busy=false;
                    try{stages.begin(0,extent,destination,1,2,101,102);}catch(const std::logic_error&){busy=true;}
                    require(busy,"attention stage recycled before final consumer");
                    require(stages.consumer_finished(first,102,0),"stage final consumer refused");
                    const auto second=stages.begin(0,extent,destination,1,2,101,102);
                    require(second.generation!=first.generation && !stages.producer_finished(first,101,0),
                        "stale attention stage generation admitted");
                    require(stages.producer_finished(second,101,0),"reused stage producer refused");
                    stages.consume(second,0,Extent::Plane::key);
                    require(stages.consumer_finished(second,102,0),"reused stage final use refused");
                    const auto cancelled=stages.begin(0,extent,destination,1,3,101,102);
                    require(stages.cancel(cancelled),"pending stage cancellation refused");
                    bool alias=false;
                    try{stages.begin(1,extent,destination,1,3,102,103);}catch(const std::invalid_argument&){alias=true;}
                    require(alias,"pending peer stage shared a live completion event");
                    for(const auto events:{std::array<std::uintptr_t,2>{101,103},
                        std::array<std::uintptr_t,2>{102,103},std::array<std::uintptr_t,2>{103,101},
                        std::array<std::uintptr_t,2>{103,102}}) {
                        auto candidate=std::make_shared<int>(4);
                        std::weak_ptr<int> candidate_lifetime=candidate;
                        bool rejected=false;
                        try{stages.begin(1,extent,candidate,2,3,events[0],events[1]);}
                        catch(const std::invalid_argument&){rejected=true;}
                        candidate.reset();
                        require(rejected && candidate_lifetime.expired() && stages.producer_pending(cancelled,101),
                            "cross-slot event alias retained candidate or altered pending producer");
                    }
                    const auto independent=stages.begin(1,extent,destination,2,3,103,104);
                    require(stages.producer_finished(independent,103,0),"distinct peer events lost admission after alias refusal");
                    stages.consume(independent,0,Extent::Plane::key);
                    require(stages.consumer_finished(independent,104,0) && stages.producer_pending(cancelled,101),
                        "healthy peer stage completion consumed cancelled producer ownership");
                    require(stages.producer_finished(cancelled,101,0),"cancelled producer failed to retire after completion");
                    const auto cancelled_use=stages.final_use_for_test(cancelled);
                    require(cancelled_use && cancelled_use->ready && cancelled_use->event==101 &&
                        cancelled_use->generation==cancelled.generation,
                        "cancelled producer certified the unused consumer event");
                    const auto after_cancel=stages.begin(0,extent,destination,1,4,101,102);
                    require(stages.producer_finished(after_cancel,101,0),"stage could not be reused after cancelled producer retired");
                    stages.consume(after_cancel,0,Extent::Plane::key);
                    require(stages.cancel(after_cancel),"consumer cancellation refused");
                    busy=false;
                    try{stages.begin(0,extent,destination,1,5,101,102);}catch(const std::logic_error&){busy=true;}
                    require(busy,"cancelled consumer released storage before its event");
                    require(stages.consumer_finished(after_cancel,102,0),"cancelled consumer final event refused");
                    auto stale_destination=std::make_shared<int>(5);
                    std::weak_ptr<int> stale_owner=stale_destination;
                    const auto stale_source=stages.begin(0,extent,stale_destination,1,6,101,102);
                    stale_destination.reset();
                    require(stages.producer_finished(stale_source,101,0),"stale-source fixture producer readiness");
                    auto original_plane=page->k[0];page->k[0].swap(original_plane);
                    bool stale_refused=false;
                    try{stages.consume(stale_source,0,Extent::Plane::key);}catch(const std::logic_error&){stale_refused=true;}
                    require(stale_refused && !stale_owner.expired() && stages.uncertain() &&
                        !stages.consumer_pending(stale_source,102),"stale source consumed or prematurely retired stage");
                    require(stages.cancel(stale_source) && stale_owner.expired() && !stages.uncertain(),
                        "ready stale-source stage could not retire without a consumer");
                    const auto stale_use=stages.final_use_for_test(stale_source);
                    require(stale_use && stale_use->ready && stale_use->event==101,
                        "unconsumed stale source retained the planned consumer event identity");
                    page->k[0].swap(original_plane);
                }
                for(bool scores:{false,true}) {
                    require(!ninfer::exl3::exl3_attention_query_pair_option(nullptr,scores) &&
                        !ninfer::exl3::exl3_attention_query_pair_option("0",scores),
                        "query-pair default or explicit opt-out changed");
                    for(const char* value:{"","2","-1","true","01","1 "," 1"}) {
                        bool refused=false;
                        try{ninfer::exl3::exl3_attention_query_pair_option(value,scores);}
                        catch(const std::invalid_argument&){refused=true;}
                        require(refused,"query-pair accepted malformed option");
                    }
                }
                require(ninfer::exl3::exl3_attention_query_pair_option("1",true),"query-pair opt-in lost");
                bool missing_scores=false;
                try{ninfer::exl3::exl3_attention_query_pair_option("1",false);}
                catch(const std::invalid_argument&){missing_scores=true;}
                require(missing_scores,"query-pair configured without parallel score storage");
                Profile scalar;scalar.scores=true;
                require(scalar.segmented(),"ordinary scalar segmented profile refused");
                auto paired=scalar;paired.k_half2=paired.v_half2=paired.gqa_pair=true;
                require(paired.segmented(),"represented GQA pair segmented profile refused");
                // Each incompatible family must exclude both otherwise eligible
                // profiles; a paired-load flag cannot override a numeric lane.
                for(auto field:{&Profile::oscar,&Profile::capture,&Profile::numeric_splitk,
                    &Profile::gqa_triple,&Profile::gqa_six,&Profile::six_scores,
                    &Profile::six_sharded,&Profile::six_packed,&Profile::six_softmax}) {
                    for(auto profile:{scalar,paired}) {
                        profile.*field=true;
                        require(!profile.segmented(),"unsupported attention profile admitted shared history");
                    }
                }
                for(auto field:{&Profile::k_half2,&Profile::v_half2}) {
                    auto profile=scalar;profile.*field=true;
                    require(!profile.segmented(),"unimplemented half2 segmented profile admitted");
                }
                paired.scores=false;
                require(!paired.segmented(),"segmented profile admitted without score scratch");
            }
            for(int rows:{1,8,16}) {
                ninfer::exl3::Exl3AttentionCausalRows mask{127,rows,127+rows};
                for(int query=0;query<rows;++query) {
                    require(mask.count(query)==128+query,"native verifier row causal endpoint");
                    for(int key=0;key<127+rows;++key)
                        require(mask.attends(query,key)==(key<=127+query),"future verifier row leaked into causal history");
                }
                require(mask.count(rows)==0 && !mask.attends(-1,0),"padded query admitted causal history");
                mask.capacity=126+rows;
                require(mask.count(0)==0,"causal view admitted insufficient capacity");
            }
            {
                std::uint16_t local=0,shared=0;
                ninfer::exl3::Exl3AttentionInputView input{&local,&local,256};
                input.shared.append(&shared,&shared,64,64,192);
                input.require_geometry(192,8);
                input.stride=512;bool stride=false;
                try{input.require_geometry(192,8);}catch(const std::invalid_argument&){stride=true;}
                input.stride=1024;input.local_v=nullptr;bool plane=false;
                try{input.require_geometry(192,8);}catch(const std::invalid_argument&){plane=true;}
                require(stride && plane,"attention input admitted unsupported stride or missing local plane");
                auto local_owner=std::make_shared<int>(1),segment_owner=std::make_shared<int>(2);
                std::weak_ptr<int> retained=segment_owner;
                {
                    ninfer::exl3::Exl3OwnedAttentionInputView owned(&local,&local,256,local_owner,local_owner);
                    bool missing=false;
                    try{owned.append({&shared,&shared,64,64},{},192);}catch(const std::invalid_argument&){missing=true;}
                    require(missing,"owned attention admitted missing segment owner");
                    const std::shared_ptr<const void> borrowed(std::shared_ptr<const void>{},&shared);
                    bool nonowning=false;
                    try{owned.append({&shared,&shared,64,64},borrowed,192);}
                    catch(const std::invalid_argument&){nonowning=true;}
                    require(nonowning,"nonowning segment alias changed attention binding");
                    owned.append({&shared,&shared,64,64},segment_owner,192);segment_owner.reset();
                    require(!retained.expired(),"owned attention lost segment lifetime");
                    for(int fault=0;fault<4;++fault) {
                        auto rejected_owner=std::make_shared<int>(3);
                        std::weak_ptr<int> rejected_lifetime=rejected_owner;
                        ninfer::exl3::Exl3AttentionPageRanges::Range range{&shared,&shared,128,64};
                        if(fault==0)range.first=127;
                        if(fault==1)range.rows=65;
                        if(fault==2)range.k=nullptr;
                        if(fault==3)range.v=nullptr;
                        bool rejected=false;
                        try{owned.append(range,rejected_owner,192);}catch(const std::invalid_argument&){rejected=true;}
                        rejected_owner.reset();
                        require(rejected && rejected_lifetime.expired() && !retained.expired(),
                            "failed attention append retained candidate or changed existing segment");
                    }
                    owned.append_local_complement(192);
                    const auto& preserved=owned.require_geometry(192,8);
                    require(preserved.shared.count==1 && preserved.shared.ranges[0].first==64 &&
                        preserved.shared.ranges[0].rows==64,
                        "owned attention lost segment geometry after coverage proof");
                }
                require(retained.expired(),"owned attention retained released input view");
                bool missing_local=false;
                try{ninfer::exl3::Exl3OwnedAttentionInputView invalid(&local,&local,256,local_owner,{});}
                catch(const std::invalid_argument&){missing_local=true;}
                require(missing_local,"owned attention accepted missing private V owner");
                const std::shared_ptr<const void> borrowed_local(std::shared_ptr<const void>{},&local);
                for(bool key:{false,true}) {
                    bool rejected=false;
                    try{ninfer::exl3::Exl3OwnedAttentionInputView invalid(&local,&local,256,
                        key?borrowed_local:local_owner,key?local_owner:borrowed_local);}
                    catch(const std::invalid_argument&){rejected=true;}
                    require(rejected,"owned attention accepted ownerless local alias");
                }
                ninfer::exl3::Exl3OwnedAttentionInputView local_only(&local,&local,256,local_owner,local_owner);
                local_only.append_local_complement(192);
                require(local_only.require_geometry(192,8).shared.count==0,"local-only view invented shared segments");
                for(int fault=0;fault<3;++fault) {
                    ninfer::exl3::Exl3OwnedAttentionInputView invalid(&local,&local,256,local_owner,local_owner);
                    invalid.append({&shared,&shared,64,64},std::make_shared<int>(4),192);
                    if(fault==0)invalid.append_local(0,63,192); // gap before shared segment
                    if(fault==1) {
                        bool overlap=false;
                        try{invalid.append_local(0,65,192);}catch(const std::invalid_argument&){overlap=true;}
                        require(overlap,"owned attention admitted local/shared overlap");
                        continue;
                    }
                    if(fault==2)invalid.append_local(0,64,192); // missing local tail
                    bool incomplete=false;
                    try{invalid.require_geometry(192,8);}catch(const std::invalid_argument&){incomplete=true;}
                    require(incomplete,"owned attention admitted incomplete chronological coverage");
                }
                ninfer::exl3::Exl3OwnedAttentionInputView reordered(&local,&local,256,local_owner,local_owner);
                reordered.append({&shared,&shared,128,64},std::make_shared<int>(5),192);
                reordered.append({&shared,&shared,0,64},std::make_shared<int>(6),192);
                reordered.append_local_complement(192);
                require(reordered.require_geometry(192,8).shared.count==2,
                    "owned attention rejected complete out-of-order segment publication");
            }
            {
                int xyz[3]{},other[3]{};
                ninfer::exl3::Exl3AttentionPositionContract position{xyz,7};
                position.require_current(xyz,7);
                bool offset=false,identity=false;
                try{position.require_current(xyz,8);}catch(const std::invalid_argument&){offset=true;}
                try{position.require_current(other,7);}catch(const std::invalid_argument&){identity=true;}
                require(offset && identity,"segmented rotary binding accepted changed offset/source");
            }
            {
                auto a=std::make_shared<ninfer::exl3::Exl3ExactKVPage>();a->rows=64;
                auto b=std::make_shared<ninfer::exl3::Exl3ExactKVPage>();b->first=64;b->rows=16;
                a->k[0].resize(64*1024);a->v[0].resize(64*1024);
                b->k[0].resize(16*1024);b->v[0].resize(16*1024);
                std::array<std::shared_ptr<const ninfer::exl3::Exl3ExactKVPage>,2> pages{a,b};
                ninfer::exl3::exl3_require_attention_history(pages,72,0);
                for(int end:{65,71,79,80})
                    ninfer::exl3::exl3_require_attention_history(pages,end,0);
                for(int end:{0,1,63,64,81}) {
                    bool refused=false;
                    try{ninfer::exl3::exl3_require_attention_history(pages,end,0);}
                    catch(const std::invalid_argument&){refused=true;}
                    require(refused,"attention history accepted extra page or absent final row");
                }
                for(int end:{1,63,64})
                    ninfer::exl3::exl3_require_attention_history(std::span(pages).first(1),end,0);
                b->first=63;
                bool overlap=false;
                try{ninfer::exl3::exl3_require_attention_history(pages,72,0);}
                catch(const std::invalid_argument&){overlap=true;}
                require(overlap,"attention history accepted overlapping page boundary");
                b->first=64;
                bool missing=false,gap=false,reordered=false;
                try{ninfer::exl3::exl3_require_attention_history(std::span(pages.data(),1),72,0);}
                catch(const std::invalid_argument&){missing=true;}
                b->first=65;
                try{ninfer::exl3::exl3_require_attention_history(pages,72,0);}catch(const std::invalid_argument&){gap=true;}
                b->first=64;std::swap(pages[0],pages[1]);
                try{ninfer::exl3::exl3_require_attention_history(pages,72,0);}catch(const std::invalid_argument&){reordered=true;}
                require(missing && gap && reordered,"attention history admitted missing/gapped/reordered prefix");
                std::swap(pages[0],pages[1]);
                b->v[0].resize(7*1024);
                bool short_value=false;
                try{ninfer::exl3::exl3_require_attention_history(pages,72,0);}catch(const std::invalid_argument&){short_value=true;}
                b->v[0].resize(16*1024);pages[1].reset();
                bool null_page=false;
                try{ninfer::exl3::exl3_require_attention_history(pages,72,0);}catch(const std::invalid_argument&){null_page=true;}
                require(short_value && null_page,"attention history admitted short V plane or null segment");
                ninfer::exl3::exl3_require_attention_history({},0,0);
                bool absent=false;
                try{ninfer::exl3::exl3_require_attention_history({},1,0);}catch(const std::invalid_argument&){absent=true;}
                require(absent,"nonempty attention accepted empty history");
            }
            {
                ninfer::exl3::Exl3AttentionPageRanges ranges;
                std::uint16_t token=0;
                ranges.append(&token,&token,128,64,256);
                ranges.append(&token,&token,0,64,256);
                require(ranges.contains_page(128,64) && ranges.contains_page(0,64) &&
                    !ranges.contains_page(64,64),"shared range gaps lost private page");
                bool overlap=false,unpublished=false;
                try{ranges.append(&token,&token,160,64,256);}catch(const std::invalid_argument&){overlap=true;}
                try{ranges.append(&token,&token,256,64,256);}catch(const std::invalid_argument&){unpublished=true;}
                require(overlap && unpublished && ranges.count==2,"range refusal mutated accepted page set");
                ranges={};
                for(int i=0;i<ranges.capacity;++i)ranges.append(&token,&token,i*64,64,4096);
                bool full=false;try{ranges.append(&token,&token,4096,64,4160);}catch(const std::invalid_argument&){full=true;}
                require(full && ranges.count==64,"fixed page range capacity not preserved");
            }
            const auto generation=sealing.begin(31,32,401,model);
            require(!sealing.seal_for_retirement(),"in-flight fill allowed storage sealing");
            for(int bank=0;bank<16;++bank)for(bool key_plane:{false,true})
                sealing.plane_submitted(generation,bank,key_plane);
            require(sealing.finish(generation,0),"sealing fixture readiness");
            auto shared=sealing.ready_handle();
            using Reader=ninfer::exl3::Exl3DevicePageReader;
            Reader first,second;
            first.bind(shared,original,shared->generation(),model);
            second.bind(shared,original,shared->generation(),model);
            require(sealing.retained_readers()==0,"unsubmitted bindings counted as active readers");
            for(const auto& invalid:std::array<std::array<std::uint64_t,3>,3>{{
                    {0,42,501},{41,0,501},{41,42,0}}}) {
                bool refused=false;
                try{first.begin(invalid[0],invalid[1],static_cast<std::uintptr_t>(invalid[2]));}
                catch(const std::logic_error&){refused=true;}
                require(refused && sealing.retained_readers()==0 && !first.uncertain() && !first.complete(),
                    "refused reader scope leaked a reservation or fabricated completion");
            }
            const auto first_use=first.begin(41,42,501),second_use=second.begin(43,44,502);
            require(sealing.retained_readers()==2,"independent reader count did not conserve aliases");
            bool repeat_begin=false;
            try{first.begin(45,46,503);}catch(const std::logic_error&){repeat_begin=true;}
            require(repeat_begin && sealing.retained_readers()==2,"repeated begin duplicated reader accounting");
            require(!first.finish(first_use+1,0) && sealing.retained_readers()==2,
                "stale event decremented reader accounting");
            require(second.finish(second_use,0) && sealing.retained_readers()==1,
                "out-of-order completion decremented other reader");
            require(!second.finish(second_use,0) && sealing.retained_readers()==1,
                "completion replay decremented reader accounting");
            require(first.finish(first_use,0) && sealing.retained_readers()==0,
                "last completion did not balance reader accounting");
            first.bind(shared,original,shared->generation(),model);
            const auto reused=first.begin(47,48,501);
            require(reused!=first_use && !first.finish(first_use,0) && sealing.retained_readers()==1,
                "reused event address accepted previous reader generation");
            require(first.finish(reused,0) && sealing.retained_readers()==0,
                "reused reader completion did not balance accounting");
            require(!sealing.seal_for_retirement(),"shared reader did not block sealing");
            shared.reset();
            {
                auto copied=sealing.view();
                require(!sealing.seal_for_retirement() && copied.plane(0,true).size()==Fill::plane_elements,
                    "value reader did not block sealing");
                auto copied_handle=std::make_shared<const Fill::View>(copied);
                Reader copied_reader;
                copied_reader.bind(copied_handle,original,copied_handle->generation(),model);
                const auto copied_use=copied_reader.begin(49,50,504);
                require(sealing.retained_readers()==1 && !sealing.seal_for_retirement(),
                    "reader through value-view copy escaped shared accounting");
                require(copied_reader.finish(copied_use,0) && sealing.retained_readers()==0,
                    "value-view reader completion did not release shared counter");
            }
            require(sealing.seal_for_retirement() && !sealing.ready_handle() && !sealing.ready() &&
                !sealing.uncertain_after_join(),"sealed completed fill exposed late handle or lost completion");
            require(!sealing.seal_for_retirement(),"seal transaction replayed");
        }
        phase="device page reader quarantine";
        const auto retained=Fill::quarantined_records();
        {
            using Reader=ninfer::exl3::Exl3DevicePageReader;
            using Transfer=ninfer::exl3::Exl3KVTransferLease;
            const auto quarantined=Transfer::quarantined_records();
            std::weak_ptr<const void> failed_storage,failed_event,successful_event;
            {
                auto storage=std::make_shared<std::vector<std::uint16_t>>(Fill::elements);
                failed_storage=storage;
                Fill owner(original,storage,*storage);storage.reset();
                const auto fill_generation=owner.begin(71,72,701,model);
                for(int bank=0;bank<16;++bank)for(bool key_plane:{false,true})
                    owner.plane_submitted(fill_generation,bank,key_plane);
                require(owner.finish(fill_generation,0),"failed-reader fixture fill incomplete");
                {
                    auto handle=owner.ready_handle();
                    auto failure_lane=std::make_shared<int>(73),success_lane=std::make_shared<int>(74);
                    failed_event=failure_lane;successful_event=success_lane;
                    Reader failed,successful;
                    failed.bind(handle,original,handle->generation(),failure_lane);
                    successful.bind(handle,original,handle->generation(),success_lane);
                    const auto failure_use=failed.begin(75,76,702),success_use=successful.begin(77,78,703);
                    handle.reset();failure_lane.reset();success_lane.reset();
                    require(failed.finish(failure_use,7) && failed.uncertain() && !failed.complete() &&
                        owner.retained_readers()==2,"failed reader prematurely balanced final-use count");
                    bool hidden=false;
                    try{failed.plane(0,true);}catch(const std::logic_error&){hidden=true;}
                    require(hidden && !failed.finish(failure_use,0),"failed reader recovered or exposed device data");
                    require(successful.finish(success_use,0) && successful_event.expired() &&
                        !failed_event.expired() && owner.retained_readers()==1 && !owner.seal_for_retirement(),
                        "successful peer released failed reader ownership");
                }
                require(Transfer::quarantined_records()==quarantined+1 &&
                    owner.retained_readers()==1 && !owner.seal_for_retirement(),
                    "failed reader destruction lost quarantined view/count");
            }
            require(!failed_storage.expired() && !failed_event.expired() && successful_event.expired(),
                "fill destruction released failed reader's quarantined dependencies");
        }
        phase="device page storage retirement";
        {
            Fill* retiring_fill=nullptr;
            bool reenter=false,observed_empty=false;
            struct RetiringStorage {
                std::vector<std::uint16_t> data=std::vector<std::uint16_t>(Fill::elements);
                mutable unsigned attempts=0;
            };
            auto storage=std::shared_ptr<RetiringStorage>(
                new RetiringStorage,[&](auto* data) noexcept {
                    delete data;
                    if(reenter)try {
                        observed_empty=!retiring_fill->ready_handle() &&
                            !retiring_fill->storage_owner_for_retirement() && retiring_fill->retained_readers()==0;
                    } catch(...) {observed_empty=false;}
                });
            std::weak_ptr<const void> storage_weak=storage;
            Fill retiring(original,storage,storage->data,[](const std::shared_ptr<const void>& owner) noexcept {
                return ++static_cast<const RetiringStorage*>(owner.get())->attempts==2;
            });
            retiring_fill=&retiring;storage.reset();
            const auto generation=retiring.begin(61,62,603,model);
            for(int bank=0;bank<16;++bank)for(bool key_plane:{false,true})
                retiring.plane_submitted(generation,bank,key_plane);
            require(retiring.finish(generation,0),"retirement retry fixture did not finish all planes");
            auto active_handle=retiring.ready_handle();
            require(active_handle && !retiring.seal_for_retirement() && !retiring.retire_sealed_storage(),
                "live page handle allowed physical retirement");
            require(static_cast<const RetiringStorage*>(storage_weak.lock().get())->attempts==0,
                "reader-blocked retirement invoked physical callback");
            active_handle.reset();
            require(retiring.seal_for_retirement(),
                "storage final-deleter fixture did not reach sealed readiness");
            require(!retiring.retire_sealed_storage() && !storage_weak.expired() &&
                retiring.sealed_for_retirement() && !retiring.ready_handle() && !observed_empty,
                "refused physical retirement released storage or reopened page visibility");
            require(static_cast<const RetiringStorage*>(storage_weak.lock().get())->attempts==1,
                "refused physical retirement did not preserve retry state");
            reenter=true;
            const bool released=retiring.retire_sealed_storage();
            reenter=false;
            require(released && storage_weak.expired() && observed_empty,
                "sealed storage final deleter could not observe empty fill ownership");
            require(!retiring.retire_sealed_storage(),"sealed storage retirement replayed");
        }
        phase="device page fill quarantine";
        {
            auto storage=std::make_shared<std::vector<std::uint16_t>>(Fill::elements);
            Fill failed(original,storage,*storage);const auto generation=failed.begin(2,4,100,model);
            failed.plane_submitted(generation,15,false);
            require(failed.finish(generation,7) && !failed.ready() && !failed.finish(generation,0),
                "failed device page fill became ready");
        }
        require(Fill::quarantined_records()==retained+1,"failed page fill lost quarantine owner");
        published->v[15].resize(64*1024+1);
        require(!original.current() && !original.same(alias),"last bank backing change retained page reuse");
    }
    mutable_page->rows=2;
    require(!captured_rows.backing_current(),"page publication changed without extent refresh");
    bool short_storage=false;try{Extent::view(short_plane,0,Extent::Plane::key,2);}catch(const std::invalid_argument&){short_storage=true;}
    require(short_storage,"extent read beyond represented storage");
    phase="borrowed linear workspace contract";
    using Linear=ninfer::exl3::Exl3LinearWorkspaceRequirements;
    {
        const auto required=Linear::derive(128,128,1,true,true);
        alignas(16) std::array<std::byte,2816> storage{};
        required.require_disjoint_borrowed_views(storage.data(),storage.data()+256);
        required.require_disjoint_borrowed_views(storage.data()+2560,storage.data());
        required.require_disjoint_borrowed_views(nullptr,nullptr);
        for(const auto offset:{2u,4u,8u,14u}) {
            bool refused=false;
            try {required.require_disjoint_borrowed_views(storage.data()+offset,nullptr);}
            catch(const std::invalid_argument&) {refused=true;}
            if(!refused)throw std::runtime_error("borrowed transform async-A misalignment admitted");
        }
        for(bool transform:{false,true}) {
            bool refused=false;
            try {required.require_disjoint_borrowed_views(transform?storage.data()+1:nullptr,
                transform?nullptr:storage.data()+1);}
            catch(const std::invalid_argument&) {refused=true;}
            if(!refused)throw std::runtime_error("borrowed workspace element misalignment admitted");
        }
        for(const auto offset:{0u,1u,255u}) {
            bool refused=false;
            try {required.require_disjoint_borrowed_views(storage.data(),storage.data()+offset);}
            catch(const std::invalid_argument&) {refused=true;}
            if(!refused)throw std::runtime_error("borrowed workspace overlap admitted");
        }
        const auto overflow=reinterpret_cast<const void*>(std::numeric_limits<std::uintptr_t>::max()-127);
        for(bool transform:{false,true}) {
            bool refused=false;
            try {required.require_disjoint_borrowed_views(transform?overflow:nullptr,transform?nullptr:overflow);}
            catch(const std::overflow_error&) {refused=true;}
            if(!refused)throw std::runtime_error("borrowed workspace address wrap admitted");
        }
    }
    phase="fixed linear workspace owners";
    {
        struct Owner {void* ptr;std::size_t bytes;std::shared_ptr<int> lifetime;};
        std::weak_ptr<int> retained;
        {
            ninfer::exl3::Exl3FixedAllocationOwners<Owner,2> owners;
            int storage=0;void* destination=nullptr;std::size_t total=0;unsigned calls=0;
            const auto factory=[&] {
                ++calls;auto lifetime=std::make_shared<int>();retained=lifetime;
                return std::make_unique<Owner>(Owner{&storage,23,std::move(lifetime)});
            };
            owners.reserve(2);
            Linear::allocate_owned(owners,total,&destination,23,factory);
            Linear::allocate_owned(owners,total,&destination,23,factory);
            bool refused=false;
            try{Linear::allocate_owned(owners,total,&destination,23,factory);}
            catch(const std::length_error&){refused=true;}
            require(refused && calls==2 && total==46 && owners.size()==2 && !retained.expired(),
                "fixed owner saturation allocated or lost retained storage");
        }
        require(retained.expired(),"fixed owner array did not release owned allocation");
    }
    phase="fallible linear workspace owners";
    {
        struct Owner {void* ptr;std::size_t bytes;std::shared_ptr<int> lifetime;};
        struct Owners:std::vector<std::unique_ptr<Owner>> {
            bool fail_reserve=true;
            void reserve(std::size_t count){if(fail_reserve)throw std::bad_alloc{};std::vector<std::unique_ptr<Owner>>::reserve(count);}
        } owners;
        int storage=0,sentinel=0;void* destination=&sentinel;std::size_t total=17;unsigned calls=0;
        const auto factory=[&]{++calls;return std::make_unique<Owner>(Owner{&storage,23,std::make_shared<int>(1)});};
        bool metadata_failed=false;
        try{Linear::allocate_owned(owners,total,&destination,23,factory);}
        catch(const std::bad_alloc&){metadata_failed=true;}
        require(metadata_failed && calls==0 && owners.empty() && total==17 && destination==&sentinel,
            "metadata failure reached allocation or published context state");
        owners.fail_reserve=false;
        bool factory_failed=false;
        try{Linear::allocate_owned(owners,total,&destination,23,[&]()->std::unique_ptr<Owner>{++calls;throw std::bad_alloc{};});}
        catch(const std::bad_alloc&){factory_failed=true;}
        require(factory_failed && calls==1 && owners.empty() && total==17 && destination==&sentinel,
            "device factory failure published context state");
        for(std::size_t actual:{22u,24u}) {
            bool mismatch=false;
            try{Linear::allocate_owned(owners,total,&destination,23,
                [&]{return std::make_unique<Owner>(Owner{&storage,actual});});}
            catch(const std::runtime_error&){mismatch=true;}
            require(mismatch && owners.empty() && total==17 && destination==&sentinel,
                "mismatched physical extent published allocation");
        }
        Linear::allocate_owned(owners,total,&destination,23,factory);
        require(calls==2 && owners.size()==1 && owners[0]->ptr==destination && destination==&storage && total==40,
            "context allocation retry lost owner or published wrong extent");
        std::weak_ptr<int> survivor=owners[0]->lifetime;
        for(unsigned invalid=0;invalid<4;++invalid) {
            std::weak_ptr<int> candidate;
            bool rejected=false;
            try{Linear::allocate_owned(owners,total,&destination,23,[&]()->std::unique_ptr<Owner>{
                if(invalid==3)return {};
                auto life=std::make_shared<int>(2);candidate=life;
                return std::make_unique<Owner>(Owner{invalid==2?nullptr:&sentinel,
                    invalid==0?22u:invalid==1?24u:23u,std::move(life)});
            });}catch(const std::runtime_error&){rejected=true;}
            require(rejected && candidate.expired() && !survivor.expired() && owners.size()==1 &&
                destination==&storage && total==40,
                "rejected allocation retained candidate or disturbed surviving ownership");
        }
        owners.clear();require(survivor.expired(),"allocation ledger retained retired owner");
    }
    require(Linear::append_owned_bytes(17,23)==40,"context cumulative allocation extent");
    require(Linear::append_owned_bytes(std::numeric_limits<std::size_t>::max()-1,1)==
        std::numeric_limits<std::size_t>::max(),"context exact-fit allocation extent");
    bool total_overflow=false,unknown_extent=false;
    try{Linear::append_owned_bytes(std::numeric_limits<std::size_t>::max(),1);}
    catch(const std::overflow_error&){total_overflow=true;}
    try{Linear::append_owned_bytes(0,0);}catch(const std::invalid_argument&){unknown_extent=true;}
    require(total_overflow && unknown_extent,"context cumulative allocation accepted overflow or missing extent");
    const auto q=Linear::derive(5120,12288,16);
    require(q.transformed_bytes==16ULL*5120*2 && q.accumulation_bytes==16ULL*12288*5*4,
        "packed Q workspace requirement");
    require(q.owned_bytes==q.transformed_bytes+q.accumulation_bytes,"workspace requirement sum");
    for(int rows:{16,128,1024}) {
        const auto shared=Linear::derive(17408,17408,rows);
        require(shared.transformed_bytes==std::uint64_t(rows)*34816 &&
            shared.accumulation_bytes==std::uint64_t(rows)*348160,
            "target shared linear arena requirements changed physical widths");
    }
    require(Linear::derive(5120,12288,16,true,true).owned_bytes==0,"borrowed arenas charged twice");
    require(Linear::derive(5120,12288,16,true,false).owned_bytes==q.accumulation_bytes,"private accumulation omitted");
    for(auto shape:{std::array<int,3>{0,128,16},{128,127,16},{128,128,0}}) {
        bool refused=false;try{Linear::derive(shape[0],shape[1],shape[2]);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"unsupported workspace requirement accepted");
    }
    bool huge=false;try{Linear::derive(2147483520,2147483520,2147483647);}
    catch(const std::overflow_error&){huge=true;}
    require(huge,"workspace accumulation overflow accepted");
    phase="resource availability and terminal inventory";
    using Availability=ninfer::exl3::Exl3DeviceAvailability;
    {
        using ninfer::exl3::exl3_host_metadata_limit;
        require(exl3_host_metadata_limit(nullptr)==std::numeric_limits<std::uint64_t>::max() &&
            exl3_host_metadata_limit("0")==0 && exl3_host_metadata_limit("4096")==4096 &&
            exl3_host_metadata_limit("18446744073709551615")==std::numeric_limits<std::uint64_t>::max(),
            "metadata cap changed absence, zero, extent or uint64 boundary");
        for(const auto* invalid:{"","-1","+1"," 1","1 ","1KiB","18446744073709551616"}) {
            bool refused=false;try{(void)exl3_host_metadata_limit(invalid);}catch(const std::invalid_argument&){refused=true;}
            require(refused,"metadata cap accepted malformed or overflowing bytes");
        }
    }
    const auto observed=Availability::Clock::time_point{}+std::chrono::seconds(10);
    const Availability observation{100,25,observed,0};
    require(observation.limit(60,5,observed,std::chrono::seconds(1))==80,"availability exact boundary");
    const auto attribution=observation.attribution(60,observed,std::chrono::seconds(1));
    require(attribution.inventoried_bytes==60 && attribution.observed_used_bytes==75 &&
        attribution.driver_unknown_bytes==15,
        "availability attribution mislabeled or lost driver-unknown bytes");
    for(auto now:{observed-std::chrono::seconds(1),observed+std::chrono::seconds(2)}) {
        bool refused=false;try{observation.limit(60,5,now,std::chrono::seconds(1));}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"stale/future observation credited");
    }
    bool missing_provider=false;try{Availability::read({});}catch(const std::invalid_argument&){missing_provider=true;}
    require(missing_provider,"missing provider accepted");
    bool failed=false;try{Availability::read([&]{return Availability{100,100,observed,1};});}
    catch(const std::runtime_error&){failed=true;}
    require(failed,"failed provider converted into free capacity");
    using Taps=ninfer::exl3::Exl3EngineTapRequirements;
    require(Taps::for_lanes(1).units[0]==5ULL*16*5120*2,"C1 allocator requirement drift");
    require(Taps::for_lanes(2).units[0]==2*Taps::for_lanes(1).units[0],"C2 allocator requirement drift");
    const auto with_metadata=Taps::for_lanes(2,40);
    require(with_metadata.units[0]==Taps::for_lanes(2).units[0] &&
        with_metadata.units[static_cast<unsigned>(Inventory::Domain::host_metadata)]==80,
        "tap owner metadata changed device charge or omitted a lane");
    bool metadata_overflow=false;
    try{Taps::for_lanes(2,UINT64_MAX);}catch(const std::overflow_error&){metadata_overflow=true;}
    require(metadata_overflow,"tap owner metadata overflow accepted");
    for(unsigned count:{0U,3U,16U}) {
        bool refused=false;try{Taps::for_lanes(count);}catch(const std::invalid_argument&){refused=true;}
        require(refused,"unsupported tap allocation menu accepted");
    }
    Inventory::Requirement requirement;requirement.configuration=2;
    requirement.add(Inventory::Domain::device,2,5ULL*16*5120*2);
    require(requirement.units[0]==2ULL*5*16*5120*2,"C2 tap requirement");
    bool overflowed=false;
    try{requirement.add(Inventory::Domain::device,UINT64_MAX,2);}
    catch(const std::overflow_error&){overflowed=true;}
    require(overflowed && requirement.units[0]==2ULL*5*16*5120*2,"requirement overflow mutated extent");
    for(auto domain:{Inventory::Domain::device,Inventory::Domain::count}) {
        bool refused=false;try{requirement.add(domain,1,0);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"unknown allocation extent accepted");
    }
    require(Inventory::device_limit(60,100,25,5)==80,"external bytes credited");
    require(Inventory::device_limit(60,100,25,25)==60,"exact reserve fit");
    require(Inventory::device_limit(60,100,40,0)==100,"capacity bound");
    for(auto values:{std::array<std::uint64_t,4>{60,100,25,26},
                     std::array<std::uint64_t,4>{60,100,101,0},
                     std::array<std::uint64_t,4>{101,100,25,0},
                     std::array<std::uint64_t,4>{60,100,80,0}}) {
        bool refused=false;try{Inventory::device_limit(values[0],values[1],values[2],values[3]);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"contradictory availability accepted");
    }
    auto weights=std::make_shared<std::array<int,2>>();
    auto private_a=std::make_shared<int>(),private_b=std::make_shared<int>();
    std::weak_ptr<const void> survivor=weights;
    Inventory old,next;
    old.add({weights,0,Inventory::Domain::device,100});
    // Aliasing shared_ptrs deduplicate by retained control block, not get().
    std::shared_ptr<const void> alias(weights,&(*weights)[1]);
    old.add({alias,0,Inventory::Domain::device,100});
    old.add({private_a,0,Inventory::Domain::device,20});
    next.add({weights,0,Inventory::Domain::device,100});
    next.add({private_b,0,Inventory::Domain::device,20});
    next.add({private_b,1,Inventory::Domain::cuda_registered_host,8});
    auto limits=Inventory::unlimited();
    require(Inventory::peak(old,next,limits)[0]==140,"old private owner omitted or weights duplicated");
    limits[0]=139;bool refused=false;
    try{Inventory::peak(old,next,limits);}catch(const std::runtime_error&){refused=true;}
    require(refused && old.totals()[0]==120,"failed assessment mutated inventory");
    refused=false;try{next.add({weights,0,Inventory::Domain::device,101});}
    catch(const std::invalid_argument&){refused=true;}require(refused,"same slot resized while live");
    weights.reset();alias.reset();require(!survivor.expired(),"model owner retired under execution");
    old={};require(!survivor.expired(),"replacement lost shared weights");next={};
    require(survivor.expired(),"inventory retained retired owner");
    Inventory overflow;overflow.add({private_a,0,Inventory::Domain::device,UINT64_MAX});
    overflow.add({private_b,0,Inventory::Domain::device,1});refused=false;
    try{overflow.totals();}catch(const std::overflow_error&){refused=true;}require(refused,"sum overflow");
    std::cout<<"RESOURCE_INVENTORY PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<phase<<": "<<e.what()<<'\n';return 1;}}
