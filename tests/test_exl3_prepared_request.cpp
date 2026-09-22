#include "exl3/dflash2_execution.h"
#include "exl3/vision_model.h"
#include "exl3/exl3_frontend_resources.h"
#include "exl3/encoded_media_cache.h"
#include <fstream>
#include <iostream>
#include <set>

using namespace ninfer;
using namespace ninfer::exl3;
namespace family=ninfer::targets::qwen3_6;
static void need(bool ok,const char* label) {
    if(!ok)throw std::runtime_error(label);
    std::cout<<"PASS "<<label<<std::endl;
}
template<class F> static bool refuses(F&& f) {try{f();return false;}catch(const std::exception&){return true;}}
static std::vector<std::uint8_t> gradient_ppm(int width,int height,std::uint8_t phase) {
    const std::string header="P6\n"+std::to_string(width)+' '+std::to_string(height)+"\n255\n";
    std::vector<std::uint8_t> bytes(header.begin(),header.end());
    bytes.reserve(bytes.size()+static_cast<std::size_t>(width)*height*3);
    for(std::size_t index=0;index<static_cast<std::size_t>(width)*height;++index) {
        bytes.push_back(static_cast<std::uint8_t>(index+phase));
        bytes.push_back(static_cast<std::uint8_t>(index*3+phase));
        bytes.push_back(static_cast<std::uint8_t>(index*7+phase));
    }
    return bytes;
}
static PromptInput prepared_image_input(std::vector<std::uint8_t> bytes,const char* name) {
    OwnedMedia media;media.bytes=std::move(bytes);
    media.media_type="image/x-portable-pixmap";media.source_name=name;
    PromptInput input;input.options.enable_thinking=false;
    input.messages.push_back({ChatRole::User,{{MessagePartKind::Text,"Describe this image."},
        {MessagePartKind::Media,"",std::move(media)}}});
    return input;
}
struct SyntheticVideoFixture {
    std::shared_ptr<family::PreparedMediaPayload> payload;
    family::VisionItem item;
    family::VisionItemControl control;
};
static SyntheticVideoFixture synthetic_video_fixture(
    const family::PreparedPromptData& data,const family::VisionItemControl& image_control) {
    const auto& image_payload=*data.media_payloads.at(0);
    SyntheticVideoFixture result;
    result.payload=std::make_shared<family::PreparedMediaPayload>();
    result.payload->storage=image_payload.storage;
    result.payload->preprocess=image_payload.preprocess;
    result.payload->preprocess.video_fps=2.0;
    result.payload->preprocess.video_min_frames=2;
    result.payload->preprocess.video_max_frames=4;
    result.payload->patch_elements=image_payload.patch_elements*2;
    result.payload->patches=std::make_unique<std::uint16_t[]>(result.payload->patch_elements);
    std::copy(image_payload.span().begin(),image_payload.span().end(),result.payload->mutable_span().begin());
    std::copy(image_payload.span().begin(),image_payload.span().end(),
        result.payload->mutable_span().begin()+static_cast<std::ptrdiff_t>(image_payload.patch_elements));

    result.item=data.vision_items.at(0);
    result.item.modality=family::PromptModality::Video;
    result.item.preprocess=result.payload->preprocess;
    result.item.grid.temporal=2;
    result.item.patch_count*=2;
    result.item.timestamps={0.25,0.75};
    const auto first=result.item.token_spans.at(0);
    result.item.token_spans={first,{first.begin+first.count,first.count}};

    result.control=image_control;
    result.control.modality=family::PromptModality::Video;
    result.control.grid.temporal=2;
    result.control.patch_count*=2;
    result.control.merged_count*=2;
    result.control.segment_count=2;
    result.control.scatter_indices.clear();
    for(const auto& span:result.item.token_spans)
        for(std::size_t offset=0;offset<span.count;++offset)
            result.control.scatter_indices.push_back(static_cast<std::int32_t>(span.begin+offset));
    const auto patches=image_control.patch_count;
    result.control.position_ids.resize(patches*4);
    for(std::size_t axis=0;axis<2;++axis)for(std::size_t frame=0;frame<2;++frame)
        std::copy_n(image_control.position_ids.begin()+static_cast<std::ptrdiff_t>(axis*patches),
            patches,result.control.position_ids.begin()+static_cast<std::ptrdiff_t>((axis*2+frame)*patches));
    result.control.position_table_indices.insert(result.control.position_table_indices.end(),
        image_control.position_table_indices.begin(),image_control.position_table_indices.end());
    result.control.position_table_weights.insert(result.control.position_table_weights.end(),
        image_control.position_table_weights.begin(),image_control.position_table_weights.end());
    return result;
}
int main(int argc,char** argv) {try {
    if(argc!=4)throw std::invalid_argument("target draft image");
    auto frontend=family::make_frontend(load_pinned_frontend_resources(argv[1]),
        {.vision_patch_storage=VisionPatchStorage::Float16,.vision_enabled=true,.max_context=512});
    std::ifstream file(argv[3],std::ios::binary);
    OwnedMedia image;image.bytes.assign(std::istreambuf_iterator<char>(file),{});image.media_type="image/png";
    PromptInput input;input.options.enable_thinking=false;
    input.messages.push_back({ChatRole::User,{{MessagePartKind::Text,"Describe this image."},{MessagePartKind::Media,"",image}}});
    auto prompt=frontend.prepare(input);
    const auto& data=family::PreparedPromptAccess::view(prompt);
    Exl3PreparedIdentity identity(data);
    {
        family::PreparedPromptData text;
        text.token_ids.resize(256,1);
        text.context_cache.opportunities={
            {PromptCacheMarkerKind::SharedStablePrefix,128,0},
            {PromptCacheMarkerKind::SharedStablePrefix,64,1},
            {PromptCacheMarkerKind::PrivateLongAnchor,192,2},
            {PromptCacheMarkerKind::SharedStablePrefix,63,3}};
        need(exl3_declared_prefix_frontier(text,256)==128 &&
            exl3_declared_prefix_frontier(text,100)==64,
            "declared checkpoint selection confused order, kind or upper frontier");
        text.identity.rewrite_checkpoint=family::RewriteCheckpointSpec{family::RewriteCheckpointKind::TurnClosure,100};
        need(exl3_declared_prefix_frontier(text,256)==64,"declared checkpoint crossed rewrite frontier");
        text.identity.reusable=false;
        need(exl3_declared_prefix_frontier(text,256)==0,"nonreusable input selected declared checkpoint");
        need(exl3_declared_prefix_frontier(data,data.token_ids.size())==0,"typed input entered declared text checkpoint route");
        text.identity.reusable=true;text.context_cache.opportunities.front().frontier=257;
        need(refuses([&]{(void)exl3_declared_prefix_frontier(text,256);}),
            "stale declared checkpoint exceeded prepared input");
    }
    need(identity.equals(Exl3PreparedIdentity(data)),"prepared identity exact repeat");
    {
        need(!data.media_payloads.empty() && data.media_payloads.front(),"media retention fixture payload");
        auto owned=std::make_shared<family::PreparedMediaPayload>();
        const auto& original=*data.media_payloads.front();
        owned->storage=original.storage;owned->preprocess=original.preprocess;
        owned->patch_elements=original.patch_elements;
        owned->patches=std::make_unique<std::uint16_t[]>(owned->patch_elements);
        std::copy(original.span().begin(),original.span().end(),owned->mutable_span().begin());
        std::weak_ptr<const family::PreparedMediaPayload> weak=owned;
        auto prepared_copy=data;prepared_copy.media_payloads.front()=owned;
        const Exl3PreparedIdentity metadata_only(prepared_copy);
        const auto patch_address=reinterpret_cast<std::uintptr_t>(owned->patches.get());
        const auto patch_bytes=owned->patch_elements*sizeof(std::uint16_t);
        std::size_t retained_identity_bytes=0;
        metadata_only.visit_allocations([&](const void* pointer,std::size_t bytes) {
            const auto address=reinterpret_cast<std::uintptr_t>(pointer);
            need(!(address>=patch_address && address-patch_address<patch_bytes),
                "identity allocation visitor included preprocessing patch storage");
            retained_identity_bytes+=bytes;
        });
        auto replay_input=prepared_copy; // Separate owner of the replay-required inputs.
        {
            family::PreparedPromptData aliases;
            aliases.media_payloads={owned,owned,{}};
            std::size_t bytes=0,visits=0;
            aliases.visit_media_payload_allocations([&](const void*,std::size_t extent){bytes+=extent;++visits;});
            const auto replay=aliases.media_payload_retention();
            need(visits==(owned->patch_elements?2U:1U) && bytes==sizeof(family::PreparedMediaPayload)+patch_bytes &&
                replay.mode==family::MediaPayloadRetentionMode::ReplayRequired &&
                replay.live_items==2 && replay.retained_bytes==bytes,
                "shared prepared payload counted once per input");
            auto peer=aliases;
            const auto union_bytes=[&] {
                std::set<const void*> seen;std::size_t total=0;
                const auto visit=[&](const void* pointer,std::size_t extent) {
                    if(seen.insert(pointer).second)total+=extent;
                };
                aliases.visit_media_payload_allocations(visit);peer.visit_media_payload_allocations(visit);
                return total;
            };
            need(union_bytes()==bytes,"cross-input media owner union double counted shared payload");
            aliases.release_all_media_payloads();visits=0;
            aliases.visit_media_payload_allocations([&](const void*,std::size_t){++visits;});
            const auto identity_only=aliases.media_payload_retention();
            need(visits==0 && identity_only.mode==family::MediaPayloadRetentionMode::IdentityOnly &&
                identity_only.live_items==0 && identity_only.retained_bytes==0,
                "released input retained payload allocation charges");
            need(union_bytes()==bytes,"one input release erased surviving peer payload charge");
            peer.release_all_media_payloads();
            need(union_bytes()==0,"released input union retained payload charge");
        }
        owned.reset();prepared_copy.release_all_media_payloads();
        need(!weak.expired() && metadata_only.equals(identity),
            "replay input retains payload while identity remains independent");
        replay_input.release_all_media_payloads();
        need(weak.expired() && metadata_only.equals(identity),
            "identity-only metadata retained preprocessing payload");
        need(replay_input.media_payloads.size()==data.media_payloads.size() &&
            std::all_of(replay_input.media_payloads.begin(),replay_input.media_payloads.end(),
                [](const auto& payload){return !payload;}) &&
            metadata_only.equals(Exl3PreparedIdentity(replay_input)),
            "media release changed typed descriptors or retained payload owners");
        std::size_t identity_bytes_after_release=0;
        metadata_only.visit_allocations([&](const void*,std::size_t bytes){identity_bytes_after_release+=bytes;});
        need(identity_bytes_after_release==retained_identity_bytes,
            "payload release changed identity-only allocation accounting");
        auto changed_descriptor=data;changed_descriptor.vision_items.front().content_digest[0]^=1;
        need(!metadata_only.equals(Exl3PreparedIdentity(changed_descriptor)),
            "payload retirement weakened media content identity");
    }
    {
        auto malformed=std::make_shared<family::PreparedMediaPayload>();
        family::PreparedPromptData owner;owner.media_payloads={malformed};
        for(bool overflow:{false,true}) {
            malformed->patch_elements=overflow?std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t)+1:1;
            unsigned visits=0;bool refused=false;
            try{owner.visit_media_payload_allocations([&](const void*,std::size_t){++visits;});}
            catch(const std::overflow_error& error) {
                refused=overflow && std::string_view(error.what())=="prepared media patch byte count overflows size_t";
            } catch(const std::invalid_argument& error) {
                refused=!overflow && std::string_view(error.what())=="prepared media patch storage missing";
            }
            need(refused && visits==0,"malformed media payload emitted allocation charges");
        }
    }
    for(int change=0;change<11;++change) {
        auto changed=data;
        switch(change) {
        case 0:changed.vision_items[0].content_digest[0]^=1;break;
        case 1:++changed.positions[0];break;
        case 2:++changed.vision_items[0].grid.height;break;
        case 3:changed.vision_items[0].timestamps.push_back(0.5);break;
        case 4:changed.vision_items[0].modality=family::PromptModality::Video;break;
        case 5:++changed.vision_items[0].patch_count;break;
        case 6:++changed.rope_delta;break;
        case 7:changed.identity.rewrite_execution_frontiers.push_back(1);break;
        case 8:changed.token_types[0]=1;break;
        case 9:changed.vision_items[0].patch_storage=VisionPatchStorage::BFloat16;break;
        case 10:++changed.vision_items[0].preprocess.minimum_pixels;break;
        }
        need(!identity.equals(Exl3PreparedIdentity(changed)),"same tokens changed typed metadata miss");
    }
    {
        auto detached=data;
        auto payload=std::make_shared<family::PreparedMediaPayload>();
        payload->storage=data.media_payloads[0]->storage;
        payload->preprocess=data.media_payloads[0]->preprocess;
        payload->patch_elements=data.media_payloads[0]->patch_elements;
        payload->patches=std::make_unique<std::uint16_t[]>(payload->patch_elements);
        detached.media_payloads[0]=payload;
        ++detached.vision_items[0].preprocess.maximum_pixels;
        need(!detached.media_payload_identity_valid(),
            "detached media preprocessing metadata matched its payload owner");
    }
    {
        auto replay=Exl3PreparedMediaReplay::create(data);
        need(replay.matches_source(data) && replay.data().media_payloads[0].get()==data.media_payloads[0].get() &&
                replay.control().items.size()==data.vision_items.size(),
            "typed media replay lost owning payload or execution partitions");
        auto missing=data;missing.media_payloads[0].reset();
        need(refuses([&]{(void)Exl3PreparedMediaReplay::create(missing);}),
            "typed media replay accepted missing encoder payload");
        auto changed_mrope=data;++changed_mrope.positions[0];
        need(!replay.matches_source(changed_mrope),
            "typed media replay accepted changed MRoPE positions");
        auto changed_partition=data;++changed_partition.vision_items[0].token_spans[0].count;
        bool partition_matched=false;
        try{partition_matched=replay.matches_source(changed_partition);}catch(const std::exception&){}
        need(!partition_matched && refuses([&]{
                (void)Exl3PreparedMediaReplay::create(changed_partition);
            }),"typed media replay accepted changed execution partition");
    }
    {
        const auto controls=family::build_vision_control(
            data,family::plan_vision_control(data),0);
        need(controls.items.size()==1,"encoded cache entry control fixture");
        auto encoder=std::make_shared<int>(1);
        auto entry=std::make_shared<Exl3EncodedMediaEntry>(
            data.media_payloads[0],data.vision_items[0],encoder,controls.items[0]);
        need(entry->completion()==Exl3EncodedMediaEntry::Completion::pending &&
                refuses([&]{(void)entry->ready_result();}),
            "pending encoded cache entry was reusable");
        auto wrong_encoder=std::make_shared<int>(2);
        need(!entry->matches(data.media_payloads[0],data.vision_items[0],wrong_encoder,controls.items[0]),
            "encoded cache entry accepted wrong encoder owner");
        auto changed_control=controls.items[0];++changed_control.grid.width;
        need(!entry->matches(data.media_payloads[0],data.vision_items[0],encoder,changed_control),
            "encoded cache entry accepted changed control geometry");
        auto clone=std::make_shared<family::PreparedMediaPayload>();
        clone->storage=data.media_payloads[0]->storage;
        clone->preprocess=data.media_payloads[0]->preprocess;
        clone->patch_elements=data.media_payloads[0]->patch_elements;
        clone->patches=std::make_unique<std::uint16_t[]>(clone->patch_elements);
        need(!entry->matches(clone,data.vision_items[0],encoder,controls.items[0]),
            "encoded cache entry accepted a different patch owner");
        std::vector<float> encoded(controls.items[0].merged_count*5120ULL,0.25f);
        entry->publish(std::move(encoded));
        const auto ready=entry->ready_result();
        need(entry->completion()==Exl3EncodedMediaEntry::Completion::ready && ready &&
                ready->patch_count==controls.items[0].patch_count &&
                ready->merged_count==controls.items[0].merged_count &&
                ready->embeddings.size()==controls.items[0].merged_count*5120ULL &&
                refuses([&]{entry->publish({});}),
            "encoded cache entry readiness/geometry was mutable");
        auto failed=std::make_shared<Exl3EncodedMediaEntry>(
            data.media_payloads[0],data.vision_items[0],encoder,controls.items[0]);
        failed->fail();
        need(failed->completion()==Exl3EncodedMediaEntry::Completion::failed &&
                refuses([&]{(void)failed->ready_result();}),
            "failed encoded cache entry became reusable");
        auto forged=std::make_shared<family::PreparedMediaPayload>();
        forged->storage=data.media_payloads[0]->storage;
        forged->preprocess=data.media_payloads[0]->preprocess;
        ++forged->preprocess.minimum_pixels;
        forged->patch_elements=data.media_payloads[0]->patch_elements;
        forged->patches=std::make_unique<std::uint16_t[]>(forged->patch_elements);
        need(refuses([&]{Exl3EncodedMediaEntry invalid(
                forged,data.vision_items[0],encoder,controls.items[0]);}),
            "forged payload association constructed encoded cache entry");
        {
            auto video=synthetic_video_fixture(data,controls.items[0]);
            Exl3EncodedMediaEntry valid_video(video.payload,video.item,encoder,video.control);
            auto changed_time=video.item;changed_time.timestamps[1]+=0.125;
            need(!valid_video.matches(video.payload,changed_time,encoder,video.control),
                "same video pixels with different timestamps reused an entry");
            auto reordered=video.item;std::swap(reordered.timestamps[0],reordered.timestamps[1]);
            need(refuses([&]{Exl3EncodedMediaEntry invalid(
                    video.payload,reordered,encoder,video.control);}),
                "reordered video frames constructed an encoded entry");
            auto partial=video.item;partial.timestamps.pop_back();
            need(refuses([&]{Exl3EncodedMediaEntry invalid(
                    video.payload,partial,encoder,video.control);}),
                "partial video timestamp ownership constructed an entry");
            auto crossed=video.control;++crossed.scatter_indices.back();
            need(refuses([&]{Exl3EncodedMediaEntry invalid(
                    video.payload,video.item,encoder,crossed);}),
                "video span/control association crossed frame ownership");
        }
        {
            RetainedHostAllocationLedger replay_ledger,output_ledger;
            auto retained_payload=std::make_shared<family::PreparedMediaPayload>();
            retained_payload->storage=data.media_payloads[0]->storage;
            retained_payload->preprocess=data.media_payloads[0]->preprocess;
            retained_payload->patch_elements=data.media_payloads[0]->patch_elements;
            retained_payload->patches=std::make_unique<std::uint16_t[]>(retained_payload->patch_elements);
            std::copy(data.media_payloads[0]->span().begin(),data.media_payloads[0]->span().end(),
                retained_payload->mutable_span().begin());
            const auto replay_bytes=sizeof(family::PreparedMediaPayload)+
                retained_payload->patch_elements*sizeof(std::uint16_t);
            const auto output_bytes=controls.items[0].merged_count*5120ULL*sizeof(float);
            auto retained=std::make_shared<Exl3EncodedMediaEntry>(retained_payload,
                data.vision_items[0],encoder,controls.items[0],replay_ledger.acquire(replay_bytes),
                output_ledger.acquire(output_bytes));
            std::weak_ptr<const family::PreparedMediaPayload> payload_weak=retained_payload;
            retained_payload.reset();
            retained->publish(std::vector<float>(controls.items[0].merged_count*5120ULL,0.5f));
            auto survivor=retained->ready_result();
            retained.reset();
            need(payload_weak.expired() && replay_ledger.bytes()==0 &&
                    output_ledger.bytes()==output_bytes && survivor->embeddings.size()*sizeof(float)==output_bytes,
                "encoded result survivor retained replay payload wrapper");
            survivor.reset();
            need(output_ledger.bytes()==0,"encoded result survivor leaked output credit");

            auto partial=std::make_shared<Exl3EncodedMediaEntry>(data.media_payloads[0],
                data.vision_items[0],encoder,controls.items[0],replay_ledger.acquire(
                    sizeof(family::PreparedMediaPayload)+data.media_payloads[0]->patch_elements*sizeof(std::uint16_t)),
                output_ledger.acquire(output_bytes));
            need(refuses([&]{partial->publish(std::vector<float>(1,0.0f));}) &&
                    partial->completion()==Exl3EncodedMediaEntry::Completion::pending,
                "partial encoded fill published or consumed reservation");
            partial->fail();partial.reset();
            need(replay_ledger.bytes()==0 && output_ledger.bytes()==0,
                "failed partial fill retained replay/output credits");
        }
    }
    const auto span=data.vision_items[0].token_spans[0];
    need(!identity.prefix_equals(identity,span.begin+1),"split media frontier refused");
    need(!identity.prefix_positions_equal(identity,span.begin+1),
        "typed position proof accepted a split media frontier");
    need(identity.prefix_equals(identity,span.begin+span.count),"complete media frontier accepted");
    auto generated=identity.append_text(3);
    auto extended=data;
    const auto original=extended.token_ids.size();extended.token_ids.resize(original+3);
    extended.token_types.resize(original+3,0);extended.positions.clear();
    for(int axis=0;axis<3;++axis) {
        extended.positions.insert(extended.positions.end(),data.positions.begin()+axis*original,data.positions.begin()+(axis+1)*original);
        for(int i=0;i<3;++i)extended.positions.push_back(static_cast<int>(original)+i+data.rope_delta);
    }
    need(generated->equals(Exl3PreparedIdentity(extended)),"generated identity all position axes");
    std::size_t metadata_bytes=0;std::set<const void*> allocations;
    identity.visit_allocations([&](const void* p,std::size_t bytes){need(p&&bytes&&allocations.insert(p).second,"distinct metadata allocation");metadata_bytes+=bytes;});
    need(metadata_bytes>=data.token_ids.size()*13,"retained metadata byte coverage");
    {
        RetainedDescriptorLedger metadata;
        const auto before=bounded_shared_live_blocks_for_test<Exl3PreparedIdentity>();bool refused=false;
        try{auto invalid=Exl3PreparedIdentity::create(data,[&](std::uint64_t bytes){return metadata.acquire(bytes-1);});}
        catch(const std::invalid_argument&){refused=true;}
        need(refused && metadata.bytes()==0 && bounded_shared_live_blocks_for_test<Exl3PreparedIdentity>()==before,
            "identity reservation refusal allocated storage or leaked credit");
        auto owned=Exl3PreparedIdentity::create(data);
        const auto bytes=owned->metadata_bytes();
        need(owned->equals(identity) && Exl3PreparedIdentity::attach_metadata_credit(owned,metadata.acquire(bytes)),
            "bounded prepared identity changed semantics or refused credit");
        auto extended_identity=owned->append_text(1);
        need(!Exl3PreparedIdentity::metadata_credit_belongs_to(extended_identity,metadata),"identity clone inherited credit");
        {
            RetainedDescriptorLedger appended;
            const auto live=bounded_shared_live_blocks_for_test<Exl3PreparedIdentity>();
            bool append_refused=false;
            try{auto invalid=owned->append_text(1,[&](std::uint64_t count){return appended.acquire(count-1);});}
            catch(const std::invalid_argument&){append_refused=true;}
            need(append_refused && appended.bytes()==0 && owned->equals(identity) &&
                bounded_shared_live_blocks_for_test<Exl3PreparedIdentity>()==live,
                "identity append reservation refusal allocated copy or changed parent");
            auto credited=owned->append_text(1,[&](std::uint64_t count){return appended.acquire(count);});
            need(credited->equals(*extended_identity) && appended.bytes()==credited->metadata_bytes(),
                "planned identity append changed semantics or metadata extent");
        }
        std::weak_ptr<Exl3PreparedIdentity> weak=owned;owned.reset();
        need(weak.expired() && metadata.bytes()==bounded_shared_allocation_bytes<Exl3PreparedIdentity>(),
            "identity final strong retirement retained vectors or lost weak block");
        weak.reset();need(metadata.bytes()==0,"identity final weak metadata leaked");
    }
    if(const auto* mode=std::getenv("NINFER_TEST_IDENTITY_METADATA_CEILING");mode && std::string_view(mode)=="1") {
        Exl3HostResidentSet registry(2ULL<<30,1ULL<<30);
        const auto domain=static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata);
        auto sample=Exl3PreparedIdentity::create(data);
        const auto parent_bytes=sample->metadata_bytes();
        const auto child_bytes=sample->append_text(1)->metadata_bytes();sample.reset();
        auto limits=Exl3ResourceInventory::unlimited();limits[domain]=parent_bytes;
        registry.set_resource_limits(limits);
        const auto reserve_identity=[&](std::uint64_t bytes){return registry.reserve_host_metadata_lifetime(bytes);};
        auto parent=Exl3PreparedIdentity::create(data,reserve_identity);
        const auto revision=registry.revision();
        registry.admit_host_metadata_lifetime(parent,parent_bytes,&Exl3PreparedIdentity::metadata_credit_belongs_to,
            &Exl3PreparedIdentity::attach_metadata_credit);
        need(registry.revision()==revision,"identity publication duplicated reservation");
        const auto live=bounded_shared_live_blocks_for_test<Exl3PreparedIdentity>();bool refused=false;
        try{auto invalid=parent->append_text(1,reserve_identity);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        need(refused && parent->equals(identity) && bounded_shared_live_blocks_for_test<Exl3PreparedIdentity>()==live &&
            registry.retained_resource_units()[domain]==parent_bytes,"identity ceiling refusal changed parent or leaked storage");
        parent.reset();
        need(registry.retained_resource_units()[domain]==0,"identity refusal fixture retained parent metadata");
        registry.close();
        Exl3HostResidentSet combined_registry(2ULL<<30,1ULL<<30);
        limits[domain]=parent_bytes+child_bytes;combined_registry.set_resource_limits(limits);
        const auto reserve_combined=[&](std::uint64_t bytes){return combined_registry.reserve_host_metadata_lifetime(bytes);};
        parent=Exl3PreparedIdentity::create(data,reserve_combined);
        auto child=parent->append_text(1,reserve_combined);
        need(combined_registry.retained_resource_units()[domain]==parent_bytes+child_bytes,"identity exact combined ceiling diverged");
        std::weak_ptr<Exl3PreparedIdentity> parent_weak=parent;
        std::weak_ptr<const Exl3PreparedIdentity> child_weak=child;
        parent.reset();
        need(parent_weak.expired() && child->size()==identity.size()+1 &&
            combined_registry.retained_resource_units()[domain]==bounded_shared_allocation_bytes<Exl3PreparedIdentity>()+child_bytes,
            "identity parent retirement retained vectors or damaged child");
        parent_weak.reset();child.reset();
        need(child_weak.expired() && combined_registry.retained_resource_units()[domain]==bounded_shared_allocation_bytes<Exl3PreparedIdentity>(),
            "identity child retirement lost weak block");
        child_weak.reset();need(combined_registry.retained_resource_units()[domain]==0,"identity combined ceiling credit leaked");
        combined_registry.close();return 0;
    }
    if(std::getenv("NINFER_PREPARED_IDENTITY_CPU_ONLY"))return 0;

    auto model=Exl3TextModel::load(argv[1],512);
    auto draft=Exl3Dflash2DraftModel::load(argv[2]);
    Exl3VisionModel vision_model(argv[1]);Exl3VisionContext vision(vision_model,512);
    auto prepare=model->create_context(true);prepare->prepare_continuation(8);
    RetainedDescriptorLedger request_metadata;
    const auto request_blocks=bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>();
    prepare->set_request_metadata_reservation([](std::uint64_t)->RetainedDescriptorLedger::Ticket {
        throw std::bad_alloc();
    });
    need(refuses([&]{Exl3VeriCacheRequest::initialize_prepared_numeric(*prepare,vision,prompt);}) &&
        prepare->position()==0 && bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>()==request_blocks,
        "prepared request reservation refusal allocated root or changed context");
    prepare->set_request_metadata_reservation([&](std::uint64_t bytes){return request_metadata.acquire(bytes);});
    int calls=0;
    need(refuses([&]{Exl3VeriCacheRequest::initialize_prepared_numeric(*prepare,vision,prompt,[&]{return ++calls==4;});})&&prepare->position()==0,
        "encoder cancellation leaves no request root");
    need(request_metadata.bytes()==0,"cancelled prepared request retained metadata credit");
    auto root=Exl3VeriCacheRequest::initialize_prepared_numeric(*prepare,vision,prompt);
    std::uint64_t tap_metadata=0;
    root->visit_tap_block_owners([&](const auto& block){
        need(Exl3VeriCacheRequest::tap_block_metadata_credit_belongs_to(block,request_metadata),"prepared tap metadata uncredited");
        tap_metadata+=Exl3VeriCacheRequest::tap_block_metadata_bytes();
    });
    root->state()->visit_page_metadata_owners([&](const auto& page){
        need(Exl3ExactKVPage::metadata_credit_belongs_to(page,request_metadata),"prepared page metadata uncredited");
        tap_metadata+=Exl3ExactKVPage::metadata_bytes();
    });
    root->visit_token_control_owners([&](const auto& node){
        need(Exl3TokenHistory::control_credit_belongs_to(node,request_metadata),"prepared token control uncredited");
        tap_metadata+=Exl3TokenHistory::node_metadata_bytes();
    });
    need(Exl3VeriCacheRequest::request_metadata_credit_belongs_to(root,request_metadata) &&
        Exl3VeriCacheRequest::media_tap_origin_credit_belongs_to(root->media_tap_origin(),request_metadata) &&
        request_metadata.bytes()==root->request_metadata_bytes()+bounded_shared_allocation_bytes<int>()+tap_metadata+
            root->prepared_metadata_owner()->metadata_bytes()+root->media_tap_origin_metadata_bytes(),
        "prepared root metadata reservation missing");
    {
        auto full_descriptor=Exl3PreparedMediaReplay::create(extended);
        auto full_context=model->create_context(true);full_context->prepare_continuation(8);
        auto full=Exl3VeriCacheRequest::replay_prepared_numeric(
            *full_context,vision,full_descriptor);
        auto ancestor_descriptor=Exl3PreparedMediaReplay::create(extended,root);
        auto ancestor_context=model->create_context(true);ancestor_context->prepare_continuation(8);
        auto attached=Exl3VeriCacheRequest::replay_prepared_numeric(
            *ancestor_context,vision,ancestor_descriptor);
        need(attached->state()->same_payload(*full->state()) &&
                attached->prepared_identity()->equals(*full->prepared_identity()) &&
                attached->token_count()==extended.token_ids.size(),
            "valid typed ancestor changed suffix target state or positions");
        auto changed_offset=extended;
        for(std::size_t axis=0;axis<3;++axis)
            ++changed_offset.positions[axis*changed_offset.token_ids.size()+data.token_ids.size()];
        need(refuses([&]{(void)Exl3PreparedMediaReplay::create(changed_offset,root);}),
            "typed ancestor accepted a changed suffix rope offset");

        OwnedMedia second;second.bytes=gradient_ppm(64,64,41);
        second.media_type="image/x-portable-pixmap";second.source_name="second-prefix.ppm";
        PromptInput two_images;two_images.options.enable_thinking=false;
        two_images.messages.push_back({ChatRole::User,{
            {MessagePartKind::Text,"Compare these images."},
            {MessagePartKind::Media,"",image},
            {MessagePartKind::Text," Then compare another."},
            {MessagePartKind::Media,"",std::move(second)}}});
        auto two_prompt=frontend.prepare(two_images);
        const auto& two_data=family::PreparedPromptAccess::view(two_prompt);
        need(two_data.vision_items.size()==2 &&
                refuses([&]{(void)Exl3PreparedMediaReplay::create(two_data,root);}),
            "added second image bypassed exact typed ancestor proof");
    }
    {
        const auto original_cache=vision.encoded_media_cache_stats();
        const auto original_control=family::build_vision_control(
            data,family::plan_vision_control(data),0);
        need(original_control.items.size()==1,"prepared cache integration control extent");
        {
            auto video=synthetic_video_fixture(data,original_control.items[0]);
            const std::vector<std::uint16_t> frozen_patches(
                video.payload->span().begin(),video.payload->span().end());
            const auto before_video_cancel=vision.encoded_media_cache_stats();
            int video_cancel_calls=0;
            need(refuses([&]{(void)vision.encode_prepared_cached(video.payload,video.item,
                    video.control,[&]{return ++video_cancel_calls==4;});}),
                "active multiframe encoder cancellation returned a result");
            const auto after_video_cancel=vision.encoded_media_cache_stats();
            need(after_video_cancel.failed_producers==before_video_cancel.failed_producers+1 &&
                    after_video_cancel.cancelled_consumers==before_video_cancel.cancelled_consumers+1 &&
                    after_video_cancel.entries==before_video_cancel.entries &&
                    std::equal(frozen_patches.begin(),frozen_patches.end(),video.payload->span().begin()),
                "cancelled multiframe encode retained a partial entry or mutated frame payload");
        }
        need(refuses([&]{(void)vision.encode_prepared_cached(
                data.media_payloads[0],data.vision_items[0],original_control.items[0],
                []{return true;});}),
            "cancelled encoded-media consumer returned a ready result");
        auto after_cancel=vision.encoded_media_cache_stats();
        need(after_cancel.cancelled_consumers==original_cache.cancelled_consumers+1 &&
                after_cancel.entries==original_cache.entries,
            "cancelled encoded-media consumer changed cache ownership");
        auto repeated_context=model->create_context(true);repeated_context->prepare_continuation(8);
        auto repeated=Exl3VeriCacheRequest::initialize_prepared_numeric(
            *repeated_context,vision,prompt);
        auto after_repeat=vision.encoded_media_cache_stats();
        need(after_repeat.hits==after_cancel.hits+1 &&
                after_repeat.misses==after_cancel.misses &&
                repeated->state()->same_payload(*root->state()) &&
                repeated->prepared_identity()->equals(*root->prepared_identity()),
            "same-process repeated image did not reuse exact immutable encoder output");

        auto base_prompt=frontend.prepare(prepared_image_input(
            gradient_ppm(64,64,0),"encoded-base.ppm"));
        auto changed_prompt=frontend.prepare(prepared_image_input(
            gradient_ppm(64,64,17),"encoded-changed.ppm"));
        auto grid_prompt=frontend.prepare(prepared_image_input(
            gradient_ppm(128,64,0),"encoded-grid.ppm"));
        const auto& base_data=family::PreparedPromptAccess::view(base_prompt);
        const auto& changed_data=family::PreparedPromptAccess::view(changed_prompt);
        const auto& grid_data=family::PreparedPromptAccess::view(grid_prompt);
        need(base_data.vision_items[0].grid.height==changed_data.vision_items[0].grid.height &&
                base_data.vision_items[0].grid.width==changed_data.vision_items[0].grid.width &&
                base_data.vision_items[0].content_digest!=changed_data.vision_items[0].content_digest &&
                (base_data.vision_items[0].grid.height!=grid_data.vision_items[0].grid.height ||
                 base_data.vision_items[0].grid.width!=grid_data.vision_items[0].grid.width),
            "changed-image and different-grid cache fixtures are not independent");
        const auto run_prepared=[&](const family::PreparedPrompt& candidate) {
            auto context=model->create_context(true);context->prepare_continuation(8);
            return Exl3VeriCacheRequest::initialize_prepared_numeric(*context,vision,candidate);
        };
        const auto before_variants=vision.encoded_media_cache_stats();
        auto base=run_prepared(base_prompt);
        auto changed_image=run_prepared(changed_prompt);
        auto changed_grid=run_prepared(grid_prompt);
        const auto after_variants=vision.encoded_media_cache_stats();
        need(after_variants.misses==before_variants.misses+3 &&
                after_variants.entries==before_variants.entries+3 &&
                !base->state()->same_payload(*changed_image->state()) &&
                !base->prepared_identity()->equals(*changed_image->prepared_identity()) &&
                !base->prepared_identity()->equals(*changed_grid->prepared_identity()),
            "changed image or different grid reused an encoded-media cache entry");
        need(root->media_tap_origin_current() && changed_image->media_tap_origin_current() &&
                !root->owns_media_tap_origin(changed_image->media_tap_origin()),
            "changed image substituted stale encoded-target tap origin");
        auto pressure_prompt=frontend.prepare(prepared_image_input(
            gradient_ppm(64,128,29),"encoded-pressure.ppm"));
        const auto before_pressure=vision.encoded_media_cache_stats();
        auto pressure=run_prepared(pressure_prompt);
        const auto after_pressure=vision.encoded_media_cache_stats();
        need(pressure && after_pressure.entries==4 &&
                after_pressure.evictions==before_pressure.evictions+1 &&
                after_pressure.retained_encoded_host_bytes>0 &&
                after_pressure.retained_replay_host_bytes>0 &&
                after_pressure.encoder_scratch_device_bytes==vision.device_bytes() &&
                after_pressure.inflight_producers==0 &&
                after_pressure.inflight_scratch_device_bytes==0,
            "encoded cache pressure or separated resource accounting diverged");
        vision.set_encoded_media_cache_limits({1,1,1});
        const auto before_quota=vision.encoded_media_cache_stats();
        need(refuses([&]{(void)run_prepared(base_prompt);}),
            "encoded cache quota refusal reached encoder/target execution");
        const auto after_quota=vision.encoded_media_cache_stats();
        need(after_quota.quota_refusals==before_quota.quota_refusals+1 &&
                after_quota.entries==0 && after_quota.retained_encoded_host_bytes==0 &&
                after_quota.retained_replay_host_bytes==0,
            "encoded cache quota refusal retained a partial fill");
        vision.set_encoded_media_cache_limits({4,64ULL<<20,64ULL<<20});
    }
    {
        auto replay=Exl3PreparedMediaReplay::create(data);
        auto replay_context=model->create_context(true);replay_context->prepare_continuation(8);
        const auto before_replay=vision.encoded_media_cache_stats();
        auto replayed=Exl3VeriCacheRequest::replay_prepared_numeric(
            *replay_context,vision,replay);
        const auto after_replay=vision.encoded_media_cache_stats();
        need(replayed->state()->same_payload(*root->state()) &&
                replayed->prepared_identity()->equals(*root->prepared_identity()) &&
                after_replay.misses==before_replay.misses+1,
            "typed replay after encoded eviction changed exact target state");
        auto cancelled_context=model->create_context(true);cancelled_context->prepare_continuation(8);
        need(refuses([&]{(void)Exl3VeriCacheRequest::replay_prepared_numeric(
                *cancelled_context,vision,replay,[]{return true;});}) &&
                cancelled_context->position()==0,
            "typed replay cancellation retained partial target state");
        std::array<std::uint16_t*,5> replay_staging{};
        struct ReplayStagingOwner {std::array<std::uint16_t*,5>& p;
            ~ReplayStagingOwner(){for(auto x:p)if(x)cudaFree(x);}} replay_staging_owner{replay_staging};
        for(auto& p:replay_staging)
            if(cudaMalloc(reinterpret_cast<void**>(&p),16ULL*5120*2)!=cudaSuccess)
                throw std::runtime_error("replay staging");
        auto original_ring=root->compact_draft(*draft,replay_staging);
        auto replay_ring=replayed->compact_draft(*draft,replay_staging);
        need(original_ring->same_projected_conditioning_for_test(*replay_ring),
            "typed replay changed private draft-ring conditioning");
    }
    prepare->set_request_metadata_reservation({});
    root.reset();prepare->reset();
    need(request_metadata.bytes()==0,"prepared admission fixture retained owners after reset");
    // The following coordinator owns an independent ledger. Build its root
    // independently rather than transplanting already-credited nested owners.
    root=Exl3VeriCacheRequest::initialize_prepared_numeric(*prepare,vision,prompt);
    need(root->prepared_identity()&&root->prepared_identity()->equals(identity),"same-process encoder binds typed root");
    need(!root->owns_shared_text_page(root->state()->shared_kv_page(0),
        root->state()->model_identity(),root->state()->rope_offset()),
        "typed root entered sequential-text page attachment");
    need(refuses([&]{root->replay_plan();}),"typed root refuses token replay");
    need(refuses([&]{(void)Exl3PreparedMediaReplay::create(data,root);}),
        "typed replay accepted a non-prefix exact ancestor");
    std::array<std::uint16_t*,5> staging{};
    struct StagingOwner {std::array<std::uint16_t*,5>& p;~StagingOwner(){for(auto x:p)if(x)cudaFree(x);}} staging_owner{staging};
    for(auto& p:staging)if(cudaMalloc(reinterpret_cast<void**>(&p),16ULL*5120*2)!=cudaSuccess)throw std::runtime_error("staging");
    root=root->compact_draft(*draft,staging);
    const auto media_origin=root->media_tap_origin();
    need(root->media_tap_origin_current() && media_origin,
        "compact draft lost exact encoded-target tap origin");
    auto tokens=root->token_suffix();
    Exl3VeriCachePrefixIndex index(4,[](auto,auto){return 7;});
    index.publish(root,"typed-research");
    need(!index.lookup(*prepare,tokens,"typed-research")&&!index.lookup_longest(*prepare,tokens,"typed-research"),"token-only lookup excludes typed roots");
    need(index.lookup(*prepare,tokens,"typed-research",&identity)==root,"typed exact cache hit after hash collision shortlist");
    need(index.lookup_longest(*prepare,tokens,"typed-research",1,&identity)==root,
        "typed longest shortlist omitted matching media root");
    auto extended_tokens=tokens;
    extended_tokens.insert(extended_tokens.end(),{17,18,19});
    need(index.lookup_longest(*prepare,extended_tokens,"typed-research",1,generated.get())==root,
        "typed longest range index omitted a complete-media ancestor");
    need(!index.lookup_longest(*prepare,
            std::span<const std::int64_t>(tokens).first(span.begin+1),
            "typed-research",1,&identity) &&
        !index.lookup_longest(*prepare,{},"typed-research",1,&identity),
        "typed longest range index crossed split-media or empty frontier");
    auto changed=data;changed.vision_items[0].content_digest[0]^=1;Exl3PreparedIdentity other(changed);
    need(!index.lookup(*prepare,tokens,"typed-research",&other),"same tokens changed image cache miss");
    need(!index.lookup_longest(*prepare,tokens,"typed-research",1,&other),
        "longest shortlist reused changed media content");
    auto changed_storage=data;
    changed_storage.vision_items[0].patch_storage=VisionPatchStorage::BFloat16;
    Exl3PreparedIdentity other_storage(changed_storage);
    need(!index.lookup(*prepare,tokens,"typed-research",&other_storage),
        "forced shortlist collision cannot alias BF16 and FP16 media roots");
    need(!index.lookup_longest(*prepare,tokens,"typed-research",1,&other_storage),
        "longest shortlist aliased media storage formats");
    need(!index.erase(*prepare,tokens,"typed-research",&other)&&!index.erase(*prepare,tokens,"typed-research"),"wrong identity cannot erase media root");
    const auto budget=index.storage_stats().accounted_bytes;
    need(!index.admit(root,"typed-research",budget-1,16ULL<<30,8ULL<<30).admitted,"complete metadata participates in cache admission");
    need(index.admit(root,"typed-research",budget,16ULL<<30,8ULL<<30).replaced,"typed cache replacement identity");
    need(index.lookup_longest(*prepare,tokens,"typed-research",1,&identity)==root &&
        !index.lookup_longest(*prepare,tokens,"typed-research",1,&other),
        "typed replacement lost longest-shortlist identity isolation");
    auto reference=model->create_context(true);reference->prepare_continuation(8);reference->restore_exact_host_state(*root->state());
    Exl3VeriCacheServingPrefixCache cache({4,1,2ULL<<30,8ULL<<30},{"media-research","pinned","pinned","greedy","typed-v6-numeric"});
    Exl3VisionContext serving_vision(vision_model,512);
    RetainedHostAllocationLedger engine_media_payloads;
    const auto reserve_media=[&](std::uint64_t replay_bytes,std::uint64_t output_bytes) {
        Exl3EncodedMediaRetentionCredits credits;
        credits.replay.emplace(engine_media_payloads.acquire(replay_bytes));
        credits.output.emplace(engine_media_payloads.acquire(output_bytes));
        return credits;
    };
    auto serving_c1=model->create_context(true);serving_c1->prepare_continuation(8);
    auto first_media=cache.prepare_media(*serving_c1,serving_vision,prompt,true,16ULL<<30,
        {},reserve_media);
    need(first_media.request && first_media.metrics.admitted &&
            first_media.request->prepared_identity() && engine_media_payloads.bytes()>0,
        "actual serving media preparation omitted typed admission or payload accounting");
    auto serving_c2=model->create_context(true);serving_c2->prepare_continuation(8);
    auto second_media=cache.prepare_media(*serving_c2,serving_vision,prompt,true,16ULL<<30,
        {},reserve_media);
    need(second_media.metrics.cache_hit &&
            second_media.metrics.reused_prompt_tokens==data.token_ids.size() &&
            second_media.request==first_media.request &&
            second_media.request->state()->same_payload(*root->state()),
        "C2 serving caller missed exact typed media root or changed target state");
    cache.reset_for_model_reload({"media-research-reset","pinned","pinned","greedy","typed-v6-numeric"});
    first_media.request.reset();second_media.request.reset();
    serving_vision.set_encoded_media_cache_limits({1,1,1});
    need(engine_media_payloads.bytes()==0,
        "typed root retirement retained encoded/replay payload credits");
    cache.reset_for_model_reload({"media-research","pinned","pinned","greedy","typed-v6-numeric"});
    Exl3VeriCacheServingCoordinator coordinator(cache,{1,1,4ULL<<30,8ULL<<30});
    auto ticket=coordinator.admit(root);auto lease=coordinator.acquire();need(bool(lease),"typed coordinator admission and VirtualLock");
    auto stale_lease=*lease;
    coordinator.yield(*lease);
    lease=coordinator.acquire();
    need(lease && lease->ticket.generation==ticket.generation &&
        lease->acquisition!=stale_lease.acquisition,"reacquire keeps cancellation ticket but changes execution ownership");
    need(refuses([&]{coordinator.yield(stale_lease);}),"prior acquisition cannot yield current media request");
    need(refuses([&]{coordinator.complete(stale_lease);}),"prior acquisition cannot retire current media request");
    Exl3Dflash2Execution execution(model->create_context(true),*draft,"typed-v6-numeric");
    execution.context().restore_exact_host_state(*root->state());
    need(execution.context().exact_host_state_resident(*root->state()),
        "typed acquisition fixture did not establish candidate residency");
    const auto typed_before=execution.stats();
    execution.acquire(*lease,&coordinator);
    const auto typed_after=execution.stats();
    need(typed_after.acquired_payload_preservations==typed_before.acquired_payload_preservations &&
        typed_after.acquired_full_resets==typed_before.acquired_full_resets+1 &&
        typed_after.acquired_draft_ring_preservations==typed_before.acquired_draft_ring_preservations &&
        typed_after.acquired_draft_ring_restores==typed_before.acquired_draft_ring_restores+1 &&
        execution.context().export_exact_host_state(nullptr,false)->same_payload(*root->state()),
        "typed acquisition bypassed full target/draft restore or changed restored media state");
    auto proposals=execution.propose(8);need(execution.stats().proposal_calls==1,"actual pinned B8 proposal call");
    auto aborted=execution.verify(proposals);execution.abort();
    need(refuses([&]{coordinator.publish_window(stale_lease,aborted.root,aborted.verification.committed_tokens);}),
        "prior acquisition cannot publish valid child of identical root");
    need(refuses([&]{execution.publish(coordinator,aborted);}),"stale media completion refused after abort");
    need(execution.context().export_exact_host_state()->same_payload(*root->state()),"abort restores immutable media root");
    const auto seed=exl3_branch_greedy(*reference);reference->decode(seed);const auto second=exl3_branch_greedy(*reference);reference->decode(second);
    const std::array<std::int64_t,2> forced{seed,(second+1)%248320};
    auto repaired=execution.verify(forced);
    need(repaired.verification.rejected&&repaired.root->state()->same_payload(*reference->export_exact_host_state()),"media forced rejection full M1 repair");
    need(repaired.root->prepared_identity()->equals(*identity.append_text(2)) &&
            repaired.root->owns_media_tap_origin(media_origin),
        "repair lost typed identity or encoded-target tap origin");
    auto publication=execution.publish(coordinator,repaired);
    need(publication.residency.page_bytes>0,"media publication resident locked");
    const std::array<std::int64_t,1> control_token{seed};auto control=execution.apply_control(control_token);reference->decode(seed);
    need(control.root->state()->same_payload(*reference->export_exact_host_state())&&control.root->prepared_identity()->equals(*identity.append_text(3)) &&
            control.root->owns_media_tap_origin(media_origin),"control state, positions and media tap origin preserved");
    execution.publish(coordinator,control);
    for(int round=0;round<2;++round) {
        auto pending=execution.verify(execution.propose(8));
        for(auto token:pending.verification.committed_tokens){need(token==exl3_branch_greedy(*reference),"real proposal L2 authority");reference->decode(token);}
        need(pending.root->state()->same_payload(*reference->export_exact_host_state()) &&
                pending.root->owns_media_tap_origin(media_origin),"real media draft state/origin M1");
        execution.publish(coordinator,pending);
    }
    coordinator.complete(execution.lease());execution.release();
    ticket=coordinator.admit(root);lease=coordinator.acquire();execution.acquire(*lease);
    auto pending=execution.verify(execution.propose(8));coordinator.cancel(ticket,true);
    need(refuses([&]{execution.publish(coordinator,pending);}),"cancelled ticket refuses late media publication");
    execution.abort();execution.release();
    auto text_context=model->create_context(true);text_context->prepare_continuation(8);
    const std::array<std::int64_t,2> text_tokens{1,2};
    auto text_root=Exl3VeriCacheRequest::initialize(*text_context,text_tokens,1);
    need(text_root->media_tap_origin_current() && !text_root->media_tap_origin(),
        "ordinary text root acquired media tap provenance");
    cache.admit_completed_authority(root,16ULL<<30);cache.reset_for_model_reload({"reload","pinned","pinned","greedy","typed-v6-numeric"});
    need(cache.roots().empty()&&root->prepared_identity()->equals(identity),"reload evicts cache without changing survivor identity");
    need(index.erase(*prepare,tokens,"typed-research",&identity),"typed erase matches exact media root");
    std::cout<<"PASS_PREPARED_MEDIA_RESEARCH quality=UNQUALIFIED public_media=false metadata_bytes="<<metadata_bytes<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
