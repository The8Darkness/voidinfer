#include "exl3/vericache_request.h"
#include "exl3/vision_model.h"
#include <ninfer/targets/qwen3_6/vision_control.h>

namespace ninfer::exl3 {
Exl3VeriCacheServingPrefixCache::Result Exl3VeriCacheServingPrefixCache::prepare_media(
    Exl3TextContext& exact,Exl3VisionContext& vision,
    const targets::qwen3_6::PreparedPrompt& prompt,bool cacheable_input,
    std::uint64_t available_physical_bytes,const std::function<bool()>& cancelled,
    const Exl3EncodedMediaRetentionReserve& reserve) {
    const auto& data=targets::qwen3_6::PreparedPromptAccess::view(prompt);
    if(!data.has_media() || data.token_ids.empty() || !data.media_payload_identity_valid())
        throw std::invalid_argument("typed serving media prompt identity");
    Exl3PreparedIdentity incoming(data);
    std::shared_ptr<const Exl3VeriCacheRequest> prefix;
    std::vector<std::int64_t> cache_tokens;
    if(cacheable_input && incoming.reusable() &&
       data.token_ids.size()>=policy_.minimum_prefix_tokens)
        cache_tokens.assign(data.token_ids.begin(),data.token_ids.end());
    std::uint64_t epoch=0;
    const auto lookup_start=std::chrono::steady_clock::now();
    {
        std::unique_lock service_lock(service_mutex_);epoch=service_epoch_;
        if(!cache_tokens.empty())
            prefix=index_.lookup_longest_prepared_positions(exact,cache_tokens,
                contract_,incoming,policy_.minimum_prefix_tokens);
    }
    Result output;auto& metrics=output.metrics;metrics.prompt_tokens=data.token_ids.size();
    metrics.lookup_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-lookup_start).count();
    metrics.cache_hit=static_cast<bool>(prefix);
    metrics.reused_prompt_tokens=prefix?prefix->token_count():0;
    metrics.executed_prompt_tokens=data.token_ids.size()-metrics.reused_prompt_tokens;
    const auto prefill_start=std::chrono::steady_clock::now();
    if(prefix && prefix->token_count()==data.token_ids.size()) {
        if(prefix->prepared_identity()->rope_delta()!=incoming.rope_delta())prefix.reset();
        else {exact.restore_exact_host_state(*prefix->state());output.request=prefix;}
    }
    if(!output.request) {
        if(prefix) {
            auto replay=Exl3PreparedMediaReplay::create(data,prefix);
            output.request=Exl3VeriCacheRequest::replay_prepared_numeric(
                exact,vision,replay,cancelled,reserve);
        } else {
            metrics.cache_hit=false;metrics.reused_prompt_tokens=0;
            metrics.executed_prompt_tokens=data.token_ids.size();
            output.request=Exl3VeriCacheRequest::initialize_prepared_numeric(
                exact,vision,prompt,cancelled,reserve);
        }
    }
    metrics.prefill_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-prefill_start).count();
    if(cancelled && cancelled())throw std::runtime_error("typed serving media preparation cancelled");
    {
        std::unique_lock service_lock(service_mutex_);
        if(epoch!=service_epoch_)throw std::runtime_error("typed serving media cache epoch changed");
        if(cacheable_input && incoming.reusable() &&
           data.token_ids.size()>=policy_.minimum_prefix_tokens) {
            const auto admission=index_.admit(output.request,contract_,policy_.byte_budget,
                available_physical_bytes,policy_.physical_reserve_bytes);
            metrics.admitted=admission.admitted;metrics.replaced=admission.replaced;
            metrics.evicted=admission.evicted;metrics.effective_budget_bytes=admission.effective_budget_bytes;
            metrics.cache_storage=admission.storage;
        } else metrics.cache_storage=index_.storage_stats();
    }
    if(!metrics.effective_budget_bytes)metrics.effective_budget_bytes=
        available_physical_bytes>policy_.physical_reserve_bytes?
            std::min(policy_.byte_budget,available_physical_bytes-policy_.physical_reserve_bytes):0;
    return output;
}

std::shared_ptr<const Exl3VeriCacheRequest> Exl3VeriCacheRequest::initialize_prepared_numeric(
    Exl3TextContext& exact,Exl3VisionContext& vision,
    const targets::qwen3_6::PreparedPrompt& prompt,const std::function<bool()>& cancelled,
    const Exl3EncodedMediaRetentionReserve& reserve) {
    namespace family=targets::qwen3_6;
    const auto& data=family::PreparedPromptAccess::view(prompt);
    return initialize_prepared_numeric_data(exact,vision,data,nullptr,{},cancelled,reserve);
}

std::shared_ptr<const Exl3VeriCacheRequest> Exl3VeriCacheRequest::replay_prepared_numeric(
    Exl3TextContext& exact,Exl3VisionContext& vision,
    const Exl3PreparedMediaReplay& replay,const std::function<bool()>& cancelled,
    const Exl3EncodedMediaRetentionReserve& reserve) {
    if(!replay.matches_source(replay.data()))
        throw std::invalid_argument("typed media replay authority changed");
    if(replay.ancestor() && replay.ancestor()->state()->model_identity()!=exact.model_identity())
        throw std::invalid_argument("typed media replay ancestor model mismatch");
    return initialize_prepared_numeric_data(exact,vision,replay.data(),&replay.control(),
        replay.ancestor(),cancelled,reserve);
}

std::shared_ptr<const Exl3VeriCacheRequest> Exl3VeriCacheRequest::initialize_prepared_numeric_data(
    Exl3TextContext& exact,Exl3VisionContext& vision,
    const targets::qwen3_6::PreparedPromptData& data,
    const targets::qwen3_6::VisionControl* replay_control,
    std::shared_ptr<const Exl3VeriCacheRequest> ancestor,
    const std::function<bool()>& cancelled,
    const Exl3EncodedMediaRetentionReserve& reserve) {
    namespace family=targets::qwen3_6;
    const auto n=data.token_ids.size();
    if(!data.has_media() || !n || n>static_cast<std::size_t>(exact.max_context()) ||
        exact.continuation_capacity()<8 || !data.media_payload_identity_valid())
        throw std::invalid_argument("prepared media request extent");
    auto identity=Exl3PreparedIdentity::create(data,exact.request_metadata_reservation());
    const auto plan=family::plan_vision_control(data);
    auto control=replay_control?*replay_control:family::build_vision_control(data,plan,0);
    if(control.items.size()!=data.vision_items.size())
        throw std::invalid_argument("prepared media control extent");
    for(auto token:data.token_ids) if(token<0 || token>=248320)
        throw std::invalid_argument("prepared media vocabulary extent");
    for(auto position:data.positions) if(position<0 || position>=exact.max_context())
        throw std::invalid_argument("prepared media position extent");
    for(const auto& payload:data.media_payloads) if(!payload)
        throw std::invalid_argument("prepared media payload retired");
    const auto check_cancel=[&] {
        if(cancelled && cancelled()) throw std::runtime_error("prepared media cancelled");
    };
    check_cancel();
    const auto ancestor_position=ancestor?ancestor->token_count():0;
    if(ancestor && (ancestor_position>=n || !ancestor->prepared_identity() ||
       ancestor->state()->rope_offset()!=ancestor->prepared_identity()->rope_delta() ||
       !ancestor->prepared_identity()->prefix_positions_equal(*identity,ancestor_position)))
        throw std::invalid_argument("prepared media replay ancestor position proof");
    auto result=Exl3VeriCacheRequest::create_planned(ancestor.get(),n-ancestor_position,true,
        exact.request_metadata_reservation(),exact.request_metadata_reservation());
    exact.reset();
    try {
        std::size_t cursor=ancestor_position;
        if(ancestor)exact.restore_exact_host_state(*ancestor->state());
        const auto text_until=[&](std::size_t end) {
            while(cursor<end) {
                check_cancel();
                for(int axis=0;axis<3;++axis)
                    if(data.token_types[cursor]!=0 || data.positions[axis*n+cursor]!=
                        static_cast<std::int64_t>(cursor)+exact.rope_offset())
                        throw std::invalid_argument("prepared text position does not match execution frontier");
                const std::int64_t token=data.token_ids[cursor];
                if(cursor==0) exact.prefill(std::span<const std::int64_t>(&token,1));
                else exact.decode(token);
                result->append(static_cast<int>(cursor),1,exact.exact_tap_rows_host(),exact.request_metadata_reservation());
                ++cursor;
            }
        };
        for(std::size_t item=0;item<data.vision_items.size();++item) {
            const auto& spans=data.vision_items[item].token_spans;
            const auto item_end=spans.back().begin+spans.back().count;
            if(item_end<=cursor)continue;
            if(spans.front().begin<cursor)
                throw std::invalid_argument("prepared media ancestor splits an item");
            // Identity and encoder inputs come from this same owning prepared
            // prompt. No arbitrary frozen embedding or detached metadata seam.
            check_cancel();
            auto output=vision.encode_prepared_cached(data.media_payloads[item],
                data.vision_items[item],control.items[item],cancelled,reserve);
            if(!output || output->embeddings.size()!=control.items[item].merged_count*5120ULL)
                throw std::logic_error("prepared encoder output extent");
            std::size_t feature_row=0;
            for(const auto& span:data.vision_items[item].token_spans) {
                text_until(span.begin);
                for(std::size_t offset=0;offset<span.count;) {
                    check_cancel();
                    const auto rows=std::min<std::size_t>(8,span.count-offset);
                    std::array<std::int32_t,24> positions{};
                    for(std::size_t row=0;row<rows;++row)for(int axis=0;axis<3;++axis)
                        positions[row*3+axis]=data.positions[axis*n+cursor+row];
                    exact.append_media_embeddings_numeric(
                        std::span<const float>(output->embeddings).subspan(feature_row*5120,rows*5120),
                        std::span<const std::int32_t>(positions).first(rows*3));
                    result->append(static_cast<int>(cursor),static_cast<int>(rows),exact.exact_tap_rows_host(),exact.request_metadata_reservation());
                    exact.finish_exact_prefill();
                    cursor+=rows;feature_row+=rows;offset+=rows;
                }
            }
            if(feature_row!=control.items[item].merged_count)
                throw std::logic_error("prepared media span consumption mismatch");
        }
        text_until(n);check_cancel();
        if(exact.position()!=static_cast<int>(n) || exact.rope_offset()!=data.rope_delta)
            throw std::logic_error("prepared media final position mismatch");
        result->state_=exact.export_exact_host_state();
        std::vector<std::int64_t> tokens(data.token_ids.begin(),data.token_ids.end());
        result->history_=Exl3TokenHistory{}.append(tokens,exact.request_metadata_reservation());
        result->prepared_identity_=std::move(identity);
        if(!result->media_tap_origin_)result->bind_media_tap_origin(exact.request_metadata_reservation());
        check_cancel();
        return result;
    } catch(...) {exact.reset();throw;}
}
} // namespace ninfer::exl3
