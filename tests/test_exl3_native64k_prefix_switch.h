#pragma once

// Prepared-only T266 coverage.  This deliberately exercises one physical
// ordinary-FP16 context across a short root, a near-64K root and two private
// branches.  The Engine lifecycle fixture owns deterministic cancellation at
// each synchronized attachment boundary; this fixture establishes that the
// state being attached is complete and that sibling suffixes remain private.
void run_native64k_prefix_switch_qualification(
    Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    require(target.max_context()==65536 && !source.empty(),
        "native64K prefix switch fixture/model extent");
    for(const auto token:source)
        require(token>=0 && token<kVocab,"native64K prefix switch token extent");

    auto context=target.create_context(true);
    context->prepare_continuation(8);
    std::vector<std::int64_t> long_prompt(65520);
    for(std::size_t row=0;row<long_prompt.size();++row)
        long_prompt[row]=source[row%source.size()];

    const auto short_root=Request::initialize(*context,
        std::span<const std::int64_t>(long_prompt.data(),64),1024);
    require(short_root->state()->position()==64 &&
        short_root->state()->native_extent_valid(target.max_context()),
        "native64K short root extent");

    // Reuse the same physical allocation for a much longer request.  The
    // immutable short image must remain intact while the context now carries
    // the complete near-capacity history.
    const auto long_root=Request::initialize(*context,long_prompt,1024);
    require(long_root->state()->position()==static_cast<int>(long_prompt.size()) &&
        long_root->state()->native_extent_valid(target.max_context()) &&
        short_root->state()->position()==64 &&
        short_root->matches_tokens(std::span<const std::int64_t>(long_prompt.data(),64)),
        "native64K short-to-long switch truncated immutable state");

    const std::int64_t suffix_a=(long_prompt.back()+1)%kVocab;
    const std::int64_t suffix_b=(long_prompt.back()+2)%kVocab;
    const std::array<std::int64_t,1> a{suffix_a},b{suffix_b};
    const auto branch_a=long_root->append_prompt(*context,a,8);
    const auto branch_b=long_root->append_prompt(*context,b,8);
    require(branch_a->state()->position()==65521 && branch_b->state()->position()==65521 &&
        branch_a->state()->native_extent_valid(target.max_context()) &&
        branch_b->state()->native_extent_valid(target.max_context()) &&
        long_root->state()->is_prefix_of(*branch_a->state()) &&
        long_root->state()->is_prefix_of(*branch_b->state()) &&
        branch_a->token_suffix(65520)==std::vector<std::int64_t>{suffix_a} &&
        branch_b->token_suffix(65520)==std::vector<std::int64_t>{suffix_b} &&
        !branch_a->state()->same_represented_payload_for_test(*branch_b->state()),
        "native64K sibling branch state/history aliased");

    // Alternate roots on the same context.  Every restore must reproduce its
    // own full recurrent/KV/position image; no prior suffix is inherited.
    context->restore_exact_host_state(*short_root->state());
    require(short_root->state()->same_payload(*context->export_exact_host_state(nullptr,false)),
        "native64K long-to-short restore changed short root");
    context->restore_exact_host_state(*branch_a->state());
    require(branch_a->state()->same_payload(*context->export_exact_host_state(nullptr,false)),
        "native64K branch A restore changed private suffix state");
    context->restore_exact_host_state(*branch_b->state());
    require(branch_b->state()->same_payload(*context->export_exact_host_state(nullptr,false)),
        "native64K branch B restore inherited sibling suffix state");

    std::cout<<"NATIVE64K_PREFIX_SWITCH PASS short=64 long="<<long_prompt.size()
        <<" branch_position="<<branch_b->state()->position()
        <<" attachment_cancel_phases=prepared_in_engine_lifecycle"<<std::endl;
}
