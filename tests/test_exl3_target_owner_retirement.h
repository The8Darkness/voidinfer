#pragma once
// Destroys the actual loaded model wrapper, not a synthetic identity token.
// Both contexts retain the uploaded backing; their private state stays separate.
void run_target_owner_retirement(std::unique_ptr<Exl3TextModel>& target,
    const std::vector<std::int64_t>& source) {
    require(target && source.size()>=24 && target->max_context()>=24,"target owner fixture extent");
    auto control=target->create_context(true),survivor=target->create_context(true);
    require(!control->oscar_enabled(),"target owner retirement requires exact FP16");
    const auto input=std::span<const std::int64_t>(source.data(),16);
    const auto suffix=std::span<const std::int64_t>(source.data()+16,8);
    control->prefill(input);survivor->prefill(input);
    control->prepare_continuation(8);survivor->prepare_continuation(8);
    auto root=survivor->export_exact_host_state();
    control->continue_rows(suffix);
    auto scores=control->continuation_logits_bits_host();
    control->finish_exact_continuation();
    auto expected=control->export_exact_host_state();
    const auto taps=control->exact_tap_rows_host();
    control.reset();target.reset();
    survivor->continue_rows(suffix);
    auto actual_scores=survivor->continuation_logits_bits_host();
    survivor->finish_exact_continuation();
    require(survivor->export_exact_host_state()->same_payload(*expected) &&
        actual_scores==scores,"retired model changed survivor state/logits");
    require(survivor->exact_tap_rows_host()==taps,"retired model changed taps");
    survivor->restore_exact_host_state(*root);survivor->continue_rows(suffix);
    survivor->finish_exact_continuation();
    require(survivor->export_exact_host_state()->same_payload(*expected),"retired model restore/replay failed");
    survivor.reset();root.reset();expected.reset();
}
