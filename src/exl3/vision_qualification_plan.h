#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace ninfer::exl3 {

enum class Exl3V6QualificationCategory : std::uint8_t {
    numerical,
    heldout_semantic,
    state_and_ring,
    lifecycle,
    client,
    count
};
enum class Exl3V6QualificationState : std::uint8_t { pending,failed,passed };

struct Exl3V6QualificationReceipt {
    Exl3V6QualificationState state=Exl3V6QualificationState::pending;
    std::string_view identity;
    std::string_view prerequisite;
    std::string_view intended_command;
};

class Exl3V6QualificationPlan {
    std::array<Exl3V6QualificationReceipt,
        static_cast<std::size_t>(Exl3V6QualificationCategory::count)> receipts_{};
public:
    Exl3V6QualificationPlan() {
        receipt(Exl3V6QualificationCategory::numerical)={
            Exl3V6QualificationState::failed,{},
            "T282 independent manifest plus every T281 boundary, finiteness, RRMS and cosine gate",
            "test_exl3_vision <target> <image> <fresh-output-directory>"};
        receipt(Exl3V6QualificationCategory::heldout_semantic)={
            Exl3V6QualificationState::pending,{},
            "resolved T283 image/video references and independent semantic review",
            "ninfer_media_client_loopback <host> <port> <model> <image-a> <image-b> <video> full"};
        receipt(Exl3V6QualificationCategory::state_and_ring)={
            Exl3V6QualificationState::pending,{},
            "same-input frozen target state, taps, request-private draft ring and first divergence",
            "test_exl3_prepared_request <target> <draft> <image>"};
        receipt(Exl3V6QualificationCategory::lifecycle)={
            Exl3V6QualificationState::pending,{},
            "encoder/target/draft cancellation, stale completion, pressure and C1/C2 retirement",
            "test_exl3_prepared_request <target> <draft> <image>"};
        receipt(Exl3V6QualificationCategory::client)={
            Exl3V6QualificationState::pending,{},
            "qualified loopback request, SSE, cache, disconnect and post-reload phases",
            "ninfer_media_client_loopback <host> <port> <model> <image-a> <image-b> <video> full"};
    }

    Exl3V6QualificationReceipt& receipt(Exl3V6QualificationCategory category) noexcept {
        return receipts_[static_cast<std::size_t>(category)];
    }
    [[nodiscard]] const Exl3V6QualificationReceipt& receipt(
        Exl3V6QualificationCategory category) const noexcept {
        return receipts_[static_cast<std::size_t>(category)];
    }
    [[nodiscard]] bool qualified() const {
        for(std::size_t i=0;i<receipts_.size();++i) {
            const auto& value=receipts_[i];
            if(value.prerequisite.empty() || value.intended_command.empty())
                throw std::invalid_argument("V6 qualification category definition incomplete");
            if(value.state!=Exl3V6QualificationState::passed || value.identity.empty())return false;
            for(std::size_t prior=0;prior<i;++prior)
                if(receipts_[prior].identity==value.identity)
                    throw std::invalid_argument("V6 qualification categories share one receipt");
        }
        return true;
    }
};

} // namespace ninfer::exl3
