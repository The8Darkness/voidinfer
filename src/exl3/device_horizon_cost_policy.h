#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <stdexcept>

namespace ninfer::exl3 {

// Request-local, greedy-only B4/B8 experiment. A sample is complete host wall
// from the post-seed proposal boundary through the next target seed readiness;
// it includes the preceding proposal, verification, repair, tap/ring staging
// and queued device work. It is an online cost proxy, not a GPU stage timer.
class Exl3DeviceHorizonCostPolicy {
public:
    struct Snapshot {
        std::uint64_t trials=0;
        unsigned recent_samples=0;
        double useful_per_ms=0;
        double mean_complete_ms=0;
        double mean_useful=0;
    };

    unsigned select(unsigned available) const noexcept {
        if(available<4)return available;
        if(arms_[1].trials<4)return 8<=available?8:4;
        if(arms_[0].trials<4)return 4;
        const unsigned best=score(arms_[0])>score(arms_[1])?4u:8u;
        // Keep an arm observable if acceptance changes later in the request.
        const unsigned selected=observed_ && observed_%12==0?
            (best==4?8u:4u):best;
        return selected<=available?selected:4;
    }

    void observe(unsigned width,unsigned useful,double complete_ms) {
        if((width!=4 && width!=8) || useful<1 || useful>width ||
           !std::isfinite(complete_ms) || complete_ms<=0)
            throw std::invalid_argument("device horizon complete-round observation");
        auto& arm=arms_[width==4?0:1];
        if(arm.recent.size()==8) {
            arm.useful_sum-=arm.recent.front().useful;
            arm.ms_sum-=arm.recent.front().ms;
            arm.recent.pop_front();
        }
        arm.recent.push_back({useful,complete_ms});
        arm.useful_sum+=useful;
        arm.ms_sum+=complete_ms;
        ++arm.trials;++observed_;
    }

    Snapshot snapshot(unsigned width) const {
        if(width!=4 && width!=8)
            throw std::invalid_argument("device horizon snapshot width");
        const auto& arm=arms_[width==4?0:1];
        const double count=static_cast<double>(arm.recent.size());
        return {arm.trials,static_cast<unsigned>(arm.recent.size()),score(arm),
                count?arm.ms_sum/count:0.0,
                count?static_cast<double>(arm.useful_sum)/count:0.0};
    }

    std::uint64_t observed() const noexcept {return observed_;}

private:
    struct Sample {unsigned useful;double ms;};
    struct Arm {
        std::deque<Sample> recent;
        std::uint64_t trials=0;
        unsigned useful_sum=0;
        double ms_sum=0;
    };
    static double score(const Arm& arm) noexcept {
        return arm.ms_sum>0?static_cast<double>(arm.useful_sum)/arm.ms_sum:0.0;
    }
    std::array<Arm,2> arms_{};
    std::uint64_t observed_=0;
};

} // namespace ninfer::exl3
