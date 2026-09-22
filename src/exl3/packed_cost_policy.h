#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::exl3 {
// Supplied estimates only; no calibration or performance claim. Units are us,
// including packing/scatter in shared_us. A bad estimate can only choose a
// slower already-admitted route, never relax ownership or arithmetic checks.
class Exl3PackedCostPolicy {
public:
    static constexpr unsigned family_count=15;
    static bool claim_before_deadline(std::chrono::steady_clock::time_point offered,
        std::chrono::steady_clock::time_point now,std::chrono::microseconds budget) noexcept {
        return budget.count()>0 && now>=offered && now-offered<budget;
    }
    struct Entry {
        unsigned family=0;
        int input=0,output=0,bits=0,first_rows=0,second_rows=0;
        std::uint64_t independent_us=0,shared_us=0,max_age_us=0;
    };
    static constexpr bool calibrated=false;
    bool has_test_bypass() const noexcept {return test_permit_underfilled_;}
    explicit Exl3PackedCostPolicy(std::span<const Entry> entries={}) {
        if(entries.size()>entries_.size())throw std::invalid_argument("packed cost table capacity");
        for(auto e:entries) {
            if(e.family>=family_count || e.input<=0 || e.output<=0 || e.bits<5 || e.bits>8 ||
               e.first_rows<1 || e.first_rows>8 || e.second_rows<1 || e.second_rows>8 ||
               !e.independent_us || !e.shared_us || e.max_age_us>50)
                throw std::invalid_argument("packed cost entry extent/age");
            for(std::size_t i=0;i<size_;++i)if(same_key(entries_[i],e))
                throw std::invalid_argument("duplicate packed cost estimate");
            entries_[size_++]=e;
        }
    }
    static Exl3PackedCostPolicy unqualified_test_permit_underfilled() noexcept {
        Exl3PackedCostPolicy policy;
        policy.test_permit_underfilled_=true;
        return policy;
    }
    bool permits(unsigned family,int input,int output,int bits,int first,int second,
                 std::uint64_t age_us) const noexcept {
        if(family>=family_count || input<=0 || output<=0 || bits<5 || bits>8 ||
           first<1 || first>8 || second<1 || second>8 ||
           age_us>(test_permit_underfilled_?5000:50))return false;
        if(test_permit_underfilled_)return true;
        if(first+second==16)return true; // existing explicit full-pair opt-in
        const Entry key{family,input,output,bits,first,second};
        for(std::size_t i=0;i<size_;++i) {
            const auto& e=entries_[i];
            if(same_key(e,key))return age_us<=e.max_age_us && e.shared_us<e.independent_us &&
                age_us<e.independent_us-e.shared_us; // no addition overflow
        }
        return false;
    }
    bool may_wait(unsigned family,int input,int output,int bits,int first) const noexcept {
        if(first==8)return permits(family,input,output,bits,8,8,0);
        for(int second=1;second<=8;++second)
            if(permits(family,input,output,bits,first,second,0))return true;
        return false;
    }
    std::uint64_t wait_budget_us(unsigned family,int input,int output,int bits,int first) const noexcept {
        std::uint64_t budget=0;
        for(int second=1;second<=8;++second) {
            if(!permits(family,input,output,bits,first,second,0))continue;
            if(test_permit_underfilled_)return 5000;
            if(first+second==16)return 50;
            const Entry key{family,input,output,bits,first,second};
            for(std::size_t i=0;i<size_;++i)if(same_key(entries_[i],key)) {
                const auto& e=entries_[i];
                // permits(0) proves a positive difference. The age must remain
                // strictly below savings, including when costs approach UINT64_MAX.
                const auto profitable_age=e.independent_us-e.shared_us-1;
                const auto limit=e.max_age_us<profitable_age?e.max_age_us:profitable_age;
                if(limit>budget)budget=limit;
            }
        }
        return budget;
    }
private:
    static bool same_key(const Entry& a,const Entry& b) noexcept {
        return a.family==b.family && a.input==b.input && a.output==b.output && a.bits==b.bits &&
            a.first_rows==b.first_rows && a.second_rows==b.second_rows;
    }
    std::array<Entry,128> entries_{};
    std::size_t size_=0;
    bool test_permit_underfilled_=false;
};
}
