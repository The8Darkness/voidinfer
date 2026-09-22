#include "exl3/suffix_proposer.h"
#include <iostream>
#include <vector>
using namespace ninfer::exl3;
static void need(bool x){if(!x)throw std::runtime_error("suffix proposal assertion");}
int main() {
    try {
        Exl3SuffixProposer index;
        need(index.propose(4,8).rows==0);
        const std::array<std::int64_t,12> history{1,2,3,4,5,6,7,8,9,1,2,3};
        index.append(history);
        auto proposal=index.propose(4,8);
        need(index.outcome_totals().rounds==0); // A lookup hit is not publication.
        index.set_selection_limits({1,2,1,1});need(index.eligible());
        bool rebound=false;try{index.set_selection_limits({1,2,0,8});}
        catch(const std::invalid_argument&){rebound=true;}need(rebound);
        index.record_published_outcome({7,0,8});
        need(!index.eligible());index.record_neural_publication();need(!index.eligible());
        index.record_neural_publication();need(index.eligible());
        need(index.outcome_totals().accepted==0 && index.outcome_totals().replay==8 &&
            index.outcome_totals().skipped_neural_blocks==1);
        for(int i=0;i<16;++i)index.record_published_outcome({7,7,0});
        need(index.outcome_totals().rounds==16 && index.outcome_totals().accepted==112 &&
            index.outcome_totals().replay==0);
        need(index.eligible());
        need(proposal.rows==8 && proposal.tokens[0]==4 && proposal.tokens[1]==5 && proposal.tokens[7]==2);
        need(index.propose(4,3).rows==3 && index.propose(4,1).rows==0);
        need(index.propose(260,8).rows==0); // same bucket as 4, full token differs
        const std::array<std::int64_t,2> invalid{7,-1};
        bool refused=false;try{index.append(invalid);}catch(const std::invalid_argument&){refused=true;}
        need(refused && index.propose(4,8).tokens==proposal.tokens);
        auto branch=index;
        for(int i=0;i<16;++i)branch.record_published_outcome({7,0,8});
        need(!branch.eligible() && index.eligible());
        const std::array<std::int64_t,4> changed{99,1,2,8};branch.append(changed);
        need(branch.propose(4,8).rows==0 && index.propose(4,8).rows==8);
        index.reset();need(index.propose(4,8).rows==0);
        need(index.outcome_totals().rounds==0);
        need(!index.has_selection_limits() && index.eligible());
        // Old indexed nodes age out. Identical last-token buckets and many
        // overwritten links must terminate with either a valid match or miss.
        std::vector<std::int64_t> old(8192,17);index.append(old);
        const auto repeated=index.propose(17,8);
        need(repeated.rows>=2 && repeated.rows<=8);
        for(std::size_t i=0;i<repeated.rows;++i)need(repeated.tokens[i]==17);
        index.reset();index.append(history);
        need(index.propose(4,8).tokens==proposal.tokens);
        Exl3SuffixProposer key3(3);
        const std::array<std::int64_t,8> key3_history{1,2,9,4,5,6,1,2};
        key3.append(key3_history);
        const auto key3_proposal=key3.propose(9,8);
        need(key3.key_tokens()==3 && key3_proposal.rows==6 &&
            key3_proposal.tokens[0]==9 && key3_proposal.tokens[1]==4 &&
            key3_proposal.tokens[5]==2);
        Exl3SuffixProposer key2(2);
        const std::array<std::int64_t,5> key2_history{7,9,4,5,7};
        key2.append(key2_history);
        const auto key2_proposal=key2.propose(9,8);
        need(key2.key_tokens()==2 && key2_proposal.rows==4 &&
            key2_proposal.tokens[0]==9 && key2_proposal.tokens[1]==4 &&
            key2_proposal.tokens[3]==7);
        bool key_refused=false;
        try{(void)Exl3SuffixProposer(1);}catch(const std::invalid_argument&){key_refused=true;}
        need(key_refused);
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
