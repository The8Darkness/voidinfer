#include "exl3/packed_projection.h"
#include <array>
#include <iostream>
using namespace ninfer::exl3;
static void need(bool x){if(!x)throw std::runtime_error("packed descriptor assertion");}
int main() {
    try {
        auto model=std::make_shared<int>(1);
        std::array<Exl3ProjectionRows,2> lanes;
        for(int i=0;i<2;++i) {
            auto input=std::make_shared<std::array<std::uint16_t,128>>();
            auto output=std::make_shared<std::array<std::uint16_t,128>>();
            lanes[i]={static_cast<std::uint64_t>(i+1),4,7,model,std::make_shared<int>(i),input,output,
                "text/fp16/K6/q",10+i,3+i,8,16,8,16,input->data(),output->data(),128,128};
        }
        const auto live=[](const auto& s){return s.acquisition==4;};
        need(lanes[0].valid_admission(8));
        need(!lanes[0].valid_admission(2) && !lanes[0].valid_admission(0));
        for(int field=0;field<12;++field) {
            auto invalid=lanes[0];
            switch(field) {
            case 0:invalid.request=0;break;
            case 1:invalid.acquisition=0;break;
            case 2:invalid.execution=0;break;
            case 3:invalid.position=-1;break;
            case 4:invalid.position=std::numeric_limits<int>::max()-invalid.rows+1;break;
            case 5:invalid.rows=0;break;
            case 6:invalid.input=nullptr;break;
            case 7:invalid.output=nullptr;break;
            case 8:invalid.input_columns=0;break;
            case 9:invalid.output_columns=0;break;
            case 10:invalid.input_stride=7;break;
            case 11:invalid.output_stride=15;break;
            }
            need(!invalid.valid_admission(8));
        }
        auto last_position=lanes[0];
        last_position.position=std::numeric_limits<int>::max()-last_position.rows;
        need(last_position.valid_admission(8));
        auto overlapping=lanes[0];overlapping.output=const_cast<std::uint16_t*>(overlapping.input);
        need(!overlapping.valid_admission(8));
        overlapping.output=const_cast<std::uint16_t*>(overlapping.input)+1;
        need(!overlapping.valid_admission(8));
        auto excessive=lanes[0];
        excessive.input_stride=std::numeric_limits<std::size_t>::max()/2;
        excessive.input_storage_elements=std::numeric_limits<std::size_t>::max();
        need(!excessive.valid_admission(8));
        for(bool input_plane:{false,true}) {
            auto single=lanes;single[0].rows=1;single[1].rows=1;
            auto& stride=input_plane?single[0].input_stride:single[0].output_stride;
            const auto original_stride=stride;
            stride=std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t)+1;
            need(!single[0].valid_admission(8) && !Exl3ProjectionRows::valid_pair(single[0],single[1],16));
            bool refused=false;
            try{(void)Exl3PackedProjectionPlan::assemble(single,16,live);}
            catch(const std::invalid_argument&){refused=true;}
            need(refused); // Admission must reject before range arithmetic.
            stride=original_stride;
            need(Exl3ProjectionRows::valid_pair(single[0],single[1],16));
            need(Exl3PackedProjectionPlan::assemble(single,16,live).rows()==2);
        }
        for(bool input_plane:{false,true}) {
            auto boundary=lanes[0];
            auto& storage=input_plane?boundary.input_storage_elements:boundary.output_storage_elements;
            const auto stride=input_plane?boundary.input_stride:boundary.output_stride;
            const auto columns=input_plane?boundary.input_columns:boundary.output_columns;
            storage=(boundary.rows-1)*stride+columns;
            need(boundary.valid_admission(8));
            --storage;need(!boundary.valid_admission(8));
            storage=0;need(!boundary.valid_admission(8));
        }
        {
            std::string identity(Exl3ProjectionContract::capacity,'a');
            Exl3ProjectionContract owned(identity);
            identity.back()='b';
            need(owned!=Exl3ProjectionContract(identity));
            identity.back()='a';
            need(owned==Exl3ProjectionContract(identity));
            identity.push_back('a');
            bool overflow_refused=false;
            try{owned=identity;}catch(const std::invalid_argument&){overflow_refused=true;}
            need(overflow_refused && owned==Exl3ProjectionContract(std::string(Exl3ProjectionContract::capacity,'a')));
            need(Exl3ProjectionContract(nullptr).empty());
            const std::string embedded("a\0b",3),other("a\0c",3);
            need(Exl3ProjectionContract(embedded)!=Exl3ProjectionContract(other));
        }
        auto plan=Exl3PackedProjectionPlan::assemble(lanes,8,live);
        {
            std::array<std::weak_ptr<const void>,8> retained;
            {
                auto offered=lanes;
                auto private_model=std::make_shared<int>(19);
                for(unsigned i=0;i<2;++i) {
                    auto input=std::make_shared<std::array<std::uint16_t,128>>();
                    auto output=std::make_shared<std::array<std::uint16_t,128>>();
                    offered[i].model=private_model;
                    offered[i].root=std::make_shared<int>(20+i);
                    offered[i].input=input->data();offered[i].input_owner=input;
                    offered[i].output=output->data();offered[i].output_owner=output;
                    retained[4*i]=offered[i].model;retained[4*i+1]=offered[i].root;
                    retained[4*i+2]=input;retained[4*i+3]=output;
                }
                auto original=Exl3PackedProjectionPlan::assemble(offered,8,live);
                offered={};private_model.reset();
                auto moved=std::move(original);
                const auto require_empty=[&](const Exl3PackedProjectionPlan& source) {
                    need(source.rows()==0 && source.lanes().empty());
                    bool refused=false;
                    try{source.validate_live(live);}catch(const std::invalid_argument&){refused=true;}
                    need(refused);
                };
                require_empty(original);
                moved.validate_live(live);
                for(const auto& owner:retained)need(!owner.expired());
                auto assigned=Exl3PackedProjectionPlan::assemble(lanes,8,live);
                assigned=std::move(moved);
                require_empty(moved);
                assigned.validate_live(live);
                need(assigned.rows()==7 && assigned.lanes()[1].packed_first==3);
                for(const auto& owner:retained)need(!owner.expired());
                auto last_consumer=assigned;
                assigned=Exl3PackedProjectionPlan::assemble(lanes,8,live);
                // One consumer has retired/rebound its plan. The other still
                // owns both private destinations and their conditioning roots.
                last_consumer.validate_live(live);
                need(last_consumer.rows()==7 && last_consumer.lanes()[1].packed_first==3);
                for(const auto& owner:retained)need(!owner.expired());
            }
            for(const auto& owner:retained)need(owner.expired());
        }
        for(unsigned invalidated=0;invalidated<2;++invalidated) {
            std::array<std::weak_ptr<const void>,4> retained;
            {
                auto offered=lanes;
                for(unsigned i=0;i<2;++i) {
                    offered[i].root=std::make_shared<int>(100+i);
                    auto output=std::make_shared<std::array<std::uint16_t,128>>();
                    output->fill(0x5a5a);
                    offered[i].output=output->data();offered[i].output_owner=output;
                    retained[2*i]=offered[i].root;retained[2*i+1]=output;
                }
                std::array<std::uint64_t,2> acquisition{4,4};
                const auto current=[&](const Exl3ProjectionRows& source) {
                    return source.request>=1 && source.request<=2 &&
                        source.acquisition==acquisition[source.request-1] && source.execution==7;
                };
                auto delayed=Exl3PackedProjectionPlan::assemble(offered,8,current);
                offered={}; // only the pending plan keeps these roots/destinations
                ++acquisition[invalidated]; // a replacement lease occupies this slot
                for(unsigned completion=0;completion<2;++completion) {
                    bool stale=false;try {delayed.validate_live(current);}
                    catch(const std::invalid_argument&) {stale=true;}
                    need(stale);
                    for(const auto& owner:retained)need(!owner.expired());
                    for(const auto& lane:delayed.lanes())
                        for(unsigned i=0;i<128;++i)need(lane.source.output[i]==0x5a5a);
                }
            }
            for(const auto& owner:retained)need(owner.expired());
        }
        for(int field=0;field<5;++field) {
            auto invalid=lanes;
            auto& backing=field==0?invalid[1].model:field==1?invalid[1].root:
                field==2?invalid[1].input_owner:field==3?invalid[1].output_owner:invalid[1].model;
            backing=field==4?std::shared_ptr<const void>(std::make_shared<int>(42),backing.get()):
                std::shared_ptr<const void>(std::shared_ptr<const void>{},backing.get());
            if(field<4)need(!invalid[1].valid_admission(8));
            bool rejected=false;
            try{Exl3PackedProjectionPlan::assemble(invalid,8,live);}
            catch(const std::invalid_argument&){rejected=true;}need(rejected);
        }
        auto alias_model=lanes;
        alias_model[1].model=std::shared_ptr<const void>(model,model.get());
        need(Exl3PackedProjectionPlan::assemble(alias_model,8,live).rows()==7);
        need(plan.rows()==7 && plan.lanes()[1].packed_first==3);
        struct BorrowedLiveness {
            BorrowedLiveness()=default;
            BorrowedLiveness(const BorrowedLiveness&)=delete;
            bool operator()(const Exl3ProjectionRows& row) const {return row.acquisition==4;}
        } borrowed;
        auto borrowed_plan=Exl3PackedProjectionPlan::assemble(lanes,8,borrowed);
        borrowed_plan.validate_live(borrowed);
        bool missing_live_refused=false;
        try{plan.validate_live(Exl3PackedProjectionPlan::Live{});}
        catch(const std::invalid_argument&){missing_live_refused=true;}need(missing_live_refused);
        bool (*missing_function)(const Exl3ProjectionRows&)=nullptr;
        missing_live_refused=false;
        try{Exl3PackedProjectionPlan::assemble(lanes,8,missing_function);}
        catch(const std::invalid_argument&){missing_live_refused=true;}need(missing_live_refused);
        {
            std::array<Exl3ProjectionRows,9> bounded;
            for(std::size_t i=0;i<bounded.size();++i) {
                auto input=std::make_shared<std::array<std::uint16_t,128>>();
                auto output=std::make_shared<std::array<std::uint16_t,128>>();
                bounded[i]={i+1,4,7,model,std::make_shared<int>(static_cast<int>(i)),input,output,
                    "text/fp16/K6/q",10,2,8,16,8,16,input->data(),output->data(),128,128};
            }
            std::weak_ptr<const void> last_destination=bounded[7].output_owner;
            {
                auto full=Exl3PackedProjectionPlan::assemble(std::span(bounded).first(8),16,live);
                need(full.rows()==16 && full.lanes().size()==8);
                for(std::size_t i=0;i<8;++i)need(full.lanes()[i].packed_first==static_cast<int>(i)*2);
                bool excessive=false;
                try{Exl3PackedProjectionPlan::assemble(bounded,16,live);}
                catch(const std::invalid_argument&){excessive=true;}need(excessive);
                bounded={};
                need(!last_destination.expired());
                full.validate_live(live);
            }
            need(last_destination.expired());
        }
        auto swapped=lanes;std::swap(swapped[0],swapped[1]);
        need(Exl3PackedProjectionPlan::assemble(swapped,8,live).lanes()[1].packed_first==4);
        for(int fault=0;fault<16;++fault) {
            auto bad=lanes;
            if(fault==0) bad[1].rows=0;
            if(fault==1) bad[1].rows=6;
            if(fault==2) bad[1].acquisition=3;
            if(fault==3) bad[1].model=std::make_shared<int>(1);
            if(fault==4) bad[1].contract="vision/fp16/K6/q";
            if(fault==5) bad[1].output=bad[0].output;
            if(fault==6) bad[1].request=bad[0].request;
            if(fault==7) bad[1].position=std::numeric_limits<int>::max();
            if(fault==8) bad[1].input_stride=std::numeric_limits<std::size_t>::max();
            if(fault==9) bad[1].input_storage_elements=1;
            if(fault==10) bad[1].output_storage_elements=1;
            if(fault==11) bad[1].output=const_cast<std::uint16_t*>(bad[0].input)+1;
            if(fault==12) bad[1].input=bad[0].output+1;
            if(fault==13) bad[1].output_stride=15;
            if(fault==14) bad[1].output=bad[0].output+16; // partial cross-row overlap
            if(fault==15) {
                bad[0].contract="Engine/text/fp16/shared-target/M16";
                bad[1].contract="Engine/prepared-media/fp16/shared-target/M16";
            }
            bool refused=false;try {Exl3PackedProjectionPlan::assemble(bad,8,live);}catch(const std::exception&){refused=true;}
            need(refused);
        }
        int visits=0;bool refused=false;
        try {Exl3PackedProjectionPlan::assemble(lanes,8,[&](const auto&){return ++visits<3;});}
        catch(const std::invalid_argument&){refused=true;}need(refused);
        std::array<std::uint16_t,128> packed_input{},packed_output{};
        plan.require_disjoint_scratch(packed_input.data(),8,packed_output.data(),16);
        refused=false;
        try{plan.require_disjoint_scratch(const_cast<std::uint16_t*>(lanes[0].input),8,packed_output.data(),16);}
        catch(const std::invalid_argument&){refused=true;}need(refused);
        refused=false;
        try{plan.require_disjoint_scratch(packed_input.data(),8,lanes[1].output,16);}
        catch(const std::invalid_argument&){refused=true;}need(refused);
        const auto owner=plan.lanes()[0].source.output_owner;
        lanes={};need(owner.use_count()>=2); // immutable plan retains private destinations
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
