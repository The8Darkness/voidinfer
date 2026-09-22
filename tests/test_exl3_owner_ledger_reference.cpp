#include "exl3/owner_ledger_reference.h"

#include <iostream>
#include <limits>
#include <stdexcept>

using Ledger=ninfer::exl3::Exl3IndependentOwnerLedgerReference;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> bool refuses(F&& call){try{call();}catch(...){return true;}return false;}
}

int main() {
    Ledger ledger;
    ledger.allocate(10,100,Ledger::Domain::host,64);
    ledger.reference(1,10,Ledger::ReferenceKind::shared_alias);
    ledger.reference(2,10,Ledger::ReferenceKind::shared_alias);
    ledger.reference(3,10,Ledger::ReferenceKind::view);
    auto snapshot=ledger.snapshot();
    need(snapshot.bytes[static_cast<std::size_t>(Ledger::Domain::host)]==64 &&
            snapshot.allocations==1 && snapshot.aliases==2 && snapshot.views==1,
        "shared control-block aliases were charged as physical allocations");

    ledger.allocate(11,101,Ledger::Domain::host,96);
    snapshot=ledger.snapshot();
    need(snapshot.total_bytes()==160 && snapshot.allocations==2,
        "old/new replacement union omitted one physical owner");
    ledger.retire(10,Ledger::Cleanup::success);
    need(refuses([&]{ledger.reference(4,10,Ledger::ReferenceKind::view);}),
        "retiring allocation accepted a late new reader");
    ledger.release_reference(1);ledger.release_reference(2);
    need(ledger.charged(10) && ledger.snapshot().total_bytes()==160,
        "late existing view did not retain the old union charge");
    ledger.release_reference(3);
    need(!ledger.charged(10) && ledger.snapshot().total_bytes()==96,
        "final old view did not release a successful retirement charge");

    ledger.allocate(12,102,Ledger::Domain::device,128);
    ledger.reference(5,12,Ledger::ReferenceKind::shared_alias);
    ledger.retire(12,Ledger::Cleanup::callback_lost);
    ledger.release_reference(5);
    snapshot=ledger.snapshot();
    need(ledger.charged(12) && snapshot.quarantined==1 &&
            snapshot.bytes[static_cast<std::size_t>(Ledger::Domain::device)]==128,
        "lost cleanup callback released or hid quarantined device bytes");

    Ledger overflow;
    overflow.allocate(20,200,Ledger::Domain::cuda_registered_host,
        std::numeric_limits<std::size_t>::max());
    overflow.allocate(21,201,Ledger::Domain::cuda_registered_host,1);
    need(refuses([&]{(void)overflow.snapshot();}),
        "per-domain reference charge overflow was accepted");
    std::cout<<"exl3_owner_ledger_reference PASS\n";
}
