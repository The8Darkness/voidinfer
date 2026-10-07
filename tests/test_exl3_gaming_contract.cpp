#include "exl3/gaming_optimizations.h"
#include <iostream>
using namespace ninfer::exl3;
static void require(bool ok) { if(!ok)throw std::runtime_error("gaming option contract"); }
int main() {
    try {
        for(const auto* flag:gopt_flags)_putenv_s(flag,"");
        // Unset flags are off except GOPT-002, the measured default.
        {
            const auto defaults=Exl3GamingOptions::from_environment().enabled;
            for(unsigned i=0;i<defaults.size();++i)
                require(defaults[i]==(i==static_cast<unsigned>(Gopt::GdnConvTrace)));
        }
        for(const auto* flag:gopt_flags)_putenv_s(flag,"0");
        for(const auto* flag:gopt_flags) {
            _putenv_s(flag,"1");
            auto options=Exl3GamingOptions::from_environment();
            unsigned count=0;for(bool value:options.enabled)count+=value;
            require(count==1);
            _putenv_s(flag,"true");bool refused=false;
            try{(void)Exl3GamingOptions::from_environment();}catch(const std::invalid_argument&){refused=true;}
            require(refused);_putenv_s(flag,"0");
        }
        _putenv_s("NINFER_GOPT_007","1");_putenv_s("NINFER_GOPT_008","1");
        bool refused=false;
        try{(void)Exl3GamingOptions::from_environment();}catch(const std::invalid_argument&){refused=true;}
        require(refused);
        for(int rows:{-1,0,1,2,3,7,8,9,16,1024}) {
            require(gopt_small_gdn(rows,true,false,true)==(rows>=2&&rows<=8));
            require(!gopt_small_gdn(rows,false,false,true));
            require(!gopt_small_gdn(rows,true,true,true));
            require(!gopt_small_gdn(rows,true,false,false));
        }
        std::cout<<"PASS default-off, strict options, conflicting consumers, small-M boundaries\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
