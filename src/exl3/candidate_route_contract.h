#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::exl3 {

enum class Exl3NumericalRouteClass : unsigned char { exact, numeric };

struct Exl3NumericalRouteIdentity {
    std::string_view option;
    Exl3NumericalRouteClass route_class=Exl3NumericalRouteClass::exact;
    std::string_view representation;
    std::string_view precision;
    bool qualified_for_engine_exact=false;
    bool composed_with_exact=false;
};

// This is an admission contract, not a numerical assertion. An option may use
// the same tensors or be default-off and still remain numeric-only or
// unqualified. Exact Engine construction must retain an explicit representation
// and precision identity and may not relabel a numeric component by composing it
// with an exact fallback.
inline void require_engine_exact_route(const Exl3NumericalRouteIdentity& route,
    std::string_view representation,std::string_view precision) {
    if(route.option.empty() || route.representation.empty() || route.precision.empty())
        throw std::invalid_argument("EXL3 numerical route identity incomplete");
    if(route.route_class!=Exl3NumericalRouteClass::exact || route.composed_with_exact)
        throw std::invalid_argument(std::string("EXL3 numeric-only route cannot enter exact Engine: ")+std::string(route.option));
    if(!route.qualified_for_engine_exact)
        throw std::invalid_argument(std::string("EXL3 exact route lacks Engine authority: ")+std::string(route.option));
    if(route.representation!=representation || route.precision!=precision)
        throw std::invalid_argument(std::string("EXL3 exact route representation changed: ")+std::string(route.option));
}

}
