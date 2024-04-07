//
// Created by david on 4/7/24.
//

#ifndef RAPTOR_GENERATORS_H
#define RAPTOR_GENERATORS_H

#include "fillers.hpp"

namespace raptor {

    template <typename T>
    vector<T> iota(std::size_t n) {

        vector<T> iota_vec(n);
        iota_vec.template fill<DEVICE>(iota_filler<T>());
        return iota_vec;
    }
}

#endif //RAPTOR_GENERATORS_H
