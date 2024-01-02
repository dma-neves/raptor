//
// Created by david on 02-01-2024.
//

#ifndef GDGRAPH_MATH_HPP
#define GDGRAPH_MATH_HPP

#include "lmarrow/cuda/atomic_functions.hpp"

namespace lmarrow::atomic {

    template <typename T>
    __device__ T min(T* address, T value) {

        return lmarrow::_atomicMin(address, value);
    }

    template <typename T>
    __device__ T max(T* address, T value) {

        return lmarrow::_atomicMax(address, value);
    }

    template <typename T>
    __device__ T exch(T* address, T value) {

        return atomicExch(address, value);
    }

    template <typename T>
    __device__ T add(T* address, T value) {

        return atomicAdd(address, value);
    }
}

#endif //GDGRAPH_MATH_HPP
