//
// Created by david on 02-01-2024.
//

#ifndef RAPTOR_ATOMIC_HPP
#define RAPTOR_ATOMIC_HPP

#include "raptor/cuda/atomic_functions.hpp"

namespace raptor::atomic {

    template <typename T>
    __device__ T min(T* address, T value) {

        return raptor::_atomicMin(address, value);
    }

    template <typename T>
    __device__ T max(T* address, T value) {

        return raptor::_atomicMax(address, value);
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

#endif //RAPTOR_MATH_HPP
