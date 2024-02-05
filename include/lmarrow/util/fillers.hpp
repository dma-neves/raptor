//
// Created by david on 04-01-2024.
//

#ifndef LMARROW_FILLERS_HPP
#define LMARROW_FILLERS_HPP

#include "lmarrow/function/function.hpp"

template <typename T>
struct counting_sequence_filler {

    __device__ __host__
    T operator()(lmarrow::coordinates_t i) {
        return (T)i;
    }
};

template <typename T>
struct value_filler {

    T val;

    value_filler(T val) : val(val) {}

    __device__ __host__
    T operator()(lmarrow::coordinates_t i) {
        return val;
    }
};

template <typename T>
struct value_filler_2d {

    T val;

    value_filler_2d(T val) : val(val) {}

    __device__ __host__
    T operator()(lmarrow::coordinates_t i, lmarrow::coordinates_t j) {
        return val;
    }
};

#endif //LMARROW_FILLERS_HPP
