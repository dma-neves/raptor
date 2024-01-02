//
// Created by david on 15-10-2023.
//

#ifndef GDGRAPH_OPERATORS_HPP
#define GDGRAPH_OPERATORS_HPP

namespace lmarrow {

    template<typename T>
    struct sum {

        __device__ __host__
        T operator()(const T& a, const T& b) const {

            return a + b;
        }
    };

    template<typename T>
    struct mult {

        __device__ __host__
        T operator()(const T& a, const T& b) const {

            return a * b;
        }
    };

    template<typename T>
    struct max {

        __device__ __host__
        T operator()(const T& a, const T& b) const {

            return a > b ? a : b;
        }
    };
}

#endif //GDGRAPH_OPERATORS_HPP
