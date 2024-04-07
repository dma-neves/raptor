//
// Created by david on 15-10-2023.
//

#ifndef RAPTOR_OPERATORS_HPP
#define RAPTOR_OPERATORS_HPP

namespace raptor {

    template<typename T>
    struct sum {

        __device__ __host__
        T operator()(const T& a, const T& b) const {

            return a + b;
        }
    };

    template<typename T>
    struct sub {

        __device__ __host__
        T operator()(const T& a, const T& b) const {

            return a - b;
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

    template<typename T>
    struct min {

        __device__ __host__
        T operator()(const T& a, const T& b) const {

            return a < b ? a : b;
        }
    };
}

#endif //RAPTOR_OPERATORS_HPP
