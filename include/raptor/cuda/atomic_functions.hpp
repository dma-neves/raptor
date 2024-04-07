//
// Created by david on 02-01-2024.
//

#ifndef RAPTOR_ATOMIC_FUNCTIONS_HPP
#define RAPTOR_ATOMIC_FUNCTIONS_HPP

/*
 * Wrapper around CUDA's natively supported atomic functions, taken from:
 * https://github.com/gunrock/gunrock/blob/main/include/gunrock/cuda/atomic_functions.hxx
 * */

namespace raptor {

    template <typename type_t>
    __device__ static type_t _atomicMin(type_t* address, type_t value) {
        return atomicMin(address, value);
    }

    __device__ static float _atomicMin(float* address, float value) {
        int* addr_as_int = reinterpret_cast<int*>(address);
        int old = *addr_as_int;
        int expected;
        do {
            expected = old;
            old = atomicCAS(addr_as_int, expected,
                              __float_as_int(::fminf(value, __int_as_float(expected))));
        } while (expected != old);
        return __int_as_float(old);
    }

    __device__ static double _atomicMin(double* address, double value) {
        unsigned long long* addr_as_longlong =
                reinterpret_cast<unsigned long long*>(address);
        unsigned long long old = *addr_as_longlong;
        unsigned long long expected;
        do {
            expected = old;
            old = atomicCAS(
                    addr_as_longlong, expected,
                    __double_as_longlong(::fmin(value, __longlong_as_double(expected))));
        } while (expected != old);
        return __longlong_as_double(old);
    }

    template <typename type_t>
    __device__ static type_t _atomicMax(type_t* address, type_t value) {
        return atomicMax(address, value);
    }

    __device__ static float atomicMax(float* address, float val) {
        int* addr_as_int = reinterpret_cast<int*>(address);
        int old = *addr_as_int;
        int expected;
        do {
            expected = old;
            old = atomicCAS(addr_as_int, expected,
                              __float_as_int(::fmaxf(val, __int_as_float(expected))));
        } while (expected != old);
        return __int_as_float(old);
    }

    __device__ static double _atomicMax(double* address, double value) {
        unsigned long long* addr_as_longlong =
                reinterpret_cast<unsigned long long*>(address);
        unsigned long long old = *addr_as_longlong;
        unsigned long long expected;
        do {
            expected = old;
            old = atomicCAS(
                    addr_as_longlong, expected,
                    __double_as_longlong(::fmax(value, __longlong_as_double(expected))));
        } while (expected != old);
        return __longlong_as_double(old);
    }
}

#endif //RAPTOR_ATOMIC_FUNCTIONS_HPP
