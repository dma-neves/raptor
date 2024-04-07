//
// Created by david on 02-01-2024.
//

#ifndef RAPTOR_RANDOM_HPP
#define RAPTOR_RANDOM_HPP

#include <curand_kernel.h>

namespace raptor::random {

    static __device__ float rand(int tid) {

        curandState state;
        curand_init((unsigned long long)clock() + tid, 0, 0, &state);
        return curand_uniform_double(&state);
    }
}

#endif //RAPTOR_RANDOM_HPP
