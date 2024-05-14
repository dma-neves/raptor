//
// Created by david on 02-01-2024.
//

#ifndef RAPTOR_RANDOM_HPP
#define RAPTOR_RANDOM_HPP

#include <curand_kernel.h>

#include <thrust/random.h>

#define THRUST_RANDOM

namespace raptor::random {

    static __device__ float rand(int tid) {

#ifdef THRUST_RANDOM
        thrust::default_random_engine rng(tid);
        thrust::uniform_real_distribution<float> u01(0,1);
        return u01(rng);
#else
        curandState state;
        curand_init((unsigned long long)clock() + tid, 0, 0, &state);
        return curand_uniform_double(&state);
#endif
    }
}

#endif //RAPTOR_RANDOM_HPP
