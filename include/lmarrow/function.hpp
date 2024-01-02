//
// Created by david on 22-11-2023.
//

#ifndef GDGRAPH_FUNCTION_HPP
#define GDGRAPH_FUNCTION_HPP

#include <type_traits>

#include "lmarrow/skeletons/map.hpp"

namespace lmarrow {

    using coordinates_t = int;

    template<typename Functor, typename... Args>
    __global__
    static void kernel_with_coordinates(int nthreads, Functor fun, Args... args) {

        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid < nthreads)
            fun(tid, args...);
    }

    template<typename Functor>
    struct function_with_coordinates {

        template<typename... Args>
        void apply(std::size_t nthreads, Args&... args) {

            Functor singleton;
            (upload_containers(args), ...);
            kernel_with_coordinates<<<def_nb(nthreads), def_tpb(nthreads)>>>(nthreads, singleton, forward_device_pointer(args)...);
        }
    };
}

#endif //GDGRAPH_FUNCTION_HPP
