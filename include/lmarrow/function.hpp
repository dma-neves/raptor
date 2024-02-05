//
// Created by david on 22-11-2023.
//

#ifndef GDGRAPH_FUNCTION_HPP
#define GDGRAPH_FUNCTION_HPP

#include <type_traits>

#include "lmarrow/skeletons/map.hpp"
#include "lmarrow/detail.hpp"

namespace lmarrow {

    using coordinates_t = int;

    template<typename Functor, typename... Args>
    __global__
    static void function_kernel(int fun_size, Functor fun, Args... args) {

        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid < fun_size)
            fun(tid, args...);
    }

    template<typename Functor>
    struct function {

        std::size_t fun_size = 0;

        template <typename Arg>
        void set_default_size(Arg& arg) {
            if constexpr (detail::is_collection<Arg>::value) {
                if(fun_size == 0 || arg.size() < fun_size) {
                    fun_size = arg.size();
                }
            }
        }

        template<typename... Args>
        void apply(Args&... args) {

            if(fun_size == 0) {
                (set_default_size(args), ...);
            }

            Functor singleton;
            (upload_containers(args), ...);
            function_kernel<<<def_nb(fun_size), def_tpb(fun_size)>>>(fun_size, singleton, forward_device_pointer(args)...);
        }

        void set_size(std::size_t size) {

            fun_size = size;
        }
    };
}

#endif //GDGRAPH_FUNCTION_HPP
