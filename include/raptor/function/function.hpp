//
// Created by david on 22-11-2023.
//

#ifndef RAPTOR_FUNCTION_HPP
#define RAPTOR_FUNCTION_HPP

#include <type_traits>

#include "detail.hpp"

namespace raptor {

    using coordinates_t = int;

    template <typename Arg>
    static decltype(auto) forward_device_pointer(Arg& arg) {
        
        if constexpr (detail::is_container<Arg>::value) {
            return arg.get_device_data();
        }
        else {
            return arg;
        }
    }

    template <typename FunctionArg, typename Arg>
    void upload_container(Arg& arg) {

        if constexpr (detail::is_container<std::remove_reference_t<Arg>>::value && detail::is_input<FunctionArg>) {
            arg.upload();
        }
        else if constexpr (detail::is_container<std::remove_reference_t<Arg>>::value) {
            arg.upload(0,true);
        }
    }

    template <typename Arg>
    void upload_unspecified_container(Arg& arg) {

        if constexpr (detail::is_container<std::remove_reference_t<Arg>>::value) {
            arg.upload();
        }
    }

    template <typename FunctionArg, typename Arg>
    void dirty_container(Arg& arg) {
        if constexpr (detail::is_container<std::remove_reference_t<Arg>>::value && detail::is_output<FunctionArg>) {
            arg.dirty_device();
        }
    }

    template<typename Functor, typename... Args>
    __global__
    static void function_kernel(int fun_size, Functor fun, Args... args) {

        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid < fun_size) {
            fun(tid, args...);
        }
    }

    template<typename Functor, typename... FunctionArgs>
    struct function {

        std::size_t fun_size = 0;

        template <typename Arg>
        void set_default_size(Arg&& arg) {
            if constexpr (detail::is_collection<std::remove_reference_t<Arg>>::value) {
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

            Functor fun;

            if constexpr (sizeof...(FunctionArgs) == 0) {
                (upload_unspecified_container<Args>(args), ...);
                function_kernel<<<def_nb(fun_size), def_tpb(fun_size)>>>(fun_size, fun, forward_device_pointer(args)...);
            }
            else {
                (upload_container<FunctionArgs, Args>(args), ...);
                function_kernel<<<def_nb(fun_size), def_tpb(fun_size)>>>(fun_size, fun, forward_device_pointer(args)...);
                (dirty_container<FunctionArgs, Args>(args), ...);
            }
        }

        void set_size(std::size_t size) {

            fun_size = size;
        }
    };
}

#endif //RAPTOR_FUNCTION_HPP
