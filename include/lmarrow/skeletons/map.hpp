//
// Created by david on 15-10-2023.
//

#ifndef GDGRAPH_MAP_HPP
#define GDGRAPH_MAP_HPP


#include "operators.hpp"
#include "lmarrow/containers/vector.hpp"

namespace lmarrow {

    template <typename Arg>
    __device__
    static decltype(auto) forward_container_elements(int tid, Arg& arg) {
        if constexpr (std::is_pointer<Arg>::value) {
            return arg[tid];
        }
        else {
            return arg;
        }
    }

    template <typename T, typename = void>
    struct is_container : std::false_type {};

    template <typename T>
    struct is_container<T, std::void_t<decltype(std::declval<T>().get_device_ptr())>> : std::true_type {};

    template <typename Arg>
    static decltype(auto) forward_device_pointer(Arg& arg) {
        if constexpr (is_container<Arg>::value) {
            return arg.get_device_ptr();
        }
        else {
            return arg;
        }
    }

    template <typename Arg>
    void upload_containers(Arg&& arg) {
        if constexpr (is_container<std::remove_reference_t<Arg>>::value) {
            arg.upload();
        }
    }

    template <typename T, typename Functor, typename... Args>
    __global__ void _map(int n, Functor map_fun, T *output, T *first_input, Args... args) {

        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if(index < n) {

            output[index] = map_fun(first_input[index], forward_container_elements(index, args)...);
        }
    }

    template <typename Functor, typename T, template<typename> class ColType, typename... Args>
    ColType<T> map(ColType<T>& first_input, Args&&... args) {

        Functor map_fun;
        int size = first_input.size();
        ColType<T> result(size);

        collection<T>* _first_input = static_cast<collection<T>*>(&first_input);
        collection<T>* _result = static_cast<collection<T>*>(&result);

        _first_input->upload();
        (upload_containers(args), ...);
        _result->upload();

        _map<<<def_nb(size), def_tpb(size)>>>(size, map_fun, _result->get_device_ptr(), _first_input->get_device_ptr(), forward_device_pointer(args)...);

        _result->dirty_on_device();
        return result;
    }
}

#endif //GDGRAPH_MAP_HPP
