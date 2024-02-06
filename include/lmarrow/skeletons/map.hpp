//
// Created by david on 15-10-2023.
//

#ifndef GDGRAPH_MAP_HPP
#define GDGRAPH_MAP_HPP


#include "operators.hpp"
#include "lmarrow/containers/collection.hpp"
#include "lmarrow/function/function.hpp"

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

    template <typename T, typename Functor, typename... Args>
    __global__ void map_kernel(int n, Functor map_fun, T *output, T *main_collection, Args... args) {

        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if(tid < n) {

            output[tid] = map_fun(main_collection[tid], forward_container_elements(tid, args)...);
        }
    }

    template <typename Functor, typename T, template<typename> class ColType, typename... Args>
    ColType<T> map(ColType<T>& main_collection, Args&&... args) {

        Functor map_fun;
        collection<T>* _main_collection = static_cast<collection<T>*>(&main_collection);
        int size = _main_collection->size();
        ColType<T> result(size);
        collection<T>* _result = static_cast<collection<T>*>(&result);

        _main_collection->upload();
        (upload_unspecified_container(args), ...);
        _result->upload();

        map_kernel<<<def_nb(size), def_tpb(size)>>>(size, map_fun, _result->get_device_ptr(), _main_collection->get_device_ptr(), forward_device_pointer(args)...);
        _result->dirty_on_device();
        return result;
    }
}

#endif //GDGRAPH_MAP_HPP
