//
// Created by david on 15-10-2023.
//

#ifndef GDGRAPH_REDUCE_HPP
#define GDGRAPH_REDUCE_HPP

#include <cub/cub.cuh>


#include "operators.hpp"
#include "lmarrow/containers/vector.hpp"
#include "lmarrow/containers/scalar.hpp"

namespace lmarrow {


    template <typename Operator, typename T, template<typename> class ColType>
    scalar<T> reduce(ColType<T>& col) {

        scalar<T> res;
        Operator op;
        auto* _col = static_cast<collection<T>*>(&col);

        _col->upload();
        res.upload();

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, _col->get_device_ptr(), res.get_device_ptr() , _col->size(), op, 0);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, _col->get_device_ptr(), res.get_device_ptr() , _col->size(), op, 0);
        cudaFree(d_temp_storage);

        res.dirty_on_device();

        return res;
    }
}

#endif //GDGRAPH_REDUCE_HPP
