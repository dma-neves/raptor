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
    T reduce(ColType<T>& col) {

        Operator op;

        auto* _col = static_cast<collection<T>*>(&col);

        _col->upload();

        scalar<T> res;
        res.upload();

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        // Allocate temporary storage
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, _col->get_device_ptr(), res.get_device_ptr() , _col->size(), op, 0);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Perform inclusive sum scan
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, _col->get_device_ptr(), res.get_device_ptr() , _col->size(), op, 0);
        // Free temporary storage
        cudaFree(d_temp_storage);

        res.download();

        return res.get_data();
    }
}

#endif //GDGRAPH_REDUCE_HPP
