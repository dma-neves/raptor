//
// Created by david on 15-10-2023.
//

#ifndef GDGRAPH_SCAN_HPP
#define GDGRAPH_SCAN_HPP

#include <cub/cub.cuh>

#include "operators.hpp"
#include "lmarrow/containers/vector.hpp"

namespace lmarrow {


    template <typename Operator, typename T, template<typename> class ColType>
    ColType<T> scan(ColType<T>& col) {

        Operator op;

        int size = col.size();
        ColType<T> result(size);

        collection<T>* _col = static_cast<collection<T>*>(&col);
        collection<T>* _result = static_cast<collection<T>*>(&result);

        _col->upload();
        _result->upload();

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        // Allocate temporary storage
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, _col->get_device_ptr(), _result->get_device_ptr(), op, size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Perform inclusive sum scan
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, _col->get_device_ptr(), _result->get_device_ptr(), op, size);
        // Free temporary storage
        cudaFree(d_temp_storage);

        _result->flag_device_dirty();
        return result;
    }
}

#endif //GDGRAPH_SCAN_HPP
