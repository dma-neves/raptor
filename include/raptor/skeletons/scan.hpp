//
// Created by david on 15-10-2023.
//

#ifndef RAPTOR_SCAN_HPP
#define RAPTOR_SCAN_HPP

#include <cub/cub.cuh>

#include "operators.hpp"
#include "raptor/containers/vector.hpp"

namespace raptor {


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

        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, _col->get_device_data(), _result->get_device_data(), op, size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, _col->get_device_data(), _result->get_device_data(), op, size);
        cudaFree(d_temp_storage);

        _result->dirty_device();
        return result;
    }
}

#endif //RAPTOR_SCAN_HPP
