//
// Created by david on 20-10-2023.
//

#ifndef GDGRAPH_UNIQUE_HPP
#define GDGRAPH_UNIQUE_HPP

#include <cub/cub.cuh>

#include "operators.hpp"
#include "lmarrow/containers/vector.hpp"
#include "lmarrow/containers/scalar.hpp"

namespace lmarrow {


    template <typename T, template<typename> class ColType>
    ColType<T> unique(ColType<T>& col) {

        int size = col.size();
        ColType<T> unique_col(size);
        scalar<int> num_selected_out;

        collection<T>* _col = static_cast<collection<T>*>(&col);
        collection<T>* _unique_col = static_cast<collection<T>*>(&unique_col);

        _col->upload();
        _unique_col->upload();
        num_selected_out.upload();

        size_t temp_storage_bytes = 0;
        void* d_temp_storage = nullptr;

        cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, col.get_device_ptr(),_unique_col->get_device_ptr(), num_selected_out.get_device_ptr(), size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, col.get_device_ptr(),_unique_col->get_device_ptr(), num_selected_out.get_device_ptr(), size);

        cudaFree(d_temp_storage);

        num_selected_out.download();

        ColType<T> result(num_selected_out.get());
        collection<T>* _result = static_cast<collection<T>*>(&result);
        _result->upload();

        _result->copy_on_device(unique_col);
       _result->dirty_on_device();
        return result;
    }
}


#endif //GDGRAPH_UNIQUE_HPP
