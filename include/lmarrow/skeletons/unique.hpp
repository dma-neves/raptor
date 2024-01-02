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

        // Allocate temporary device storage
        size_t temp_storage_bytes = 0;
        void* d_temp_storage = nullptr;

        cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, col.get_device_ptr(),_unique_col->get_device_ptr(), num_selected_out.get_device_ptr(), size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Remove duplicates using CUB
        cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, col.get_device_ptr(),_unique_col->get_device_ptr(), num_selected_out.get_device_ptr(), size);

        // Free temporary storage
        cudaFree(d_temp_storage);

        //unique_vec.download();

        num_selected_out.download();

        ColType<T> result(num_selected_out.get_data());
        collection<T>* _result = static_cast<collection<T>*>(&result);
        _result->upload();

        _result->copy_on_device(unique_col);
       _result->flag_device_dirty();
        return result;
    }
}


#endif //GDGRAPH_UNIQUE_HPP
