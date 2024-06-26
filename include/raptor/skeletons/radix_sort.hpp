//
// Created by david on 20-10-2023.
//

#ifndef RAPTOR_RADIX_SORT_HPP
#define RAPTOR_RADIX_SORT_HPP

namespace raptor {


    template <typename T, template<typename> class ColType>
    ColType<T> radix_sort(ColType<T>& col) {

        int size = col.size();
        vector<T> result(size);

        collection<T>* _col = static_cast<collection<T>*>(&col);
        collection<T>* _result = static_cast<collection<T>*>(&result);

        _col->upload();
        _result->upload();

        // Allocate temporary device storage
        size_t temp_storage_bytes = 0;
        void* d_temp_storage = nullptr;

        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, _col->get_device_data(), _result->get_device_data(), size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Remove duplicates using CUB
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, _col->get_device_data(), _result->get_device_data(), size);
        // Free temporary storage
        cudaFree(d_temp_storage);

        _result->dirty_device();
        return result;
    }
}

#endif //RAPTOR_RADIX_SORT_HPP
