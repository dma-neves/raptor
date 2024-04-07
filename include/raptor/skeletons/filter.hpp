//
// Created by david on 17-10-2023.
//

#ifndef RAPTOR_FILTER_HPP
#define RAPTOR_FILTER_HPP

#include <cub/cub.cuh>

#include "operators.hpp"
#include "reduce.hpp"
#include "raptor/containers/vector.hpp"

namespace raptor {


    template <typename T, template<typename> class ColType>
    //typename std::enable_if<std::is_base_of<collection<T>, ColType<T>>::value, ColType<T>>::type // Only enable if ColType inherits from collection<T>
    ColType<T> filter(ColType<T>& col, vector<int>& flags) {

        scalar<int> num_selected_out;
        auto filtered_vec_size = (std::size_t)reduce<sum<int>>(flags).get();
        ColType<T> filtered_col(filtered_vec_size);

        auto* _col = static_cast<collection<T>*>(&col);
        auto* _filtered_col = static_cast<collection<T>*>(&filtered_col);

        _col->upload();
        flags.upload();
        _filtered_col->upload();
        num_selected_out.upload();

        // Allocate temporary storage
        size_t temp_storage_bytes = 0;
        void *d_temp_storage = nullptr;
        cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, _col->get_device_data(), flags.get_device_data(), _filtered_col->get_device_data(), num_selected_out.get_device_data(), _col->size(), cudaStreamDefault);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Perform the Flagged operation
        cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, _col->get_device_data(), flags.get_device_data(), _filtered_col->get_device_data(), num_selected_out.get_device_data(), _col->size(), cudaStreamDefault);
        num_selected_out.download();

        cudaFree(d_temp_storage);

        _filtered_col->dirty_device();
        return filtered_col;
    }
}

#endif //RAPTOR_FILTER_HPP
