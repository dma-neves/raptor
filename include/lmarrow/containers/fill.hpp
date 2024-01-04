//
// Created by david on 04-01-2024.
//

#ifndef LMARROW_FILL_HPP
#define LMARROW_FILL_HPP

template<typename T, typename Functor>
__global__
void dev_fill(T *v, std::size_t size, Functor fun) {

    std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size)
        v[index] = fun(index);
}

template<typename T, typename Functor>
__global__
void dev_fill_flat(T *v, std::size_t size, std::size_t garr_size, Functor fun) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    std::size_t gvec_index = index / garr_size;
    std::size_t garr_index = index % garr_size;

    if (index < size)
        v[index] = fun(gvec_index, garr_index);
}

#endif //LMARROW_FILL_HPP
