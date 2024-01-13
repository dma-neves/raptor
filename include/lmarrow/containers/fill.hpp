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
void dev_fill_2d(T *v, std::size_t size_y, std::size_t size_x, Functor fun) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    std::size_t index_y = index / size_x;
    std::size_t index_x = index % size_x;

    if (index < size_y*size_x)
        v[index] = fun(index_y, index_x);
}

#endif //LMARROW_FILL_HPP
