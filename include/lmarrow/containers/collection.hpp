//
// Created by david on 13-10-2023.
//

#ifndef GDGRAPH_COLLECTION_HPP
#define GDGRAPH_COLLECTION_HPP

#include "lmarrow/cuda/grid.hpp"

namespace lmarrow {

    enum sync_granularity {
        FINE = 0,
        COARSE = 1
    };

    template<typename T, typename Functor>
    __global__
    void dev_fill(T *v, std::size_t size, Functor fun) {

        std::size_t index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < size)
            v[index] = fun(index);
    }

    template <typename T>
    struct fill_val_fun {

        T val;

        fill_val_fun(T val) : val(val) {}

        __device__
        T operator()(std::size_t i) {
            return val;
        }
    };

    template<typename T>
    class collection {

    public:

        virtual void free(cudaStream_t stream = 0) = 0;

        virtual void fill(T &val) = 0;

        virtual void fill(T &&val) = 0;

        virtual void fill_on_device(T &val) = 0;

        virtual void fill_on_device(T &&val) = 0;


//        template<typename Functor>
//        void fill(Functor fun) = 0;

//        template<typename Functor>
//        void fill_on_device(Functor fun);

        virtual T &operator[](std::size_t i) = 0;

        virtual T& get(std::size_t i) = 0;

        virtual void set(std::size_t i, T &&val) = 0;

        virtual void set(std::size_t i, T &val) = 0;

        virtual std::size_t size() = 0;

        virtual void copy(collection<T>& col) = 0;

        virtual void copy_on_device(collection<T>& col) = 0;

        virtual void dirty() = 0;

        virtual void dirty_on_device() = 0;


        //protected:
        virtual T *get_device_ptr() = 0;

        virtual T *get_data() = 0;

        virtual void upload(cudaStream_t stream = 0) = 0;

        virtual void download(cudaStream_t stream = 0) = 0;
    };
}

#endif //GDGRAPH_COLLECTION_HPP
