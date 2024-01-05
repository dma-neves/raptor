//
// Created by david on 13-10-2023.
//

#ifndef GDGRAPH_COLLECTION_HPP
#define GDGRAPH_COLLECTION_HPP

#include "lmarrow/cuda/grid.hpp"
#include "lmarrow/containers/fill.hpp"

namespace lmarrow {

    enum sync_granularity {
        FINE = 0,
        COARSE = 1
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
