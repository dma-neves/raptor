//
// Created by david on 13-10-2023.
//

#ifndef RAPTOR_COLLECTION_HPP
#define RAPTOR_COLLECTION_HPP

#include "raptor/cuda/grid.hpp"
#include "raptor/containers/fill.hpp"

namespace raptor {

    template<typename T, std::size_t N>
    class array;

    enum sync_granularity {
        FINE = 0,
        COARSE = 1
    };

    enum target {
        HOST = 0,
        DEVICE = 1
    };

    template<typename T>
    class collection {

    public:

        template <typename U>
        struct base_data_type {
            using type = U;
        };

        template <typename U, std::size_t N>
        struct base_data_type<array<U, N>> {
            using type = U;
        };

        virtual void free() = 0;

        template <target targ = HOST>
        void fill(typename base_data_type<T>::type &val);

        template <target targ = HOST>
        void fill(typename base_data_type<T>::type &&val);

        template<typename Functor, target targ = HOST>
        void fill(Functor fun) ;

        virtual T& operator[](std::size_t i) = 0;

        virtual T& get(std::size_t i) = 0;

        virtual void set(std::size_t i, T &&val) = 0;

        virtual void set(std::size_t i, T &val) = 0;

        virtual std::size_t size() = 0;

        template <target targ = HOST>
        void copy(collection<T>& col) {}

        bool contains(typename base_data_type<T>::type &val) ;

        bool contains(typename base_data_type<T>::type &&val) ;

        virtual void dirty_host() = 0;

        virtual void dirty_device() = 0;

        virtual typename base_data_type<T>::type* get_device_data() = 0;

        virtual typename base_data_type<T>::type* get_host_data() = 0;

        virtual void upload() = 0;

        virtual void download() = 0;

    protected:
        virtual void free(cudaStream_t stream = 0) = 0;

        virtual void upload(cudaStream_t stream, bool ignore_dirty) = 0;

        virtual void download(cudaStream_t stream, bool ignore_dirty) = 0;
    };
}

#endif //RAPTOR_COLLECTION_HPP
