//
// Created by david on 13-10-2023.
//

#ifndef RAPTOR_VECTOR_OF_ARRAY_HPP
#define RAPTOR_VECTOR_OF_ARRAY_HPP

#include <vector>
#include <set>
#include <chrono>
#include <iostream>

#include "vector_base.hpp"
#include "collection.hpp"
#include "array.hpp"
#include "fill.hpp"
#include "raptor/cuda/error_check.hpp"

#define FLAT_RESERVED_SIZE reserved_size * N
#define FLAT_CURRENT_SIZE current_size * N

namespace raptor {

    template<typename U, std::size_t N>
    class vector<raptor::array<U, N>> : public collection<array<U,N>>
    {

    public:

        vector() {

            current_size = 0;
            reserved_size = 0;
            dev_realloc = false;
            host_realloc = false;
        }

        vector(std::size_t size) {

            reserved_size = size;
            current_size = size;

            dev_realloc = true;
            host_realloc = true;
        }

        ~vector() {

            free();
        }

        void free() {

            free(0);
        }

        void free(cudaStream_t stream) {

            if(device_data_ptr != nullptr) {
                if(stream != 0)
                    device_data_ptr.get()->freeAsync(stream);
                device_data_ptr = nullptr;
            }

            child_arrays.clear();
            host_data_ptr.clear();
            host_dirty_elements.clear();
            host_dirty = false;
            dev_dirty = false;
            dev_realloc = false;
            host_realloc = false;
        }

        void resize(std::size_t size) {

            current_size = size;
            if(child_arrays.size() != size) {
                host_realloc = true;
            }

            if (size > reserved_size) {
                reserved_size = size;
                dev_realloc = true;
                host_realloc = true; // Should be covered by previous case ?
            }
        }

        void reserve(std::size_t size) {

            reserved_size = size;
            host_realloc = true;
            dev_realloc = true;
        }

        void copy_on_host(collection<array<U,N>>& col) {

            // TODO
        }

        void copy_on_device(collection<array<U,N>>& col) {

            // TODO
        }

        template<target targ = HOST>
        void copy(collection<array<U,N>>& col) {

            switch(targ) {
                case HOST: copy_on_host(col);
                case DEVICE: copy_on_device(col);
            }
        }

        template<typename Functor>
        void fill_on_host(Functor fun) {

            if(host_realloc)
                allocate_host();

            for (int i = 0; i < current_size; i++) {
                for (int j = 0; j < N; j++) {
                    host_data_ptr[i * N + j] = fun(i, j);
                }
            }

            dirty_host();
        }

        template<typename Functor>
        void fill_on_device(Functor fun) {

            if(dev_realloc)
                allocate_device();

            dev_fill_2d<<<def_nb(FLAT_CURRENT_SIZE), def_tpb(FLAT_CURRENT_SIZE)>>>(get_device_data(), current_size, N, fun);

            dirty_device();
        }

        template<target targ = HOST, typename Functor>
        void fill(Functor fun) {

            switch(targ) {
                case HOST: fill_on_host(fun);
                case DEVICE: fill_on_device(fun);
            }
        }

        template <target targ = HOST>
        void fill(U &&val) {

            switch(targ) {
                case HOST: fill_on_host(value_filler_2d<U>(val));
                case DEVICE: fill_on_device(value_filler_2d<U>(val));
            }
        }

        template <target targ = HOST>
        void fill(U &val) {

            switch(targ) {
                case HOST: fill_on_host(value_filler_2d<U>(val));
                case DEVICE: fill_on_device(value_filler_2d<U>(val));
            }
        }

        void emplace_back() {

            array<U,N> val;

            if (current_size + 1 > reserved_size) {
                increase_capacity();
            }

            download();

            setup_child_array(val, current_size);

            child_arrays.push_back(val);
            dirty_index(current_size);
            current_size++;
        }

        void push_back(array<U, N> &val) {

            if (current_size + 1 > reserved_size) {
                increase_capacity();
            }

            download();

            setup_child_array(val, current_size);

            child_arrays.push_back(val);
            memcpy(&host_data_ptr[current_size * N], val.get_host_data(), sizeof(U) * N);
            dirty_index(current_size);
            current_size++;
        }

        void push_back(array<U, N> &&val) {

            if (current_size + 1 > reserved_size) {
                increase_capacity();
            }

            download();

            setup_child_array(val, current_size);

            child_arrays.push_back(val);
            memcpy(&host_data_ptr[current_size * N], val.get_host_data(), sizeof(U) * N);
            dirty_index(current_size);
            current_size++;
        }

        std::size_t size() {

            return current_size;
        }

        array<U, N>& operator[](std::size_t i) {

            download();
            dirty_index(i);
            return child_arrays[i];
        }

        array<U, N>& get(std::size_t i) {

            download();
            return child_arrays[i];
        }

        void set(std::size_t i, array<U, N> &val) {

            setup_child_array(val, i);

            download();
            child_arrays[i] = val;
            memcpy(&host_data_ptr[i * N], val.get_host_data(), sizeof(U) * N);
            dirty_index(i);
        }

        void set(std::size_t i, array<U, N> &&val) {

            setup_child_array(val, i);

            download();
            child_arrays[i] = val;
            memcpy(&host_data_ptr[i * N], val.get_host_data(), sizeof(U) * N);
            dirty_index(i);
        }

        void dirty_host() {
            host_dirty = true;
        }

        void dirty_device() {
            dev_dirty = true;
        }

        bool contains(U&& val) {

            download();
            for(int i = 0; i < current_size; i++) {
                if(child_arrays[i].contains(val)) {
                    return true;
                }
            }

            return false;
        }

        bool contains(U& val) {

            download();
            for(int i = 0; i < current_size; i++) {
                if(child_arrays[i].contains(val)) {
                    return true;
                }
            }

            return false;
        }

    //protected:

        void upload() {

            upload(0,false);
        }

        void upload(cudaStream_t stream, bool ignore_dirty) {

            // TODO: copy whole vector in a single cudamemcpy if the number of dirty_host elements
            //  is very large (ex: more than 50% of all elements)

            if(ignore_dirty) {
                host_dirty = false;
                host_dirty_elements.clear();
            }

            std::size_t n_elements_to_copy = std::min(current_size, child_arrays.size()); // only copy elements that are already on host

            // Ensure dev allocation whenever upload is called
            if(dev_realloc) {
                allocate_device();

                // When we reallocate the device its data is whiped
                // and we must consider all elements on host dirty_host.
                // But actually only if the host has any usefull data to copy
                if(n_elements_to_copy > 0)
                    dirty_host();
            }

            if(host_dirty || host_dirty_elements.size() > 0) {

                if (host_realloc) {
                    allocate_host();
                }

                if(host_dirty) {

                    std::size_t _size = sizeof(U) * n_elements_to_copy*N;
                    cudaErrorCheck( cudaMemcpyAsync(get_device_data(), host_data_ptr.data(), _size, cudaMemcpyHostToDevice, stream) );
                }
                else if(host_dirty_elements.size() > 0) {

                    bool using_default_stream = (stream == 0);
                    if(using_default_stream) {
                        // If default stream is passed, create a new stream to parallelize the mem copies
                        cudaErrorCheck( cudaStreamCreate(&stream) );
                    }

                    std::size_t _size = sizeof(U) * N;
                    for (auto dirty_element: host_dirty_elements) {

                        if(dirty_element < n_elements_to_copy) {
                            U *dst = get_device_data() + dirty_element * N;
                            U *src = host_data_ptr.data() + dirty_element * N;
                            cudaErrorCheck( cudaMemcpyAsync(dst, src, _size, cudaMemcpyHostToDevice, stream) );
                        }
                    }

                    if(using_default_stream) {
                        cudaErrorCheck( cudaStreamSynchronize(stream) );
                        cudaErrorCheck( cudaStreamDestroy(stream) );
                    }
                }

                host_dirty_elements.clear();
                host_dirty = false;
            }
        }

        void download() {

            download(0,false);
        }

        void download(cudaStream_t stream, bool ignore_dirty) {

            if(ignore_dirty) {
                dev_dirty = false;
            }

            // Ensure host allocation whenever download is called
            if(host_realloc)
                allocate_host();

            if(dev_dirty) {

                if (dev_realloc) {
                    // what should happen? Reallocating device wipes its data ...
                }
                else {

                    std::size_t _size = sizeof(U) * FLAT_CURRENT_SIZE;
                    cudaErrorCheck( cudaMemcpyAsync(host_data_ptr.data(), get_device_data(), _size, cudaMemcpyDeviceToHost, stream) );
                }
                dev_dirty = false;
            }
        }

        U* get_device_data() {

            return device_data_ptr.get()->get();
        }

        U* get_host_data() {

            return host_data_ptr.data();
        }



    private:

        void increase_capacity() {

            // TODO: improve this
            reserve(reserved_size + DEFAULT_SIZE);
        }

        void dirty_index(std::size_t i) {

            // If device will be reallocated or the whole host is marked as dirty_host,
            // no need to track dirty_host elements
            if(!dev_realloc && !host_dirty) {
                if(granularity == COARSE)
                    host_dirty = true;
                else
                    host_dirty_elements.insert(i);
            }
        }

        void setup_child_array(array<U,N>& arr, std::size_t index) {

            arr.child = true;
            arr.host_data_parent_ptr = &host_data_ptr[index * N];
            arr.parent_index = index;
            arr.parent_dirty_index_callback = [&](std::size_t i) { dirty_index(i); };
            arr.parent_dirty_device_callback = [&]() { dirty_device(); };
        }

        void allocate_host() {

            std::size_t old_size = child_arrays.size();
            child_arrays.reserve(reserved_size);
            child_arrays.resize(current_size);
            if(host_data_ptr.size() < FLAT_RESERVED_SIZE) {
                host_data_ptr.resize(FLAT_RESERVED_SIZE);
            }

            // TODO: improve this?
            for(int i = old_size; i < current_size; i++) {
                setup_child_array(child_arrays[i], i);
            }

            host_realloc = false;
        }

        void allocate_device() {

            device_data_ptr = std::make_shared<dev_ptr<U>>(FLAT_RESERVED_SIZE * sizeof(U) );
            dev_realloc = false;
        }

        std::vector<array<U,N>> child_arrays;
        std::vector<U> host_data_ptr;
        std::shared_ptr<dev_ptr<U>> device_data_ptr = nullptr;

        std::size_t reserved_size = 0;
        std::size_t current_size = 0;

        sync_granularity granularity = FINE;
        bool dev_realloc = false;
        bool host_realloc = false;
        bool host_dirty = false;
        std::set<std::size_t> host_dirty_elements;
        bool dev_dirty = false;
    };

}

#endif //RAPTOR_VECTOR_OF_ARRAY_HPP
