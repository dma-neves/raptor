//
// Created by david on 13-10-2023.
//

#ifndef GDGRAPH_VECTOR_OF_ARRAY_HPP
#define GDGRAPH_VECTOR_OF_ARRAY_HPP

#include <vector>
#include <set>
#include <chrono>
#include <iostream>

#include "vector_base.hpp"
#include "collection.hpp"
#include "array.hpp"
#include "fill.hpp"

#define FLAT_RESERVED_SIZE reserved_size * N
#define FLAT_CURRENT_SIZE current_size * N

namespace lmarrow {

    template<typename U, std::size_t N>
    class vector<lmarrow::array<U, N>> : public collection<array<U,N>>
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

        void free(cudaStream_t stream = 0) {

            if(device_data_ptr != nullptr) {
                if(stream != 0)
                    device_data_ptr.get()->freeAsync(stream);
                device_data_ptr = nullptr;
            }

            child_arrays.clear();
            host_data.clear();
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

        void copy(collection<array<U,N>>& col) {

            // TODO
        }

        void copy_on_device(collection<array<U,N>>& col) {

            // TODO
        }

        void fill(U& val) {

            fill(value_filler_2d<U>(val));
        }

        void fill(U&& val) {

            fill(value_filler_2d<U>(val));
        }

        void fill_on_device(U& val) {

            fill(value_filler_2d<U>(val));
        }

        void fill_on_device(U&& val) {

            fill(value_filler_2d<U>(val));
        }

        template<typename Functor>
        void fill(Functor fun) {

            if(host_realloc)
                allocate_host();

            for (int i = 0; i < current_size; i++) {
                for (int j = 0; j < N; j++) {
                    host_data[i * N + j] = fun(i, j);
                }
            }

            dirty();
        }

        template<typename Functor>
        void fill_on_device(Functor fun) {

            if(dev_realloc)
                allocate_device();

            dev_fill_2d<<<def_nb(FLAT_CURRENT_SIZE), def_tpb(FLAT_CURRENT_SIZE)>>>(get_device_ptr(),current_size, N, fun);

            dirty_on_device();
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
            memcpy(&host_data[current_size*N], val.get_data(), sizeof(U)*N);
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
            memcpy(&host_data[current_size*N], val.get_data(), sizeof(U)*N);
            dirty_index(current_size);
            current_size++;
        }

        std::size_t size() {

            return current_size;
        }

        array<U, N>& operator[](std::size_t i) {

            download();
            //dirty_index(i);
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
            memcpy(&host_data[i*N], val.get_data(), sizeof(U)*N);
            dirty_index(i);
        }

        void set(std::size_t i, array<U, N> &&val) {

            setup_child_array(val, i);

            download();
            child_arrays[i] = val;
            memcpy(&host_data[i*N], val.get_data(), sizeof(U)*N);
            dirty_index(i);
        }

        void dirty() {
            host_dirty = true;
        }

        void dirty_on_device() {
            dev_dirty = true;
        }

    //protected:

        void upload(cudaStream_t stream = 0, bool ignore_dirty = false) {

            // TODO: copy whole vector in a single cudamemcpy if the number of dirty elements
            //  is very large (ex: more than 50% of all elements)

            std::size_t n_elements_to_copy = std::min(current_size, child_arrays.size()); // only copy elements that are already on host

            // Ensure dev allocation whenever upload is called
            if(dev_realloc) {
                allocate_device();

                // When we reallocate the device its data is whiped
                // and we must consider all elements on host dirty.
                // But actually only if the host has any usefull data to copy
                if(n_elements_to_copy > 0)
                    dirty();
            }

            if(host_dirty || host_dirty_elements.size() > 0) {

                if (host_realloc) {
                    allocate_host();
                }

                if(host_dirty) {

                    std::size_t _size = sizeof(U) * n_elements_to_copy*N;
                    cudaMemcpyAsync(get_device_ptr(), host_data.data(), _size, cudaMemcpyHostToDevice, stream);
                }
                else if(host_dirty_elements.size() > 0) {

                    bool default_stream = (stream == 0);
                    if(default_stream) {
                        // If default stream is passed, create a new stream to parallelize the mem copies
                        cudaError_t streamCreationError = cudaStreamCreate(&stream);
                        if (streamCreationError != cudaSuccess) {

                            std::cerr << "Failed to create stream during upload" << std::endl;
                            default_stream = false;
                            stream = 0;
                        }
                    }

                    for (auto dirty_element: host_dirty_elements) {

                        if(dirty_element < n_elements_to_copy) {
                            U *dst = get_device_ptr() + dirty_element * N;
                            U *src = host_data.data() + dirty_element * N;
                            std::size_t _size = sizeof(U) * N;
                            cudaMemcpyAsync(dst, src, _size, cudaMemcpyHostToDevice, stream);
                        }
                    }

                    if(default_stream) {
                        cudaStreamSynchronize(stream);
                        cudaStreamDestroy(stream);
                    }
                }

                host_dirty_elements.clear();
                host_dirty = false;
            }
        }

        void download(cudaStream_t stream = 0, bool ignore_dirty = false) {

            // Ensure host allocation whenever download is called
            if(host_realloc)
                allocate_host();

            if(dev_dirty) {

                if (dev_realloc) {
                    // what should happen? Reallocating device wipes its data ...
                }
                else {

                    std::size_t _size = sizeof(U) * FLAT_CURRENT_SIZE;
                    cudaMemcpyAsync(host_data.data(), get_device_ptr(), _size, cudaMemcpyDeviceToHost, stream);
                }
                dev_dirty = false;
            }
        }

        U* get_device_ptr() {

            return device_data_ptr.get()->get();
        }

        U* get_data() {

            return host_data.data();
        }



    private:

        void increase_capacity() {

            // TODO: improve this
            reserve(reserved_size + DEFAULT_SIZE);
        }

        void dirty_index(std::size_t i) {

            // If device will be reallocated or the whole host is marked as dirty,
            // no need to track dirty elements
            if(!dev_realloc && !host_dirty) {
                if(granularity == COARSE)
                    host_dirty = true;
                else
                    host_dirty_elements.insert(i);
            }
        }

        void setup_child_array(array<U,N>& arr, std::size_t index) {

            arr.child = true;
            arr.host_data_parent_ptr = &host_data[index*N];
            arr.parent_index = index;
            arr.parent_dirty_index_callback = [&](std::size_t i) { dirty_index(i); };
            arr.parent_dirty_on_device_callback = [&]() { dirty_on_device(); };
        }

        void allocate_host() {

            std::size_t old_size = child_arrays.size();
            child_arrays.reserve(reserved_size);
            child_arrays.resize(current_size);
            if(host_data.size() < FLAT_RESERVED_SIZE) {
                host_data.resize(FLAT_RESERVED_SIZE);
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
        std::vector<U> host_data;
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

#endif //GDGRAPH_VECTOR_OF_ARRAY_HPP
