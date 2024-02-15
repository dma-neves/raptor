//
// Created by david on 08-10-2023.
//

#ifndef GDGRAPH_VECTOR_BASE_HPP
#define GDGRAPH_VECTOR_BASE_HPP

#include <vector>
#include <set>
#include <functional>
#include <algorithm>
#include <memory>

#include "collection.hpp"
#include "lmarrow/cuda/dev_ptr.hpp"
#include "lmarrow/util/fillers.hpp"

namespace lmarrow {

#define DEFAULT_SIZE 1024

    template<typename T>
    class vector : public collection<T> {

    public:

        vector(sync_granularity granularity = sync_granularity::COARSE) {

            this->granularity = granularity;
            current_size = 0;
            reserved_size = 0;
            dev_realloc = false;
            host_realloc = false;

            host_data = std::make_shared<std::vector<T>>();
        }

        vector(std::size_t size, sync_granularity granularity = sync_granularity::COARSE) {

            this->granularity = granularity;

            reserved_size = size;
            current_size = size;

            dev_realloc = true;
            host_realloc = true;

            host_data = std::make_shared<std::vector<T>>();
        }

        ~vector() {

            free();
        }

        void free(cudaStream_t stream = 0) {

            if(stream_ptr != nullptr) {
                if(stream_ptr.unique()) {
                    cudaStreamSynchronize(*stream_ptr);
                    cudaStreamDestroy(*stream_ptr);
                }
                stream_ptr.reset();
            }

            if(device_data_ptr != nullptr) {
                if(stream != 0) {
                    device_data_ptr.get()->freeAsync(stream);
                }
                device_data_ptr.reset();
            }

            if(host_data != nullptr) {
                host_data.reset();
            }

            host_dirty_elements.clear();
            host_dirty = false;
            dev_dirty = false;
            dev_realloc = false;
            host_realloc = false;
        }

        void resize(std::size_t size) {

            current_size = size;
            if(host_data->size() != size) {
                host_realloc = true;
            }

            if (size > reserved_size) {
                reserved_size = size;
                dev_realloc = true;
                host_realloc = true; // Should be covered by previous case ?
            }
        }

        void reserve(std::size_t size) {

            if(reserved_size != size) {
                reserved_size = size;
                host_realloc = true;
                dev_realloc = true;
            }
        }

        void fill(T &&val) {

            download(0,true);
            std::fill(host_data->begin(), host_data->end(), val);
            dirty();
        }

        void fill(T &val) {

            download(0,true);
            std::fill(host_data->begin(), host_data->end(), val);
            dirty();
        }

        void fill_on_device(T &&val) {

            upload(0,true);

            if constexpr (std::is_arithmetic_v<T>) {

                if(val == 0) {
                    init_stream();
                    cudaMemsetAsync(get_device_ptr(), 0, sizeof(T) * current_size, *stream_ptr);
                }
                else {
                    fill_on_device(value_filler(val));
                }
            }
            else {
                fill_on_device(value_filler(val));
            }
            dirty_on_device();
        }

        void fill_on_device(T &val) {

            upload(0,true);

            if constexpr (std::is_arithmetic_v<T>) {

                if(val == 0) {
                    init_stream();
                    cudaMemsetAsync(get_device_ptr(), 0, sizeof(T) * current_size, *stream_ptr);
                }
                else {
                    fill_on_device(value_filler(val));
                }
            }
            else {
                fill_on_device(value_filler(val));
            }
            dirty_on_device();
        }

        template<typename Functor>
        void fill(Functor fun) {

            download(0,true);

            for (int i = 0; i < current_size; i++)
                (*host_data)[i] = fun(i);
            dirty();
        }

        template<typename Functor>
        void fill_on_device(Functor fun) {

            upload(0,true);

            init_stream();
            dev_fill<<<def_nb(current_size), def_tpb(current_size), 0, *stream_ptr>>>(get_device_ptr(), current_size, fun);
            dirty_on_device();
        }

        void emplace_back() {

            if (current_size + 1 > reserved_size) {
                increase_capacity();
            }

            download();

            host_data->emplace_back();
            dirty_index(current_size);
            current_size++;
        }

        void push_back(T &val) {

            if (current_size + 1 > reserved_size) {
                increase_capacity();
            }

            download();

            host_data->push_back(val);
            dirty_index(current_size);
            current_size++;
        }

        void push_back(T &&val) {

            if (current_size + 1 > reserved_size) {
                increase_capacity();
            }

            download();

            host_data->push_back(val);
            dirty_index(current_size);
            current_size++;
        }

        std::size_t size() {

            return current_size;
        }

        T &operator[](std::size_t i) {

            download();
            // dirty_index(i);
            return (*host_data)[i];
        }

        T &get(std::size_t i) {

            download();
            return (*host_data)[i];
        }

        void set(std::size_t i, T &val) {

            download();
            (*host_data)[i] = val;
            dirty_index(i);
        }

        void set(std::size_t i, T &&val) {

            download();

//            vec[i] = std::forward(val);
            (*host_data)[i] = val;
            dirty_index(i);
        }

        bool contains(T&& val) {

            download();
            return std::find(host_data->begin(), host_data->end(), val) != host_data->end();
        }

        bool contains(T& val) {

            download();
            return std::find(host_data->begin(), host_data->end(), val) != host_data->end();
        }

        void copy(collection<T>& col) {

            this->resize(col.size());
            this->download(0,true);
            col.download();

            init_stream();
            cudaMemcpyAsync(host_data->data(), col.get_data(), sizeof(T) * current_size, cudaMemcpyHostToHost, *stream_ptr);
            dirty();
        }

        void copy_on_device(collection<T>& col) {

            this->resize(col.size());
            this->upload(0,true);
            col.upload();

            init_stream();
            cudaMemcpyAsync(get_device_ptr(), col.get_device_ptr(), sizeof(T) * current_size, cudaMemcpyDeviceToDevice, *stream_ptr);
            dirty_on_device();
        }

        void dirty() {
            host_dirty = true;
        }

        void dirty_on_device() {
            dev_dirty = true;
        }

    //protected:

        T* get_device_ptr() {

            return device_data_ptr.get()->get();
        }

        T* get_data() {

            return host_data->data();
        }

        void upload(cudaStream_t stream = 0, bool ignore_dirty = false) {

            if(stream_ptr != nullptr) {
                cudaStreamSynchronize(*stream_ptr);
            }

            if(ignore_dirty) {
                host_dirty = false;
                host_dirty_elements.clear();
            }

            // only copy elements that are already on host
            std::size_t n_elements_to_copy = std::min(current_size, host_data->size());

            // Ensure dev allocation whenever upload is called
            if(dev_realloc) {
                allocate_device();

                // When we reallocate the device its data is wiped
                // and we must consider all elements on host dirty.
                // But actually only if the host has any useful data to copy
                if(n_elements_to_copy > 0) {
                    dirty();
                }
            }

            if(host_dirty || host_dirty_elements.size() > 0) {

                if (host_realloc) {
                    allocate_host();
                }

                if(host_dirty) {
                    cudaMemcpyAsync(get_device_ptr(), host_data->data(), n_elements_to_copy * sizeof(T), cudaMemcpyHostToDevice, stream);
                }
                else if (host_dirty_elements.size() > 0) {

                    for (auto dirty_element: host_dirty_elements) {

                        if(dirty_element < n_elements_to_copy) {
                            T *dst = get_device_ptr() + dirty_element;
                            T *src = &(*host_data)[dirty_element];
                            std::size_t _size = sizeof(T);
                            cudaMemcpyAsync(dst, src, _size, cudaMemcpyHostToDevice, stream);
                        }
                    }
                }
                

                host_dirty_elements.clear();
                host_dirty = false;
            }
        }

        void download(cudaStream_t stream = 0, bool ignore_dirty = false) {

            if(stream_ptr != nullptr) {
                cudaStreamSynchronize(*stream_ptr);
            }

            if(ignore_dirty) {
                dev_dirty = false;
            }

            // Ensure host allocation whenever download is called
            if(host_realloc) {
                allocate_host();
            }

            if(dev_dirty) {

                if (dev_realloc) {

                    // what should happen? Reallocating device wipes its data ...
                }
                else {

                    cudaMemcpyAsync(host_data->data(), get_device_ptr(), current_size * sizeof(T), cudaMemcpyDeviceToHost, stream);
                    dev_dirty = false;
                }
            }
        }

    private:

        void increase_capacity() {

            // TODO: Find better heuristic
            reserve(reserved_size + DEFAULT_SIZE);
        }

        void dirty_index(unsigned i) {

            // If the whole host is marked as dirty,
            // no need to track dirty elements
            if(!host_dirty) {
                if(granularity == FINE) {
                    host_dirty_elements.insert(i);
                }
                else {
                    host_dirty = true; // TODO: recheck this
                }
            }
        }

        void allocate_host() {

            host_data->reserve(reserved_size);
            host_data->resize(current_size);
            host_realloc = false;
        }

        void allocate_device() {

            device_data_ptr = std::make_shared<dev_ptr<T>>(reserved_size * sizeof(T) );
            dev_realloc = false;
        }

        void init_stream() {
            if(stream_ptr == nullptr) {
                stream_ptr = std::make_shared<cudaStream_t>();
                cudaStreamCreate(stream_ptr.get());
            }
        }

        std::shared_ptr<std::vector<T>> host_data = nullptr;
        std::shared_ptr<dev_ptr<T>> device_data_ptr = nullptr;

        std::size_t reserved_size = 0;
        std::size_t current_size = 0;

        sync_granularity granularity;
        bool dev_realloc = false;
        bool host_realloc = false;
        bool host_dirty = false;
        std::set<std::size_t> host_dirty_elements;
        bool dev_dirty = false;

        std::shared_ptr<cudaStream_t> stream_ptr = nullptr;
    };
}

#endif //GDGRAPH_VECTOR_BASE_HPP
