//
// Created by david on 08-10-2023.
//

#ifndef RAPTOR_VECTOR_BASE_HPP
#define RAPTOR_VECTOR_BASE_HPP

#include <vector>
#include <set>
#include <functional>
#include <algorithm>
#include <memory>

#include "collection.hpp"
#include "raptor/cuda/dev_ptr.hpp"
#include "raptor/util/fillers.hpp"

namespace raptor {

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

            host_data_ptr = std::make_shared<std::vector<T>>();
        }

        vector(std::size_t size, sync_granularity granularity = sync_granularity::COARSE) {

            this->granularity = granularity;
            reserved_size = size;
            current_size = size;
            dev_realloc = true;
            host_realloc = true;

            host_data_ptr = std::make_shared<std::vector<T>>();
        }

        ~vector() {

            free();
        }

        void free() {

            free(0);
        }

        void free(cudaStream_t stream) {

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

            if(host_data_ptr != nullptr) {
                host_data_ptr.reset();
            }

            host_dirty_elements.clear();
            host_dirty = false;
            dev_dirty = false;
            dev_realloc = false;
            host_realloc = false;
        }

        void resize(std::size_t size) {

            current_size = size;
            if(host_data_ptr->size() != size) {
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

        void fill_on_host(T &&val) {

            download(0,true);
            std::fill(host_data_ptr->begin(), host_data_ptr->end(), val);
            dirty_host();
        }

        void fill_on_host(T &val) {

            download(0,true);
            std::fill(host_data_ptr->begin(), host_data_ptr->end(), val);
            dirty_host();
        }

        void fill_on_device(T &&val) {

            upload(0,true);

            if constexpr (std::is_arithmetic_v<T>) {

                if(val == 0) {
                    init_stream();
                    cudaMemsetAsync(get_device_data(), 0, sizeof(T) * current_size, *stream_ptr);
                }
                else {
                    fill_on_device(value_filler(val));
                }
            }
            else {
                fill_on_device(value_filler(val));
            }
            dirty_device();
        }

        void fill_on_device(T &val) {

            upload(0,true);

            if constexpr (std::is_arithmetic_v<T>) {

                if(val == 0) {
                    init_stream();
                    cudaMemsetAsync(get_device_data(), 0, sizeof(T) * current_size, *stream_ptr);
                }
                else {
                    fill_on_device(value_filler(val));
                }
            }
            else {
                fill_on_device(value_filler(val));
            }
            dirty_device();
        }

        template <target targ = HOST>
        void fill(T &val) {

            switch(targ) {
                case HOST: fill_on_host(val);
                case DEVICE: fill_on_device(val);
            }
        }

        template <target targ = HOST>
        void fill(T &&val) {

            switch(targ) {
                case HOST: fill_on_host(val);
                case DEVICE: fill_on_device(val);
            }
        }


        template<typename Functor>
        void fill_on_host(Functor fun) {

            download(0,true);

            for (int i = 0; i < current_size; i++)
                (*host_data_ptr)[i] = fun(i);
            dirty_host();
        }

        template<typename Functor>
        void fill_on_device(Functor fun) {

            upload(0,true);

            init_stream();
            dev_fill<<<def_nb(current_size), def_tpb(current_size), 0, *stream_ptr>>>(get_device_data(), current_size, fun);
            dirty_device();
        }

        template<target targ = HOST, typename Functor>
        void fill(Functor fun) {

            switch(targ) {
                case HOST: fill_on_host(fun);
                case DEVICE: fill_on_device(fun);
            }
        }

        void emplace_back() {

            if (current_size + 1 > reserved_size) {
                increase_capacity();
            }

            download();

            host_data_ptr->emplace_back();
            dirty_index(current_size);
            current_size++;
        }

        void push_back(T &val) {

            if (current_size + 1 > reserved_size) {
                increase_capacity();
            }

            download();

            host_data_ptr->push_back(val);
            dirty_index(current_size);
            current_size++;
        }

        void push_back(T &&val) {

            if (current_size + 1 > reserved_size) {
                increase_capacity();
            }

            download();

            host_data_ptr->push_back(val);
            dirty_index(current_size);
            current_size++;
        }

        std::size_t size() {

            return current_size;
        }

        T &operator[](std::size_t i) {

            download();
            dirty_index(i);
            return (*host_data_ptr)[i];
        }

        T &get(std::size_t i) {

            download();
            return (*host_data_ptr)[i];
        }

        void set(std::size_t i, T &val) {

            download();
            (*host_data_ptr)[i] = val;
            dirty_index(i);
        }

        void set(std::size_t i, T &&val) {

            download();

//            vec[i] = std::forward(val);
            (*host_data_ptr)[i] = val;
            dirty_index(i);
        }

        bool contains(T&& val) {

            download();
            return std::find(host_data_ptr->begin(), host_data_ptr->end(), val) != host_data_ptr->end();
        }

        bool contains(T& val) {

            download();
            return std::find(host_data_ptr->begin(), host_data_ptr->end(), val) != host_data_ptr->end();
        }

        void copy_on_host(collection<T>& col) {

            this->resize(col.size());
            this->download(0,true);
            col.download();

            init_stream();
            cudaMemcpyAsync(get_host_data(), col.get_host_data(), sizeof(T) * current_size, cudaMemcpyHostToHost, *stream_ptr);
            dirty_host();
        }

        void copy_on_device(collection<T>& col) {

            this->resize(col.size());
            this->upload(0,true);
            col.upload();

            init_stream();
            cudaMemcpyAsync(get_device_data(), col.get_device_data(), sizeof(T) * current_size, cudaMemcpyDeviceToDevice, *stream_ptr);
            dirty_device();
        }

        template<target targ = HOST>
        void copy(collection<T>& col) {

            switch(targ) {
                case HOST: copy_on_host(col);
                case DEVICE: copy_on_device(col);
            }
        }

        void dirty_host() {
            host_dirty = true;
        }

        void dirty_device() {
            dev_dirty = true;
        }

    //protected:

        T* get_device_data() {

            return device_data_ptr.get()->get();
        }

        T* get_host_data() {

            return host_data_ptr->data();
        }

        void upload() {

            upload(0, false);
        }

        void upload(cudaStream_t stream, bool ignore_dirty) {

            if(stream_ptr != nullptr) {
                cudaStreamSynchronize(*stream_ptr);
            }

            if(ignore_dirty) {
                host_dirty = false;
                host_dirty_elements.clear();
            }

            // only copy elements that are already on host
            std::size_t n_elements_to_copy = std::min(current_size, host_data_ptr->size());

            // Ensure dev allocation whenever upload is called
            if(dev_realloc) {
                allocate_device();

                // When we reallocate the device its data is wiped
                // and we must consider all elements on host dirty_host.
                // But actually only if the host has any useful data to copy
                if(n_elements_to_copy > 0) {
                    dirty_host();
                }
            }

            if(host_dirty || host_dirty_elements.size() > 0) {

                if (host_realloc) {
                    allocate_host();
                }

                if(host_dirty) {
                    cudaMemcpyAsync(get_device_data(), host_data_ptr->data(), n_elements_to_copy * sizeof(T), cudaMemcpyHostToDevice, stream);
                }
                else if (host_dirty_elements.size() > 0) {

                    for (auto dirty_element: host_dirty_elements) {

                        if(dirty_element < n_elements_to_copy) {
                            T *dst = get_device_data() + dirty_element;
                            T *src = &(*host_data_ptr)[dirty_element];
                            std::size_t _size = sizeof(T);
                            cudaMemcpyAsync(dst, src, _size, cudaMemcpyHostToDevice, stream);
                        }
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

                    cudaMemcpyAsync(host_data_ptr->data(), get_device_data(), current_size * sizeof(T), cudaMemcpyDeviceToHost, stream);
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

            // If the whole host is marked as dirty_host,
            // no need to track dirty_host elements
            if(!host_dirty) {
                if(granularity == FINE) {
                    host_dirty_elements.insert(i);
                }
                else {
                    host_dirty = true;
                }
            }
        }

        void allocate_host() {

            host_data_ptr->reserve(reserved_size);
            host_data_ptr->resize(current_size);
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

        std::shared_ptr<std::vector<T>> host_data_ptr = nullptr;
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

#endif //RAPTOR_VECTOR_BASE_HPP
