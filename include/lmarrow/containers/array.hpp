//
// Created by david on 13-10-2023.
//

#ifndef GDGRAPH_ARRAY_HPP
#define GDGRAPH_ARRAY_HPP

#include <array>
#include <algorithm>
#include <memory>
#include <functional>

#include "collection.hpp"
#include "lmarrow/cuda/dev_ptr.hpp"
#include "lmarrow/util/fillers.hpp"

namespace lmarrow {

    template<typename T, std::size_t N>
    class array : collection<T> {

        friend class vector<lmarrow::array<T, N>>;

    public:

        array() {

            dev_alloc = true;
            host_alloc = true;
        }

        ~array() {

            device_data_ptr = nullptr;
        }

        void free(cudaStream_t stream = 0) {

            if(device_data_ptr != nullptr) {

                if(stream != 0)
                    device_data_ptr.get()->freeAsync(stream);
                device_data_ptr = nullptr;
            }

            host_data.clear();
        }

        void fill(T &val) {

            if (host_alloc)
                allocate_host();

            if(child) {

                std::fill(host_data_parent_ptr, host_data_parent_ptr+N, val);
            }
            else {

                std::fill(host_data.begin(), host_data.end(), val);
            }

            dirty();
        }

        void fill(T &&val) {

            if (host_alloc)
                allocate_host();

            if(child) {

                std::fill(host_data_parent_ptr, host_data_parent_ptr+N, val);
            }
            else {

                std::fill(host_data.begin(), host_data.end(), val);
            }

            dirty();
        }

        void fill_on_device(T &val) {

            fill_on_device(value_filler(val));
        }

        void fill_on_device(T &&val) {

            fill_on_device(value_filler(val));
        }

        template<typename Functor>
        void fill(Functor fun) {

            if(host_alloc)
                allocate_host();

            if(child) {
                for (int i = 0; i < N; i++)
                    host_data_parent_ptr[i] = fun(i);
            }
            else {
                for (int i = 0; i < N; i++)
                    host_data[i] = fun(i);
            }

            dirty();
        }

        template<typename Functor>
        void fill_on_device(Functor fun) {

            if(dev_alloc)
                allocate_device();

            if(child) {
                dev_fill<<<def_nb(N), def_tpb(N)>>>(device_data_parent_ptr, N, fun);
            }
            else {
                dev_fill<<<def_nb(N), def_tpb(N)>>>(get_device_ptr(), N, fun);
            }

            dirty_on_device();
        }

        T &operator[](std::size_t i) {

            download();
            return child ? host_data_parent_ptr[i] : host_data[i];
        }

        T &get(std::size_t i) {

            download();
            return child ? host_data_parent_ptr[i] : host_data[i];
        }

        void set(std::size_t i, T &&val) {

            download();
            if(child) {
                host_data_parent_ptr[i] = val;
            }
            else {
                host_data[i] = val;
            }
            dirty();
        }

        void set(std::size_t i, T &val) {

            download();
            if(child) {
                host_data_parent_ptr[i] = val;
            }
            else {
                host_data[i] = val;
            }
            dirty();
        }

        bool contains(T&& val) {

            download();
            if(child) {
                return std::find(host_data_parent_ptr, host_data_parent_ptr+N, val) != host_data_parent_ptr+N;
            }
            else {
                return std::find(host_data.begin(), host_data.end(), val) != host_data.end();
            }
        }

        bool contains(T& val) {

            download();
            if(child) {
                return std::find(host_data_parent_ptr, host_data_parent_ptr+N, val) != host_data_parent_ptr+N;
            }
            else {
                return std::find(host_data.begin(), host_data.end(), val) != host_data.end();
            }
        }

        std::size_t size() { return N; }

        void copy(collection<T>& col) {

            this->download();
            col.download();
            if(child) {
                memcpy(host_data_parent_ptr, col.get_data(), sizeof(T) * N);
            }
            else {
                memcpy(host_data.data(), col.get_data(), sizeof(T) * N);
            }
            dirty();
        }

        void copy_on_device(collection<T>& col) {

            this->upload();
            col.upload();

            if(child) {
                cudaMemcpy(device_data_parent_ptr, col.get_device_ptr(), sizeof(T) * N, cudaMemcpyDeviceToDevice);
            }
            else {
                cudaMemcpy(get_device_ptr(), col.get_device_ptr(), sizeof(T) * N, cudaMemcpyDeviceToDevice);
            }
            dirty_on_device();
        }

        void dirty() {

            if(child) {
                set_parent_dirty();
            }
            host_dirty = true;
        }

        void dirty_on_device() {

            if(child) {
                set_parent_dirty_on_device();
            }
            dev_dirty = true;
        }

        T* get_device_ptr() {

            return device_data_ptr.get()->get();
        }

        T* get_data() {

            download();
            return host_data.data();
        }

        //protected:

        void upload(cudaStream_t stream = 0) {

            // Ensure dev allocation whenever upload is called
            if (dev_alloc) {
                allocate_device();
            }

            if (host_dirty) {
                if(child) {
                    // TODO: call parent upload? (maybe unnecessary)
                }
                else {
                    if(host_alloc) {
                        // Shouldn't happen
                    }
                    cudaMemcpyAsync(get_device_ptr(), host_data.data(), N * sizeof(T), cudaMemcpyHostToDevice, stream);
                }
                host_dirty = false;
            }
        }

        void download(cudaStream_t stream = 0) {

            // Ensure host allocation whenever download is called
            if(host_alloc) {
                allocate_host();
            }

            if(dev_dirty) {

                if(child) {
                    // TODO: call parent download? (maybe unnecessary)
                }
                else {
                    if (dev_alloc) {
                        // Shouldn't happen
                    } else {
                        cudaMemcpyAsync(host_data.data(), get_device_ptr(), N * sizeof(T), cudaMemcpyDeviceToHost,
                                        stream);
                    }
                }
                dev_dirty = false;
            }
        }

    private:

        void allocate_device() {

            if(child) {
                // TODO: call parent allocate_device ? (maybe not necessary)
            }
            else{
                device_data_ptr = std::make_shared<dev_ptr<T>>(N * sizeof(T));
            }
            dev_alloc = false;
        }

        void allocate_host() {

            if(child) {
                // TODO: call parent allocate_host ? (maybe not necessary)
            }
            else {
                host_data.resize(N);
            }
            host_alloc = false;
        }

        bool child = false;
        bool dev_alloc = true;
        bool host_alloc = true;
        bool host_dirty = false;
        bool dev_dirty = false;

        T* host_data_parent_ptr = nullptr;
        std::vector<T> host_data;
        T* device_data_parent_ptr = nullptr;
        std::shared_ptr<dev_ptr<T>> device_data_ptr = nullptr;


    protected:

        /* ############################### parent container logic ###################################
         *
         *         In case array is an element of a parent container (vector_of_array), it must
         *         update the parent's container dirty elements
         *
         *         TODO:
         *             - possibly find a cleaner solution for this
         *             - should be protected and not public
         */

        void set_parent_dirty() {

            if(parent_dirty_index_callback != nullptr)
                parent_dirty_index_callback(parent_index);
        }

        void set_parent_dirty_on_device() {

            if(parent_dirty_on_device_callback != nullptr)
                parent_dirty_on_device_callback();
        }

        int parent_index = 0;
        std::function<void(std::size_t)> parent_dirty_index_callback = nullptr;
        std::function<void(void)> parent_dirty_on_device_callback = nullptr;
    };
}

#endif //GDGRAPH_ARRAY_HPP
