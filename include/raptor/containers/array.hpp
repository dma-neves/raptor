//
// Created by david on 13-10-2023.
//

#ifndef RAPTOR_ARRAY_HPP
#define RAPTOR_ARRAY_HPP

#include <array>
#include <algorithm>
#include <memory>
#include <functional>

#include "collection.hpp"
#include "raptor/cuda/dev_ptr.hpp"
#include "raptor/util/fillers.hpp"

namespace raptor {

    template<typename T, std::size_t N>
    class array : collection<T> {

        friend class vector<raptor::array<T, N>>;

    public:

        array() {

            dev_alloc = true;
            host_alloc = true;
        }

        ~array() {

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

                if(stream != 0)
                    device_data_ptr.get()->freeAsync(stream);
                device_data_ptr.reset();
            }

            if(host_data_ptr != nullptr) {
                host_data_ptr.reset();
            }

            dev_alloc = false;
            host_alloc = false;
            host_dirty = false;
            dev_dirty = false;
        }


        void fill_on_host(T &val) {

            download(0,true);

            if(child) {

                std::fill(host_data_parent_ptr, host_data_parent_ptr+N, val);
            }
            else {

                std::fill(host_data_ptr->begin(), host_data_ptr->end(), val);
            }

            dirty_host();
        }

        void fill_on_host(T &&val) {

            download(0,true);

            if(child) {

                std::fill(host_data_parent_ptr, host_data_parent_ptr+N, val);
            }
            else {

                std::fill(host_data_ptr->begin(), host_data_ptr->end(), val);
            }

            dirty_host();
        }

        void fill_on_device(T &&val) {

            upload(0,true);

            if constexpr (std::is_arithmetic_v<T>) {

                if(val == 0) {
                    init_stream();
                    cudaMemsetAsync(get_device_data(), 0, sizeof(T) * N, *stream_ptr);
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
                    cudaMemsetAsync(get_device_data(), 0, sizeof(T) * N, *stream_ptr);
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

            if(child) {
                for (int i = 0; i < N; i++)
                    host_data_parent_ptr[i] = fun(i);
            }
            else {
                for (int i = 0; i < N; i++)
                    (*host_data_ptr)[i] = fun(i);
            }

            dirty_host();
        }

        template<typename Functor>
        void fill_on_device(Functor fun) {

            upload(0,true);

            if(child) {
                dev_fill<<<def_nb(N), def_tpb(N)>>>(device_data_parent_ptr, N, fun);
            }
            else {

                init_stream();
                dev_fill<<<def_nb(N), def_tpb(N), 0, *stream_ptr>>>(get_device_data(), N, fun);
            }

            dirty_device();
        }

        template<target targ = HOST, typename Functor>
        void fill(Functor fun) {

            switch(targ) {
                case HOST: fill_on_host(fun);
                case DEVICE: fill_on_device(fun);
            }
        }

        T& operator[](std::size_t i) {

            download();
            dirty_host();
            return child ? host_data_parent_ptr[i] : (*host_data_ptr)[i];
        }

        T& get(std::size_t i) {

            download();
            return child ? host_data_parent_ptr[i] : (*host_data_ptr)[i];
        }

        void set(std::size_t i, T &&val) {

            download();
            if(child) {
                host_data_parent_ptr[i] = val;
            }
            else {
                (*host_data_ptr)[i] = val;
            }
            dirty_host();
        }

        void set(std::size_t i, T &val) {

            download();
            if(child) {
                host_data_parent_ptr[i] = val;
            }
            else {
                (*host_data_ptr)[i] = val;
            }
            dirty_host();
        }

        bool contains(T&& val) {

            download();
            if(child) {
                return std::find(host_data_parent_ptr, host_data_parent_ptr+N, val) != host_data_parent_ptr+N;
            }
            else {
                return std::find(host_data_ptr->begin(), host_data_ptr->end(), val) != host_data_ptr->end();
            }
        }

        bool contains(T& val) {

            download();
            if(child) {
                return std::find(host_data_parent_ptr, host_data_parent_ptr+N, val) != host_data_parent_ptr+N;
            }
            else {
                return std::find(host_data_ptr->begin(), host_data_ptr->end(), val) != host_data_ptr->end();
            }
        }

        std::size_t size() { return N; }

        void copy_on_host(collection<T>& col) {

            download(0,true);
            col.download();

            if(child) {
                memcpy(host_data_parent_ptr, col.get_host_data(), sizeof(T) * N);
            }
            else {
                memcpy(host_data_ptr->data(), col.get_host_data(), sizeof(T) * N);
            }
            dirty_host();
        }

        void copy_on_device(collection<T>& col) {

            upload(0,true);
            col.upload();

            if(child) {
                cudaMemcpy(device_data_parent_ptr, col.get_device_data(), sizeof(T) * N, cudaMemcpyDeviceToDevice);
            }
            else {
                cudaMemcpy(get_device_data(), col.get_device_data(), sizeof(T) * N, cudaMemcpyDeviceToDevice);
            }
            dirty_device();
        }

        template<target targ = HOST, typename Functor>
        void copy(collection<T>& col) {

            switch(targ) {
                case HOST: copy_on_host(col);
                case DEVICE: copy_on_device(col);
            }
        }

        void dirty_host() {

            if(child) {
                set_parent_dirty_host();
            }
            host_dirty = true;
        }

        void dirty_device() {

            if(child) {
                set_parent_dirty_device();
            }
            dev_dirty = true;
        }

        T* get_device_data() {

            return device_data_ptr.get()->get();
        }

        T* get_host_data() {

            download();
            return host_data_ptr->data();
        }

        //protected:

        void upload() {

            upload(0,false);
        }

        void upload(cudaStream_t stream, bool ignore_dirty) {

            if(stream_ptr != nullptr) {
                cudaStreamSynchronize(*stream_ptr);
            }

            if(ignore_dirty) {
                host_dirty = false;
            }

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
                    cudaMemcpyAsync(get_device_data(), host_data_ptr->data(), N * sizeof(T), cudaMemcpyHostToDevice, stream);
                }
                host_dirty = false;
            }
        }

        void download() {

            download(0, false);
        }

        void download(cudaStream_t stream, bool ignore_dirty) {

            if(stream_ptr != nullptr) {
                cudaStreamSynchronize(*stream_ptr);
            }

            if(ignore_dirty) {
                dev_dirty = false;
            }

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
                        cudaMemcpyAsync(host_data_ptr->data(), get_device_data(), N * sizeof(T), cudaMemcpyDeviceToHost, stream);
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
                host_data_ptr = std::make_shared<std::array<T,N>>();
            }
            host_alloc = false;
        }

        void init_stream() {
            if(stream_ptr == nullptr) {
                stream_ptr = std::make_shared<cudaStream_t>();
                cudaStreamCreate(stream_ptr.get());
            }
        }

        bool child = false;
        bool dev_alloc = true;
        bool host_alloc = true;
        bool host_dirty = false;
        bool dev_dirty = false;

        T* host_data_parent_ptr = nullptr;
        std::shared_ptr<std::array<T,N>> host_data_ptr = nullptr;
        T* device_data_parent_ptr = nullptr;
        std::shared_ptr<dev_ptr<T>> device_data_ptr = nullptr;

        std::shared_ptr<cudaStream_t> stream_ptr;

    protected:

        /* ############################### parent container logic ###################################
         *
         *         In case array is an element of a parent container (vector_of_array), it must
         *         update the parent's container dirty_host elements
         *
         *         TODO:
         *             - possibly find a cleaner solution for this
         *             - should be protected and not public
         */

        void set_parent_dirty_host() {

            if(parent_dirty_index_callback != nullptr)
                parent_dirty_index_callback(parent_index);
        }

        void set_parent_dirty_device() {

            if(parent_dirty_device_callback != nullptr)
                parent_dirty_device_callback();
        }

        int parent_index = 0;
        std::function<void(std::size_t)> parent_dirty_index_callback = nullptr;
        std::function<void(void)> parent_dirty_device_callback = nullptr;
    };
}

#endif //RAPTOR_ARRAY_HPP
