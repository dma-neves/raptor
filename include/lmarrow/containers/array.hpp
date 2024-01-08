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

    public:

        array() {

            dev_alloc = true;
        }

        ~array() {

            device_shared_ptr = nullptr;
        }

        void free(cudaStream_t stream = 0) {

            if(device_shared_ptr != nullptr) {

                if(stream != 0)
                    device_shared_ptr.get()->freeAsync(stream);
                device_shared_ptr = nullptr;
            }
        }

        void fill(T &val) {

            std::fill(arr.begin(), arr.end(), val);
            dirty();
        }

        void fill(T &&val) {

            std::fill(arr.begin(), arr.end(), val);
            dirty();
        }

        void fill_on_device(T &val) {

            upload();
            fill_on_device(value_filler(val));
            dirty_on_device();
        }

        void fill_on_device(T &&val) {

            if(dev_alloc)
                allocate_device();

            fill_on_device(value_filler(val));
            dirty_on_device();
        }

        template<typename Functor>
        void fill(Functor fun) {

            for (int i = 0; i < N; i++)
                arr[i] = fun(i);

            dirty();
        }

        template<typename Functor>
        void fill_on_device(Functor fun) {

            if(dev_alloc)
                allocate_device();

            dev_fill<<<def_nb(N), def_tpb(N)>>>(get_device_ptr(), N, fun);
            dirty_on_device();
        }

        T &operator[](std::size_t i) {

            download();
            return arr[i];
        }

        T &get(std::size_t i) {

            download();
            return arr[i];
        }

        void set(std::size_t i, T &&val) {

            download();
            arr[i] = val;
            dirty();
        }

        void set(std::size_t i, T &val) {

            download();
            arr[i] = val;
            dirty();
        }

        bool contains(T&& val) {

            download();
            return std::find(arr.begin(), arr.end(), val) != arr.end();
        }

        bool contains(T& val) {

            download();
            return std::find(arr.begin(), arr.end(), val) != arr.end();
        }

        std::size_t size() { return N; }

        void copy(collection<T>& col) {

            this->download();
            col.download();
            memcpy(arr.data(), col.get_data(), sizeof(T) * N);
            dirty();
        }

        void copy_on_device(collection<T>& col) {

            this->upload();
            col.upload();

            cudaMemcpy(get_device_ptr(), col.get_device_ptr(), sizeof(T) * N, cudaMemcpyDeviceToDevice);
            dirty_on_device();
        }

        void dirty() {

            host_dirty = true;
            set_parent_dirty();
        }

        void dirty_on_device() {

            dev_dirty = true;
        }

    //protected:
        T *get_device_ptr() {

            return device_shared_ptr.get()->get();
        }

        T *get_data() {

            if(dev_dirty)
                download();
            
            return arr.data();
        }

        void upload(cudaStream_t stream = 0) {

            // Ensure dev allocation whenever upload is called
            if (dev_alloc) {
                allocate_device();
            }

            if (host_dirty) {
                cudaMemcpyAsync(get_device_ptr(), arr.data(), N * sizeof(T), cudaMemcpyHostToDevice, stream);
                host_dirty = false;
            }
        }

        void download(cudaStream_t stream = 0) {

            if(dev_dirty) {

                if (dev_alloc) {
                    // Shouldn't happen
                }
                else {
                    cudaMemcpyAsync(arr.data(), get_device_ptr(), N * sizeof(T), cudaMemcpyDeviceToHost, stream);
                }
                dev_dirty = false;
            }
        }

    private:

        void allocate_device() {

            device_shared_ptr = std::make_shared<dev_ptr<T>>( N * sizeof(T) );
            dev_alloc = false;
        }

        bool dev_alloc = true;
        bool host_dirty = false;
        bool dev_dirty = false;

        std::array<T, N> arr;
        std::shared_ptr<dev_ptr<T>> device_shared_ptr = nullptr;


    public:

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

            return;
            if(parent_dirty_index_callback != nullptr)
                parent_dirty_index_callback(parent_index);
        }

        int parent_index = 0;
        std::function<void(std::size_t)> parent_dirty_index_callback = nullptr;
    };
}

#endif //GDGRAPH_ARRAY_HPP
