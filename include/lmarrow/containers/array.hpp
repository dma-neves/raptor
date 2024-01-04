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
            set_host_dirty();
        }

        void fill(T &&val) {

            std::fill(arr.begin(), arr.end(), val);
            set_host_dirty();
        }

        void fill_on_device(T &val) {

            if(dev_alloc)
                allocate_device();

            fill_on_device(fill_val_fun(val));
            dev_dirty = true;
        }

        void fill_on_device(T &&val) {

            if(dev_alloc)
                allocate_device();

            fill_on_device(fill_val_fun(val));
            dev_dirty = 1;
        }

        template<typename Functor>
        void fill(Functor fun) {

            for (int i = 0; i < N; i++)
                arr[i] = fun(i);

            set_host_dirty();
        }

        template<typename Functor>
        void fill_on_device(Functor fun) {

            if(dev_alloc)
                allocate_device();

            dev_fill<<<def_nb(N), def_tpb(N)>>>(get_device_ptr(), N, fun);
            dev_dirty = true;
        }

        T &operator[](std::size_t i) {

            if(dev_dirty)
                download();

            return arr[i];
        }

        T &get(std::size_t i) {

            if(dev_dirty)
                download();

            return arr[i];
        }

        void set(std::size_t i, T &&val) {

            arr[i] = val;
            set_host_dirty();
        }

        void set(std::size_t i, T &val) {

            arr[i] = val;
            set_host_dirty();
        }


        std::size_t size() { return N; }

        void copy(collection<T>& col) {

            memcpy(arr.data(), col.get_data(), sizeof(T) * N);
            set_host_dirty();
        }

        void copy_on_device(collection<T>& col) {

            if(dev_alloc)
                allocate_device();

            cudaMemcpy(get_device_ptr(), col.get_device_ptr(), sizeof(T) * N, cudaMemcpyDeviceToDevice);
            dev_dirty = 1;
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

        void dirty() {
            host_dirty = true;
        }

        void dirty_on_device() {
            dev_dirty = true;
        }

        void upload(cudaStream_t stream = 0) {

            if (dev_alloc) {
                allocate_device();
            }

            if (host_dirty)
                cudaMemcpyAsync(get_device_ptr(), arr.data(), N * sizeof(T), cudaMemcpyHostToDevice, stream);

            host_dirty = false;

        }

        void download(cudaStream_t stream = 0) {

            cudaMemcpyAsync(arr.data(), get_device_ptr(), N * sizeof(T), cudaMemcpyDeviceToHost, stream);
            dev_dirty = false;
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

        void set_host_dirty() {

            host_dirty = true;

            if(parent_dirty_index_callback != nullptr)
                parent_dirty_index_callback(parent_index);
        }

        int parent_index = 0;
        std::function<void(std::size_t)> parent_dirty_index_callback = nullptr;
    };
}

#endif //GDGRAPH_ARRAY_HPP
