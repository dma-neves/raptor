//
// Created by david on 17-10-2023.
//

#ifndef RAPTOR_SCALAR_HPP
#define RAPTOR_SCALAR_HPP

namespace raptor {

    template <typename T>
    class scalar {

    public:

        scalar() {
        }

        scalar(T val) {

            this-> val = val;
        }

        ~scalar() {

            device_data_ptr = nullptr;
        }

        void set(T&& data) {
            host_dirty = true;
            this->val = data;
        }

        T get() {

            if(dev_dirty)
                download();

            return val;
        }

    //protected:

        void dirty_host() {
            host_dirty = true;
        }

        void dirty_device() {
            dev_dirty = true;
        }

        void upload(cudaStream_t stream = 0) {

            if(device_data_ptr == nullptr)
                device_data_ptr = std::make_shared<dev_ptr<T>>(sizeof(T) );

            cudaMemcpyAsync(device_data_ptr.get()->get(), &val, sizeof(T), cudaMemcpyHostToDevice, stream);

            dev_dirty = false;
        }

        void download(cudaStream_t stream = 0) {

            cudaMemcpyAsync(&val, device_data_ptr.get()->get(), sizeof(T), cudaMemcpyDeviceToHost, stream);
            host_dirty = false;
        }

        T* get_device_data() {

            return device_data_ptr == nullptr ? nullptr : device_data_ptr.get()->get();
        }

    private:
        T val;
        std::shared_ptr<dev_ptr<T>> device_data_ptr = nullptr;

        bool host_dirty = false;
        bool dev_dirty = false;
    };
}

#endif //RAPTOR_SCALAR_HPP
