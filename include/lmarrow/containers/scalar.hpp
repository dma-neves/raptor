//
// Created by david on 17-10-2023.
//

#ifndef GDGRAPH_SCALAR_HPP
#define GDGRAPH_SCALAR_HPP

namespace lmarrow {

    template <typename T>
    class scalar {

    public:

        scalar() {
        }

        scalar(T val) {

            this-> val = val;
        }

        ~scalar() {

            shared_device_ptr = nullptr;
        }

        void set_data(T&& data) {
            host_dirty = true;
            this->val = data;
        }

        T get_data() {

            if(dev_dirty)
                download();

            return val;
        }

    //protected:

        void upload(cudaStream_t stream = 0) {

            if(shared_device_ptr == nullptr)
                shared_device_ptr = std::make_shared<dev_ptr<T>>(sizeof(T) );

            cudaMemcpyAsync(shared_device_ptr.get()->get(), &val, sizeof(T), cudaMemcpyHostToDevice, stream);

            dev_dirty = false;
        }

        void download(cudaStream_t stream = 0) {

            cudaMemcpyAsync(&val, shared_device_ptr.get()->get(), sizeof(T), cudaMemcpyDeviceToHost, stream);
            host_dirty = false;
        }

        T* get_device_ptr() {

            return shared_device_ptr == nullptr ? nullptr : shared_device_ptr.get()->get();
        }

    private:
        T val;
        std::shared_ptr<dev_ptr<T>> shared_device_ptr = nullptr;

        bool host_dirty = false;
        bool dev_dirty = false;
    };
}

#endif //GDGRAPH_SCALAR_HPP
