//
// Created by david on 31-10-2023.
//

#ifndef RAPTOR_DEV_PTR_HPP
#define RAPTOR_DEV_PTR_HPP

namespace raptor {

    template<typename T>
    class dev_ptr {

    public:
        dev_ptr(T* dptr = nullptr) : dptr(dptr) {

        }

        dev_ptr(std::size_t size) {

            allocate(size);
        }

        ~dev_ptr() {

            free();
        }

        T* get() {
            return dptr;
        }

        void allocate(std::size_t size) {

            cudaMalloc((void**) &dptr, size);
        }

        void free() {

            if(dptr != nullptr && !async_free) {
                cudaFree(dptr);
            }
        }

        void freeAsync(cudaStream_t& stream) {

            if(dptr != nullptr && !async_free) {
                async_free = true;
                cudaFreeAsync(dptr, stream);
            }
        }

    private:
        T* dptr = nullptr;
        bool async_free = false;
    };
}

#endif //RAPTOR_DEV_PTR_HPP
