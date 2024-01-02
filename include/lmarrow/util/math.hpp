//
// Created by david on 02-01-2024.
//

#ifndef LMARROW_MATH_HPP
#define LMARROW_MATH_HPP

#include <thrust/complex.h>

namespace lmarrow::math {

    template <typename T>
    class complex : public thrust::complex<T> {
    public:

        __device__ complex() : thrust::complex<T>() {}

        __device__ complex(T real, T imag) : thrust::complex<T>(real, imag) {}

        template<typename U = T>
        __device__ complex<T>& operator=(U&& c) {
            return *this;
        }

        __device__
        lmarrow::math::complex<float> operator*(lmarrow::math::complex<float> c)
        {
            //return thrust::complex<T>::operator*(c); not sure why this doesn't work
            lmarrow::math::complex<float> r(
                    this->real() * c.real() - this->imag() * c.imag(),
                    this->imag() * c.real() + this->real() * c.imag());
            return r;
        }

        __device__
        lmarrow::math::complex<float> operator+(lmarrow::math::complex<float> c)
        {
            //return thrust::complex<T>::operator+(c); not sure why this doesn't work
            lmarrow::math::complex<float> r(
                    this->real() + c.real(),
                    this->imag() + c.imag());
            return r;
        }
    };
}

#endif //LMARROW_MATH_HPP
