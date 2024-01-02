//
// Created by david on 16-10-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/containers/vector.hpp"

#define N 100000000

using namespace lmarrow;

struct FillFunctor7 {

    __device__ __host__
    int operator()(std::size_t i) {
        return (int)(i+1);
    }
};

FillFunctor7 fill_fun7;

TEST(Vector, FillOnDevice) {

    constexpr int size = 1024;

    vector<int> vec(size);
    vec.fill_on_device(fill_fun7);

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(vec[i], i+1);
    }
}

TEST(Vector, FillOnDeviceVal) {

    constexpr int size = 1024;

    vector<int> vec(size);
    vec.fill_on_device(42);

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(vec[i], 42);
    }
}