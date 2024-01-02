//
// Created by david on 13-10-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/containers/array.hpp"

using namespace lmarrow;

struct FillFunctor2 {

    __device__ __host__
    int operator()(std::size_t i) {
        return (int)i;
    }
};

FillFunctor2 fill_fun2;

TEST(GArray, InitFillHost) {

    int n = 1024;
    array<int, 1024> arr;

    arr.fill(fill_fun2);

    for(int i = 0 ; i < 10; i++)
        ASSERT_EQ(arr[i], fill_fun2(i));
}

TEST(GArray, InitFillDevice) {

    array<int, 1024> arr;

    arr.fill_on_device(fill_fun2);

    for(int i = 0 ; i < 10; i++)
        ASSERT_EQ(arr[i], fill_fun2(i));
}