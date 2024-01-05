//
// Created by david on 16-10-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/lmarrow.hpp"

#define N 100000000

using namespace lmarrow;

TEST(Vector, FillOnDevice) {

    constexpr int size = 1024;

    vector<int> vec(size);
    vec.fill_on_device(counting_sequence_filler<int>());

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(vec[i], i);
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