//
// Created by david on 11-12-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/containers/vector.hpp"

using namespace lmarrow;

TEST(DevPointer, DevPointerDestruct) {

    int* dev_ptr = nullptr;
    {
        vector<int> vec(1024);
        vec.fill_on_device(2);
        dev_ptr = vec.get_device_ptr();
    }

    cudaPointerAttributes attributes;
    auto result = cudaPointerGetAttributes(&attributes, dev_ptr);
    ASSERT_EQ(attributes.devicePointer, nullptr);
}

TEST(DevPointer, DevPointerCopyConstructor) {

    int* dev_ptr = nullptr;
    vector<int> vec_copy;
    {
        vector<int> vec(1024);
        vec.fill_on_device(2);
        dev_ptr = vec.get_device_ptr();

        vec_copy = vec;
    }

    cudaPointerAttributes attributes;
    auto result = cudaPointerGetAttributes(&attributes, dev_ptr);
    ASSERT_NE(attributes.devicePointer, nullptr);
}