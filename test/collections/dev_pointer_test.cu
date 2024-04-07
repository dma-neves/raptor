//
// Created by david on 11-12-2023.
//

#include <gtest/gtest.h>

#include "raptor.hpp"

using namespace raptor;

TEST(DevPointerTest, DevPointerTestDestruct) {

    int* dev_ptr = nullptr;
    {
        vector<int> vec(1024);
        vec.fill_on_device(2);
        dev_ptr = vec.get_device_data();
    }

    cudaPointerAttributes attributes;
    auto result = cudaPointerGetAttributes(&attributes, dev_ptr);
    ASSERT_EQ(attributes.devicePointer, nullptr);
}

TEST(DevPointerTest, DevPointerTestCopyConstructor) {

    int* dev_ptr = nullptr;
    vector<int> vec_copy;
    {
        vector<int> vec(1024);
        vec.fill_on_device(2);
        dev_ptr = vec.get_device_data();

        vec_copy = vec;
    }

    cudaPointerAttributes attributes;
    auto result = cudaPointerGetAttributes(&attributes, dev_ptr);
    ASSERT_NE(attributes.devicePointer, nullptr);
}