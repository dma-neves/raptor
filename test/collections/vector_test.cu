//
// Created by david on 16-10-2023.
//

#include <gtest/gtest.h>

#include "raptor.hpp"

#define N 100000000

using namespace raptor;

TEST(VectorTest, FillOnDevice) {

    constexpr int size = 1024;

    vector<int> vec(size);
    vec.fill<DEVICE>(iota_filler<int>());

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(vec.get(i), i);
    }
}

TEST(VectorTest, FillOnDeviceVal) {

    constexpr int size = 1024;

    vector<int> vec(size);
    vec.fill<DEVICE>(42);

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(vec.get(i), 42);
    }
}

TEST(VectorTest, CopyOnDevice) {

    constexpr int size = 1024;

    vector<int> vec(size);
    vec.fill<DEVICE>(iota_filler<int>());

    vector<int> replica(size);
    replica.copy<DEVICE>(vec);

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(vec.get(i), replica.get(i));
    }
}

TEST(VectorTest, CopyOnHost) {

    constexpr int size = 1024;

    vector<int> vec(size);
    vec.fill<HOST>(iota_filler<int>());

    vector<int> replica(size);
    replica.copy<HOST>(vec);

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(vec.get(i), replica.get(i));
    }
}