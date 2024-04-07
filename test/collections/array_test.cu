//
// Created by david on 13-10-2023.
//

#include <gtest/gtest.h>

#include "raptor.hpp"

using namespace raptor;

TEST(ArrayTest, InitFillHost) {

    iota_filler<int> iota;

    array<int, 1024> arr;

    arr.fill(iota);

    for(int i = 0 ; i < 10; i++)
        ASSERT_EQ(arr.get(i), iota(i));
}

TEST(ArrayTest, InitFillDevice) {

    iota_filler<int> iota;

    array<int, 1024> arr;

    arr.fill_on_device(iota);

    for(int i = 0 ; i < 10; i++)
        ASSERT_EQ(arr.get(i), iota(i));
}