//
// Created by david on 13-10-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/lmarrow.hpp"

using namespace lmarrow;

TEST(GArray, InitFillHost) {

    counting_sequence_filler<int> counting_sequence;

    array<int, 1024> arr;

    arr.fill(counting_sequence);

    for(int i = 0 ; i < 10; i++)
        ASSERT_EQ(arr[i], counting_sequence(i));
}

TEST(GArray, InitFillDevice) {

    counting_sequence_filler<int> counting_sequence;

    array<int, 1024> arr;

    arr.fill_on_device(counting_sequence);

    for(int i = 0 ; i < 10; i++)
        ASSERT_EQ(arr[i], counting_sequence(i));
}