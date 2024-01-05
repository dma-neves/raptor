//
// Created by david on 15-10-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/lmarrow.hpp"

using namespace lmarrow;


TEST(Reduce, ReducePlus) {

    constexpr int size = 1024;
    constexpr int last_number = size-1;
    vector<int> vec(size);
    vec.fill_on_device(counting_sequence_filler<int>());

    int reduce_result = lmarrow::reduce<sum<int>>(vec);
    int expected_result = last_number * (last_number+1) / 2;
    ASSERT_EQ(reduce_result, expected_result);
}