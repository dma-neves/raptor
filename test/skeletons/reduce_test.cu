//
// Created by david on 15-10-2023.
//

#include <gtest/gtest.h>

#include "raptor.hpp"

using namespace raptor;


TEST(Reduce, ReducePlus) {

    constexpr int size = 1024;
    constexpr int last_number = size-1;
    vector<int> vec(size);
    vec.fill_on_device(iota_filler<int>());

    scalar<int> reduce_result = raptor::reduce<sum<int>>(vec);
    int expected_result = last_number * (last_number+1) / 2;
    ASSERT_EQ(reduce_result.get(), expected_result);
}