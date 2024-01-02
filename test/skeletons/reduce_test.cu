//
// Created by david on 15-10-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/containers/vector.hpp"
#include "lmarrow/skeletons/reduce.hpp"

using namespace lmarrow;

struct FillFunctor6 {

    __device__ __host__
    int operator()(std::size_t i) {
        return (int)(i+1);
    }
};


TEST(Reduce, ReducePlus) {

    FillFunctor6 fill_fun6;

    constexpr int size = 1024;
    vector<int> vec(size);
    vec.fill_on_device(fill_fun6);

    int reduce_result = lmarrow::reduce<sum<int>>(vec);
    int expected_result = size * (size + 1) / 2;
    ASSERT_EQ(reduce_result, expected_result);
}