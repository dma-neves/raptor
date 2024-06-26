//
// Created by david on 15-10-2023.
//

#include "raptor.hpp"

#include <gtest/gtest.h>

using namespace raptor;

struct DoubleFun {

    __device__
    int operator()(int input) {

        return input*2;
    }
};

struct SumFun {

    __device__
    int operator()(int a, int b) {

        return a+b;
    }
};

TEST(Map, MapDouble) {
    
    constexpr int size = 1024;
    vector<int> vec(size);
    vec.fill_on_device(iota_filler<int>());

    vector<int> map_result = raptor::map<DoubleFun>(vec);

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(map_result.get(i), vec.get(i)*2);
    }
}

TEST(Map, MapSum) {

    constexpr int size = 1024;
    vector<int> a(size);
    a.fill_on_device(iota_filler<int>());

    vector<int> b(size);
    b.fill_on_device(iota_filler<int>());

    vector<int> map_result = raptor::map<SumFun>(a, b);

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(map_result.get(i), a.get(i)+b.get(i));
    }
}