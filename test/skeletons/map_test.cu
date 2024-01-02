//
// Created by david on 15-10-2023.
//

#include "lmarrow/skeletons/map.hpp"
#include "lmarrow/containers/vector.hpp"

#include <gtest/gtest.h>

using namespace lmarrow;

struct FillFunctor5 {

    __device__
    int operator()(std::size_t i) {
        return (int)i;
    }
};


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

    FillFunctor5 fill_fun5;

    constexpr int size = 1024;
    vector<int> vec(size);
    vec.fill_on_device(fill_fun5);

    vector<int> map_result = lmarrow::map<DoubleFun>(vec);

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(map_result[i], vec[i]*2);
    }
}

TEST(Map, MapSum) {

    FillFunctor5 fill_fun5;
    SumFun sum_fun;

    constexpr int size = 1024;
    vector<int> a(size);
    a.fill_on_device(fill_fun5);

    vector<int> b(size);
    b.fill_on_device(fill_fun5);

    vector<int> map_result = lmarrow::map<SumFun>(a, b);

    for(int i = 0; i < size; i++) {

        ASSERT_EQ(map_result[i], a[i]+b[i]);
    }
}