//
// Created by david on 13-10-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/containers/vector.hpp"
#include "lmarrow/containers/array.hpp"
#include "lmarrow/function.hpp"

#define N 512
#define TPB 512

using namespace lmarrow;

struct FillFunctor3 {

    __device__ __host__
    float operator()(std::size_t i, std::size_t j) {
        return (int)(i * 1000 + j);
    }
};

FillFunctor3 fill_fun3;


TEST(GVectorOfGarray, Init) {

    vector<array<int, N>> vec(1024);

    vec.fill_on_device(fill_fun3);

    array<int, N>& a = vec[0];
    int& b = a[0];

    for(int i = 0; i < 10; i++) {

        for(int j = 0; j < 10; j++)
            ASSERT_EQ(vec[i][j], fill_fun3(i, j));
    }
}

struct check_values_fun : function_with_coordinates<check_values_fun> {

    __device__
    void operator()(coordinates_t index, int *values, int *results, std::size_t gvec_size, std::size_t garr_size, FillFunctor3& fun) {

        int flat_size = gvec_size * garr_size;

        if (index < flat_size) {

            int gvec_index = index / garr_size;
            int garr_index = index % garr_size;

            if (gvec_index == 2 && garr_index == 3) {

                results[index] = values[index] == 123;
            } else if (gvec_index == 7 && garr_index == 1) {

                results[index] = values[index] == 321;
            } else {
                results[index] = values[index] == fun(gvec_index, garr_index);
            }
        }
    }
};

TEST(GVectorOfGarray, InitAndUpdate) {

    vector<array<int, N>> vec(1024);

    vec.fill_on_device(fill_fun3);


    vec[2].set(3,123);
    vec[7].set(1, 321);


    constexpr std::size_t results_size = N*1024;
    array<int, results_size> results;

    check_values_fun cv;
    std::size_t gvec_size = 1024;
    std::size_t garr_size = 1024;
    cv.apply(results_size, vec, results, gvec_size, garr_size, fill_fun3);
    results.dirty();

    for(int i = 0; i < results_size; i++) {

        ASSERT_EQ(results[i], 1);
    }
}