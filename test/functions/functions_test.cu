//
// Created by david on 27-11-2023.
//

#include "lmarrow/lmarrow.hpp"

#include <gtest/gtest.h>

using namespace lmarrow;

struct saxpy_fun : function<saxpy_fun, in<float>, in<float*>, inout<float*>> {

    __device__
    void operator()(coordinates_t tid, float a, float* x, float* y) {

        y[tid] = a * x[tid] + y[tid];
    }
};

TEST(FunctionTest, SaxpyWithCoordinates) {

    int n = 10;

    float a = 2.0f;
    vector<float> x(n);
    vector<float> y(n);

    x.fill(4.0f);
    y.fill(3.0f);

    saxpy_fun saxpy;
    // saxpy.set_size(n); // optional
    saxpy.apply(a, x, y);

    for(int i = 0; i < n; i++) {
        ASSERT_EQ(y[i], 2.0*4.0+3.0);
    }
}
