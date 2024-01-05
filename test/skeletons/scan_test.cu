//
// Created by david on 15-10-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/lmarrow.hpp"

using namespace lmarrow;

TEST(Scan, PlusScan) {

    constexpr int size = 1024;
    vector<int> vec(size);
    vec.fill_on_device(counting_sequence_filler<int>());

    vector<int> scan_result = lmarrow::scan<sum<int>>(vec);

    int sum = 0;
    for(int i = 0; i < size; i++) {

        sum += vec[i];

        ASSERT_EQ(scan_result[i], sum);
    }
}

TEST(Scan, PlusScanSingleElement) {

    constexpr int size = 10;
    vector<int> vec(size);
    vec.fill_on_device(counting_sequence_filler<int>());

    vector<int> scan_result = lmarrow::scan<sum<int>>(vec);


    int sum = 0;
    for(int i = 0; i < size; i++) {

        sum += vec[i];

        ASSERT_EQ(scan_result[i], sum);
    }
}

TEST(Scan, MultScan) {

    constexpr int size = 1024;
    vector<int> vec(size);
    vec.fill_on_device(counting_sequence_filler<int>());

    vector<int> scan_result = lmarrow::scan<mult<int>>(vec);

    int mul = 1;
    for(int i = 0; i < size; i++) {

        mul *= vec[i];

        ASSERT_EQ(scan_result[i], mul);
    }
}