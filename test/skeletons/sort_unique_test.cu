//
// Created by david on 20-10-2023.
//

#include <gtest/gtest.h>

#include "lmarrow/skeletons/unique.hpp"
#include "lmarrow/skeletons/radix_sort.hpp"

using namespace lmarrow;

TEST(Unique, SmallVec) {

    vector<int> vec;

    vec.push_back(2);
    vec.push_back(3);
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(4);
    vec.push_back(3);
    vec.push_back(5);
    vec.push_back(6);
    vec.push_back(1);


    vector<int> sorted_vec = lmarrow::radix_sort(vec);
    sorted_vec[0];
    vector<int> unique_vec = lmarrow::unique(sorted_vec);

    ASSERT_EQ(unique_vec.size(), 6);
    ASSERT_TRUE(unique_vec.contains(1));
    ASSERT_TRUE(unique_vec.contains(2));
    ASSERT_TRUE(unique_vec.contains(3));
    ASSERT_TRUE(unique_vec.contains(4));
    ASSERT_TRUE(unique_vec.contains(5));
    ASSERT_TRUE(unique_vec.contains(6));

}