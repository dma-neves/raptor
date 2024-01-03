include_directories(${LMARROW_INCLUDE_DIRS})

set (LMARROW_SKELTONS_TEST_SOURCES
        scan_test.cu
        map_test.cu
        reduce_test.cu
        sort_unique_test.cu
        )
add_executable(lmarrow_skeleton_tests ${LMARROW_SKELTONS_TEST_SOURCES})

#add_test(NAME lmarrow_tests COMMAND lmarrow_tests)

set (GTEST_BOTH_LIBRARIES gtest_main gtest)

target_link_libraries(lmarrow_skeleton_tests
        #PUBLIC
        ${GTEST_BOTH_LIBRARIES} # why does gtest have to be the first param?
        ${LMARROW_LIB}
        )