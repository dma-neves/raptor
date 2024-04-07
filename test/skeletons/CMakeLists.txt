include_directories(${RAPTOR_INCLUDE_DIRS})

set (RAPTOR_SKELTONS_TEST_SOURCES
        scan_test.cu
        map_test.cu
        reduce_test.cu
        # sort_unique_test.cu TODO: Fix unique
        )
add_executable(raptor_skeleton_tests ${RAPTOR_SKELTONS_TEST_SOURCES})

#add_test(NAME raptor_tests COMMAND raptor_tests)

set (GTEST_BOTH_LIBRARIES gtest_main gtest)

target_link_libraries(raptor_skeleton_tests
        #PUBLIC
        ${GTEST_BOTH_LIBRARIES} # why does gtest have to be the first param?
        ${RAPTOR_LIB}
        )