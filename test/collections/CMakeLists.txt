include_directories(${RAPTOR_INCLUDE_DIRS})

set (RAPTOR_COLLECTIONS_TEST_SOURCES
        array_test.cu
        vector_of_array_test.cu
        vector_test.cu
        dev_pointer_test.cu
        vector_of_array_block_size_performance_test.cu
        )

add_executable(raptor_collections_tests ${RAPTOR_COLLECTIONS_TEST_SOURCES})

set (GTEST_BOTH_LIBRARIES gtest_main gtest)

target_link_libraries(raptor_collections_tests
        #PUBLIC
        ${GTEST_BOTH_LIBRARIES} # why does gtest have to be the first param?
        ${RAPTOR_LIB}
        )