include_directories(${LMARROW_INCLUDE_DIRS})

set (LMARROW_COLLECTIONS_TEST_SOURCES
        array_test.cu
        vector_of_array_test.cu
        vector_test.cu
        dev_pointer_test.cu
        )

add_executable(lmarrow_collections_tests ${LMARROW_COLLECTIONS_TEST_SOURCES})

set (GTEST_BOTH_LIBRARIES gtest_main gtest)

target_link_libraries(lmarrow_collections_tests
        #PUBLIC
        ${GTEST_BOTH_LIBRARIES} # why does gtest have to be the first param?
        ${LMARROW_LIB}
        )