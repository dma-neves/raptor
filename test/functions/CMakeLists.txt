include_directories(${LMARROW_INCLUDE_DIRS})

set (LMARROW_FUNCTIONS_TEST_SOURCES
        functions_test.cu
        )

add_executable(lmarrow_functoins_tests ${LMARROW_FUNCTIONS_TEST_SOURCES})


set (GTEST_BOTH_LIBRARIES gtest_main gtest)

target_link_libraries(lmarrow_functoins_tests
        #PUBLIC
        ${GTEST_BOTH_LIBRARIES} # why does gtest have to be the first param?
        ${LMARROW_LIB}
        )