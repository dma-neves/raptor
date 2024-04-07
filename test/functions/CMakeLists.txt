include_directories(${RAPTOR_INCLUDE_DIRS})

set (RAPTOR_FUNCTIONS_TEST_SOURCES
        functions_test.cu
        )

add_executable(raptor_functoins_tests ${RAPTOR_FUNCTIONS_TEST_SOURCES})


set (GTEST_BOTH_LIBRARIES gtest_main gtest)

target_link_libraries(raptor_functoins_tests
        #PUBLIC
        ${GTEST_BOTH_LIBRARIES} # why does gtest have to be the first param?
        ${RAPTOR_LIB}
        )