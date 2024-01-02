
cmake_minimum_required(VERSION 3.24)
project(lmarrow LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set (LMARROW_LIB lmarrow)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SUBDIRECTORIES ${CMAKE_SOURCE_DIR}/include/*)
foreach(SUBDIRECTORY ${SUBDIRECTORIES})
    if(IS_DIRECTORY ${SUBDIRECTORY})
        include_directories(${SUBDIRECTORY})
    endif()
endforeach()

add_subdirectory(src)

add_library(${LMARROW_LIB} SHARED ${LMARROW_SOURCES})

set_target_properties(lmarrow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(google_tests)
add_subdirectory(test)
add_subdirectory(examples)