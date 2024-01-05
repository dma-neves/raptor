
cmake_minimum_required(VERSION 3.24)
project(lmarrow LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set (LMARROW_LIB lmarrow)

set(LMARROW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SUBDIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
foreach(SUBDIRECTORY ${SUBDIRECTORIES})
    if(IS_DIRECTORY ${SUBDIRECTORY})
        list(APPEND LMARROW_INCLUDE_DIRS ${SUBDIRECTORY})
    endif()
endforeach()
include_directories(${LMARROW_INCLUDE_DIRS})

add_subdirectory(src)

add_library(${LMARROW_LIB} SHARED ${LMARROW_SOURCES})

set_target_properties(lmarrow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(google_tests)
add_subdirectory(test)
add_subdirectory(examples)


############# To Parent scope #############

get_directory_property(hasParent PARENT_DIRECTORY)

if (hasParent)
    set (LMARROW_INCLUDE_DIRS ${LMARROW_INCLUDE_DIRS} PARENT_SCOPE)
    set (LMARROW_LIB ${LMARROW_LIB} PARENT_SCOPE)
endif()