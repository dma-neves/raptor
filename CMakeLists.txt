
cmake_minimum_required(VERSION 3.24)
project(raptor LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set (RAPTOR_LIB raptor)

set(RAPTOR_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SUBDIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
foreach(SUBDIRECTORY ${SUBDIRECTORIES})
    if(IS_DIRECTORY ${SUBDIRECTORY})
        list(APPEND RAPTOR_INCLUDE_DIRS ${SUBDIRECTORY})
    endif()
endforeach()
include_directories(${RAPTOR_INCLUDE_DIRS})

add_subdirectory(src)

add_library(${RAPTOR_LIB} SHARED ${RAPTOR_SOURCES} include/raptor/util/generators.h)

set_target_properties(raptor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(external/googletest)

#add_subdirectory(test)
#add_subdirectory(examples)


############# To Parent scope #############

get_directory_property(hasParent PARENT_DIRECTORY)

if (hasParent)
    set (RAPTOR_INCLUDE_DIRS ${RAPTOR_INCLUDE_DIRS} PARENT_SCOPE)
    set (RAPTOR_LIB ${RAPTOR_LIB} PARENT_SCOPE)
else()
    add_subdirectory(test)
    add_subdirectory(examples)
endif()