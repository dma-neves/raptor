include_directories(${RAPTOR_INCLUDE_DIRS})

find_package(SDL2 REQUIRED)

add_executable(mandelbrot mandelbrot.cu)

target_link_libraries(mandelbrot
        ${RAPTOR_LIB}
        ${SDL2_LIBRARIES}   # Link SDL2 libraries
        )

target_include_directories(mandelbrot
        PRIVATE ${SDL2_INCLUDE_DIRS}   # Include SDL2 headers
        )

target_link_libraries(mandelbrot
        #PUBLIC
        ${RAPTOR_LIB}
        )