add_subdirectory(mandelbrot)

add_executable(saxpy saxpy.cu)
add_executable(montecarlo montecarlo.cu)

target_link_libraries(saxpy
        #PUBLIC
        ${LMARROW_LIB}
)

target_link_libraries(montecarlo
        #PUBLIC
        ${LMARROW_LIB}
        )

