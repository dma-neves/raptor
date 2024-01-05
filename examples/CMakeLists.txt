add_subdirectory(mandelbrot)

include_directories(${LMARROW_INCLUDE_DIRS})

add_executable(saxpy saxpy.cu)
add_executable(montecarlo montecarlo.cu)
add_executable(riemann_sum riemann_sum.cu)

target_link_libraries(saxpy
        #PUBLIC
        ${LMARROW_LIB}
)

target_link_libraries(montecarlo
        #PUBLIC
        ${LMARROW_LIB}
        )

target_link_libraries(riemann_sum
        #PUBLIC
        ${LMARROW_LIB}
        )
