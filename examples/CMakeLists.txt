set (SAXPY_SOURCES saxpy.cu)
add_executable(saxpy ${SAXPY_SOURCES})

target_link_libraries(saxpy
        #PUBLIC
        ${LMARROW_LIB}
)