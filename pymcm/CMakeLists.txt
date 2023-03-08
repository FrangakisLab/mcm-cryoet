if(SKBUILD)
    add_cython_target(_pymcm _pymcm.pyx OUTPUT_VAR X)
    add_library(_pymcm MODULE ${X})

    target_include_directories(_pymcm
            PRIVATE
                ${CMAKE_SOURCE_DIR}/include
                ${NumPy_INCLUDE_DIRS}
                ${PYTHON_INCLUDE_DIRS}
            )

    target_link_libraries(_pymcm
            PUBLIC
            mcm)

    set_target_properties(_pymcm
            PROPERTIES
            PREFIX "")

    install(TARGETS _pymcm LIBRARY DESTINATION pymcm/)

    if(CUDAToolkit_FOUND)
        add_cython_target(_pymcm_cuda _pymcm_cuda.pyx OUTPUT_VAR Y)
        add_library(_pymcm_cuda MODULE ${Y})

        target_include_directories(_pymcm_cuda
                PRIVATE
                    ${CMAKE_SOURCE_DIR}/include
                    ${NumPy_INCLUDE_DIRS}
                    ${PYTHON_INCLUDE_DIRS}
                )

        target_link_libraries(_pymcm_cuda
                PUBLIC
                mcm
                mcm_cuda)

        set_target_properties(_pymcm_cuda
                PROPERTIES
                PREFIX "")

        install(TARGETS _pymcm_cuda LIBRARY DESTINATION pymcm/)
    endif()
endif()