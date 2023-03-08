# MCM cuda library
add_library (mcm_cuda STATIC
        combi.cu
        mcm.cu
        util.cu
        geodesic.cu
        libmcm_cuda.cu
        )

target_include_directories (mcm_cuda
        INTERFACE
            ${CMAKE_SOURCE_DIR}/include
        PRIVATE
            .
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
target_link_libraries(mcm_cuda
        LINK_PUBLIC
            m
            CUDA::cudart
            CUDA::npps
        )
set_target_properties(mcm_cuda
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES all
            POSITION_INDEPENDENT_CODE ON
        )


