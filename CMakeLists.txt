cmake_minimum_required (VERSION 3.23)
project(MCM LANGUAGES C CUDA)

find_package(CUDAToolkit)

# Executables go to bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Libraries go to lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(apps)

if(SKBUILD)
    message(STATUS "The project is built using scikit-build.")
    find_package(PythonExtensions REQUIRED)
    find_package(Cython REQUIRED)
    find_package(NumPy REQUIRED)

    add_subdirectory(pymcm)
endif()



# MCM library
#add_library(libmcm STATIC
#        src/libc/combi.c
#        src/libc/mcm.c
#        src/libc/geodesic.c
#        src/libc/nrutil.c
#        src/libc/util.c
#        )
#target_include_directories (libmcm
#        PUBLIC
#            src/libc
#        )
#target_link_libraries(libmcm
#        LINK_PUBLIC
#            m
#        )

## MCM cuda library
#add_library (libmcm_cuda STATIC
#        src/libcuda/combi.cu
#        src/libcuda/mcm.cu
#        src/libcuda/util.cu
#        src/libcuda/geodesic.cu
#        )
#target_include_directories (libmcm_cuda
#        PUBLIC
#            src/libcuda
#        PRIVATE
#            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#        )
#set_target_properties(libmcm_cuda
#        PROPERTIES
#            CUDA_SEPARABLE_COMPILATION OFF
#            CUDA_ARCHITECTURES OFF
#        )
#target_link_libraries(libmcm_cuda
#        LINK_PUBLIC
#            CUDA::cudart
#            CUDA::npps
#        )
#
## Combi executable
#add_executable (mcm_levelset apps/mcm_levelset.c)
#target_include_directories (mcm_levelset
#        PUBLIC
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/libc
#        )
#target_link_libraries(mcm_levelset
#        LINK_PUBLIC
#            m
#            libmcm
#        )
#
## Combi executable CUDA
#add_executable (mcm_levelset_cuda apps/mcm_levelset.c)
#target_compile_definitions(mcm_levelset_cuda
#                PRIVATE
#                    USE_CUDA)
#target_include_directories (mcm_levelset_cuda
#        PUBLIC
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/libc
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/libcuda
#        )
#target_link_libraries(mcm_levelset_cuda
#        LINK_PUBLIC
#            m
#            libmcm
#            libmcm_cuda
#        )
#
## MCM executable
#add_executable (mcm apps/mcm.c)
#target_include_directories (mcm
#        PUBLIC
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/libc
#        )
#target_link_libraries(mcm
#        LINK_PUBLIC
#            m
#            libmcm
#        )
#
## MCM executable CUDA
#add_executable (mcm_cuda apps/mcm.c)
#target_compile_definitions(mcm_cuda
#        PRIVATE
#            USE_CUDA)
#target_include_directories (mcm_cuda
#        PUBLIC
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/libc
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/libcuda
#        )
#target_link_libraries(mcm_cuda
#        LINK_PUBLIC
#            m
#            libmcm
#            libmcm_cuda
#        )
#
## Geodesic Path executable
#add_executable (geodesic_path apps/geodesic.c)
#target_include_directories (geodesic_path
#        PUBLIC
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/libc
#        )
#target_link_libraries(geodesic_path
#        LINK_PUBLIC
#            m
#            libmcm
#        )
#
## Geodesic Path executable CUDA
#add_executable (geodesic_trace_cuda apps/geodesic.c)
#target_compile_definitions(geodesic_trace_cuda
#        PRIVATE
#        USE_CUDA)
#target_include_directories (geodesic_trace_cuda
#        PUBLIC
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/libc
#            ${CMAKE_CURRENT_SOURCE_DIR}/src/libcuda
#        )
#target_link_libraries(geodesic_trace_cuda
#        LINK_PUBLIC
#            m
#            libmcm
#            libmcm_cuda
#        )
