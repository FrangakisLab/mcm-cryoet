add_subdirectory(mcm)

if(CUDAToolkit_FOUND)
    add_subdirectory(mcm_cuda)
endif()