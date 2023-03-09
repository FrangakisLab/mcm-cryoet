# Combi executable
add_executable (mcm_levelset
        mcm_levelset.c)

target_include_directories (mcm_levelset
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/mcm
        )

target_link_libraries(mcm_levelset
        LINK_PUBLIC
            mcm
        )

# MCM executable
add_executable (mcm_3D
        mcm.c)

target_include_directories (mcm_3D
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/mcm
        )

target_link_libraries(mcm_3D
        LINK_PUBLIC
            mcm
        )

# Geodesic path executable
add_executable (geodesic_trace
        geodesic.c)

target_include_directories (geodesic_trace
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/mcm
        )

target_link_libraries(geodesic_trace
        LINK_PUBLIC
        mcm
        )

#if(SKBUILD)
#    install(TARGETS mcm_levelset RUNTIME DESTINATION bin/)
#    install(TARGETS mcm_3D RUNTIME DESTINATION bin/)
#    install(TARGETS geodesic_trace RUNTIME DESTINATION bin/)
#endif()



if(CUDAToolkit_FOUND)
    # Combi executable CUDA
    add_executable (mcm_levelset_cuda
            mcm_levelset.c)

    target_compile_definitions(mcm_levelset_cuda
                    PRIVATE
                        USE_CUDA)

    target_include_directories (mcm_levelset_cuda
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
            PRIVATE
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                ${CMAKE_SOURCE_DIR}/src/mcm
                ${CMAKE_SOURCE_DIR}/src/mcm_cuda
            )

    target_link_libraries(mcm_levelset_cuda
            LINK_PUBLIC
                mcm
                mcm_cuda
            )

    # MCM executable CUDA
    add_executable (mcm_3D_cuda
            mcm.c)

    target_compile_definitions(mcm_3D_cuda
            PRIVATE
                USE_CUDA)

    target_include_directories (mcm_3D_cuda
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
            PRIVATE
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                ${CMAKE_SOURCE_DIR}/src/mcm
                ${CMAKE_SOURCE_DIR}/src/mcm_cuda
            )

    target_link_libraries(mcm_3D_cuda
            LINK_PUBLIC
                mcm
                mcm_cuda
            )

    # Geodesic executable CUDA
    add_executable (geodesic_trace_cuda
            geodesic.c)

    target_compile_definitions(geodesic_trace_cuda
            PRIVATE
                USE_CUDA)

    target_include_directories (geodesic_trace_cuda
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
            PRIVATE
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                ${CMAKE_SOURCE_DIR}/src/mcm
                ${CMAKE_SOURCE_DIR}/src/mcm_cuda
            )

    target_link_libraries(geodesic_trace_cuda
            LINK_PUBLIC
                mcm
                mcm_cuda
            )

#    if(SKBUILD)
#        install(TARGETS mcm_levelset_cuda RUNTIME DESTINATION bin/)
#        install(TARGETS mcm_3D_cuda RUNTIME DESTINATION bin/)
#        install(TARGETS geodesic_trace_cuda RUNTIME DESTINATION bin/)
#    endif()
endif()