add_library(mcm
        STATIC
            combi.c
            mcm.c
            geodesic.c
            nrutil.c
            util.c
            emfile.c
            libmcm.c
        )
target_include_directories (mcm
        INTERFACE
            ${CMAKE_SOURCE_DIR}/include
        PRIVATE
            .
        )
target_link_libraries(mcm
        LINK_PUBLIC
            m
        )
set_property(
        TARGET
            mcm
        PROPERTY
            POSITION_INDEPENDENT_CODE ON)

