option(PERIDYNO_PLUGIN_GMSH "Enable binding the Gmsh library" OFF)

if(PERIDYNO_PLUGIN_GMSH)
    if(WIN32)
        set(LIB_DEPENDENCY Core Framework GmshLib Topology winmm wsock32 ws2_32 psapi)
    elseif(UNIX)
        set(LIB_DEPENDENCY Core Framework GmshLib Topology)
    endif()

    include_directories("${CMAKE_SOURCE_DIR}/external/gmsh-fork/api")

    add_plugin(GmshIO LIB_DEPENDENCY)
endif()