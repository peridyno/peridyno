cmake_minimum_required(VERSION 3.10)

set(PERIDYNO_INCLUDE_PATH ${CMAKE_BINARY_DIR}/include CACHE STRING "include path in binary directory" FORCE)
set(PERIDYNO_LIB_CMAKES "")
set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES "")

add_subdirectory(Core)
add_subdirectory(Framework)

add_subdirectory(Topology)

add_subdirectory(Rendering)

option(PERIDYNO_LIBRARY_IO "Enable binding the io library" ON)
if(PERIDYNO_LIBRARY_IO)
    add_subdirectory(IO)
endif()

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    option(PERIDYNO_LIBRARY_MODELING "Enable binding the io library" ON)
    option(PERIDYNO_LIBRARY_INTERACTION "Enable binding the io library" ON)

    if(PERIDYNO_LIBRARY_MODELING)
        add_subdirectory(Interaction)
    endif()
    
    if(PERIDYNO_LIBRARY_MODELING)
        add_subdirectory(Modeling)
    endif()
endif()

add_subdirectory(Dynamics)

# Write/install version file
include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/PeridynoConfigVersion.cmake")
write_basic_package_version_file(${version_file}
    VERSION ${PERIDYNO_LIBRARY_VERSION}
    COMPATIBILITY AnyNewerVersion)

get_property(PERIDYNO_LIB_CMAKES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")

configure_file(PeridynoConfig.cmake.in PeridynoConfig.cmake @ONLY)
install(FILES   "${CMAKE_CURRENT_BINARY_DIR}/PeridynoConfig.cmake"
                ${version_file}
        DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR})