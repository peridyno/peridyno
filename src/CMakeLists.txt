cmake_minimum_required(VERSION 3.10)

set(PERIDYNO_INCLUDE_PATH ${CMAKE_BINARY_DIR}/include CACHE STRING "include path in binary directory" FORCE)
set(PERIDYNO_LIB_CMAKES "")
set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES "")

add_subdirectory(Core)
add_subdirectory(Framework)
add_subdirectory(IO)
add_subdirectory(Rendering)

add_subdirectory(Dynamics)
add_subdirectory(GUI)

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
        DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
        )