cmake_minimum_required(VERSION 3.10)                                                            #指定CMake的最低版本为3.10。若版本过低，请自行升级CMake

project(peridyno LANGUAGES CXX CUDA)                                                             #指定本项目的编译语言为C++、CUDA

# define library version (update: apparently you can also do it in project()!)
set(PERIDYNO_LIBRARY_VERSION_MAJOR 0 CACHE STRING "peridyno major version" FORCE)
set(PERIDYNO_LIBRARY_VERSION_MINOR 6 CACHE STRING "peridyno minor version" FORCE)
set(PERIDYNO_LIBRARY_VERSION_PATCH 1 CACHE STRING "patch version of peridyno" FORCE)
set(PERIDYNO_LIBRARY_VERSION "${PERIDYNO_LIBRARY_VERSION_MAJOR}.${PERIDYNO_LIBRARY_VERSION_MINOR}.${PERIDYNO_LIBRARY_VERSION_PATCH}" CACHE STRING "Peridyno version" FORCE)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)                                                    #为sln内部projects设置管理folder
set(CMAKE_CXX_STANDARD 11 CACHE STRING "CXX STANDARD VERSION 11,14,17")                         #默认为本项目下各编译目标指定C++11语言特性


set(CMAKE_POSITION_INDEPENDENT_CODE ON)                                                         #默认为本项目下各编译目标开启fPIC模式，cuda下面会开启rdc模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "DEFAULT SET SEPERABLE COMPILATION MODE")    #默认为本项目下各编译目标指定cuda分离编译模式
set(CUDA_ARCH_FLAGS 60 CACHE STRING "cuda architectures" FORCE)                                 #默认为所有cuda代码生成computability7.0下二进制代码

set(PERIDYNO_ASSET_PATH "${CMAKE_SOURCE_DIR}/data" CACHE PATH "" FORCE)
file(COPY "data/" DESTINATION "data")

#set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})

IF(MSVC)
    SET( CMAKE_DEBUG_POSTFIX "d" )
ENDIF(MSVC)

# installation destinations
if(UNIX)
    include(GNUInstallDirs)

    set(PERIDYNO_INC_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/peridyno")
    set(PERIDYNO_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
    set(PERIDYNO_LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(PERIDYNO_ARCHIVE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(PERIDYNO_FRAMEWORK_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")

    set(PERIDYNO_CMAKE_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/peridyno/cmake")
    set(PERIDYNO_ADDITIONAL_FILES_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/peridyno")

    set(PERIDYNO_RUNTIME_INSTALL_DIR "bin") # for the nodesize_dbg, just ignore version and the like
    set(PERIDYNO_INC_INSTALL_DIR "include/peridyno") # header filestree

elseif(WIN32)
    set(PERIDYNO_INC_INSTALL_DIR "include/peridyno")
    set(PERIDYNO_RUNTIME_INSTALL_DIR   "bin")
    set(PERIDYNO_LIBRARY_INSTALL_DIR   "bin")
    set(PERIDYNO_ARCHIVE_INSTALL_DIR   "lib")
    set(PERIDYNO_FRAMEWORK_INSTALL_DIR "bin")

    set(PERIDYNO_CMAKE_CONFIG_INSTALL_DIR "share/peridyno/cmake")
    set(PERIDYNO_ADDITIONAL_FILES_INSTALL_DIR "share/peridyno")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
    message(FATAL_ERROR "Could not set install folders for this platform!")
endif()

option(PERIDYNO_LIBRARY_FRAMEWORK "Enable binding the framework library" ON)
option(PERIDYNO_LIBRARY_IO "Enable binding the io library" ON)
option(PERIDYNO_LIBRARY_RENDERING "Enable binding the rendering library" ON)
option(PERIDYNO_LIBRARY_ALEMBIC "Enable binding the alembic library" ON)
option(PERIDYNO_LIBRARY_GMSH "Enable binding the Gmsh library" OFF)

option(PERIDYNO_PYTHON_BINDING "Enable python binding with pybind11" OFF)
if(PERIDYNO_PYTHON_BINDING)
    add_subdirectory(python)
endif()

option(PERIDYNO_TESTS "Enable python binding with pybind11" OFF)
if(PERIDYNO_TESTS)
    add_subdirectory(external/gtest)
    add_subdirectory(tests)
endif()

add_subdirectory(external/glfw-3.3.0)
add_subdirectory(external/glad-4.6)
add_subdirectory(external/imgui)

if(PERIDYNO_LIBRARY_ALEMBIC)
    add_subdirectory(external/Imath)
    add_subdirectory(external/alembic/)
endif()

if(PERIDYNO_LIBRARY_GMSH)
    add_subdirectory(external/gmsh-fork)
endif()

# VTK rendering support, off by default
option(PERIDYNO_PLUGIN_VTK "Enable VTK plugin for visualization" OFF)
# Qt GUI support, off by default
option(PERIDYNO_Qt5_GUI "Enable building Qt-based applications" OFF)

if(PERIDYNO_Qt5_GUI)
    add_subdirectory(external/nodeeditor)
endif()


option(PERIDYNO_EXAMPLE "Enable building examples" ON)
if(PERIDYNO_EXAMPLE)
    add_subdirectory(examples)
endif()

add_subdirectory(src)


