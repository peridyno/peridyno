cmake_minimum_required(VERSION 3.10)                                                            #指定CMake的最低版本为3.10。若版本过低，请自行升级CMake

include(cmake/modules.cmake)

set(PERIDYNO_GPU_BACKEND "CUDA" CACHE STRING "GPU Backend")
set_property(CACHE PERIDYNO_GPU_BACKEND PROPERTY STRINGS "CUDA;Vulkan;NoGPU")

set(PERIDYNO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(SUPPORTED_LANGUAGES "CXX;C")
set(GPU_BACKEND "NO_BACKEND")
if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    set(SUPPORTED_LANGUAGES "CUDA;CXX;C")
endif()                                                           #指定本项目的编译语言为C++、CUDA

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
	set(GPU_BACKEND "CUDA_BACKEND")
elseif("${PERIDYNO_GPU_BACKEND}" STREQUAL "Vulkan")
	set(GPU_BACKEND "VK_BACKEND")
else()
	set(GPU_BACKEND "NO_BACKEND")
endif()

project(peridyno LANGUAGES ${SUPPORTED_LANGUAGES})

option(CMAKE_MSVC_PARALLEL_ENABLE "Enable /MP for msvc" ON)
add_compile_options("$<$<AND:$<BOOL:${CMAKE_MSVC_PARALLEL_ENABLE}>,$<OR:$<COMPILE_LANG_AND_ID:C,MSVC>,$<COMPILE_LANG_AND_ID:CXX,MSVC>>>:/MP>")

# define library version (update: apparently you can also do it in project()!)
set(PERIDYNO_LIBRARY_VERSION_MAJOR 0 CACHE STRING "peridyno major version" FORCE)
set(PERIDYNO_LIBRARY_VERSION_MINOR 9 CACHE STRING "peridyno minor version" FORCE)
set(PERIDYNO_LIBRARY_VERSION_PATCH 1 CACHE STRING "patch version of peridyno" FORCE)
set(PERIDYNO_LIBRARY_VERSION "${PERIDYNO_LIBRARY_VERSION_MAJOR}.${PERIDYNO_LIBRARY_VERSION_MINOR}.${PERIDYNO_LIBRARY_VERSION_PATCH}" CACHE STRING "Peridyno version" FORCE)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)                                                    #为sln内部projects设置管理folder
set(CMAKE_CXX_STANDARD 17 CACHE STRING "CXX STANDARD VERSION 11,14,17")                         #默认为本项目下各编译目标指定C++11语言特性
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CMAKE_C_CREATE_STATIC_LIBRARY "")
#set(CMAKE_C_COMPILE_OBJECT  "")
set(PERIDYNO_ASSET_PATH "${PERIDYNO_ROOT}/data" CACHE PATH "" FORCE)
set(PERIDYNO_PLUGIN_PATH "${CMAKE_BINARY_DIR}/bin"  CACHE PATH "" FORCE)
#file(COPY "data/" DESTINATION "data")


if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
	project(peridyno LANGUAGES ${SUPPORTED_LANGUAGES})  

    #find_package(CUDA REQUIRED)
    # if(CUDA_FOUND)
    #     include_directories(${CUDA_INCLUDE_DIRS})
    #     link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    #     link_libraries(${CUDA_LIBRARIES})
    # else()
    #     message("Cuda not found")
    # endif()      

	set(CMAKE_POSITION_INDEPENDENT_CODE ON)                                                         #默认为本项目下各编译目标开启fPIC模式，cuda下面会开启rdc模式
	#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "DEFAULT SET SEPERABLE COMPILATION MODE")    #默认为本项目下各编译目标指定cuda分离编译模式
	set(CUDA_ARCH_FLAGS 60 CACHE STRING "cuda architectures" FORCE)                                 #默认为所有cuda代码生成computability7.0下二进制代码
else()
	include_directories(external)
	include_directories(external/glm-0.9.9.7)
	include_directories(external/gli)
	include_directories(external/imgui)
	include_directories(external/tinygltf)
	include_directories(external/ktx/include)
	include_directories(external/ktx/other_include)
	include_directories(Source/base)
endif()


#set output directories

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})

IF(MSVC)
    SET( CMAKE_DEBUG_POSTFIX "d" )
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
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
option(PERIDYNO_LIBRARY_PLUGIN "Enable binding the plugin libraries" ON)
option(PERIDYNO_PYTHON_BINDING "Enable python binding with pybind11" OFF)

option(PERIDYNO_TESTS "Enable python binding with pybind11" OFF)
if(PERIDYNO_TESTS)
	add_subdirectory(external/gtest)
	add_subdirectory(tests)
endif()

add_subdirectory(external/glfw-3.3.0)
add_subdirectory(external/glad-4.6)
add_subdirectory(external/imgui)

# VTK rendering support, off by default
option(PERIDYNO_PLUGIN_VTK "Enable VTK plugin for visualization" OFF)
# Qt GUI support, off by default
option(PERIDYNO_QT_GUI "Enable building Qt{5 or 6}-based applications" OFF)
if(PERIDYNO_QT_GUI)
	add_subdirectory(external/nodeeditor)
endif()

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
	file(GLOB FILESYSTEM_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/external/filesystem/ghc/*.*")
	install(FILES ${FILESYSTEM_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/external/filesystem/ghc)
elseif("${PERIDYNO_GPU_BACKEND}" STREQUAL "Vulkan")

	# Use FindVulkan module added with CMAKE 3.7
	if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
		message(STATUS "Using module to find Vulkan")
		find_package(Vulkan)
	endif()

	IF(UNIX AND NOT APPLE)
		set(LINUX TRUE)
	ENDIF()

	IF(WIN32)
		IF (NOT Vulkan_FOUND)
			find_library(Vulkan_LIBRARY NAMES vulkan-1 vulkan PATHS ${PERIDYNO_ROOT}/libs/vulkan)
			IF (Vulkan_LIBRARY)
				set(Vulkan_FOUND ON)
				MESSAGE("Using bundled Vulkan library version")
			ENDIF()
		ENDIF()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
	ELSEIF(LINUX)
		IF (NOT Vulkan_FOUND)
			find_library(Vulkan_LIBRARY NAMES vulkan HINTS "$ENV{VULKAN_SDK}/lib" "${CMAKE_SOURCE_DIR}/libs/vulkan" REQUIRED)
			IF (Vulkan_LIBRARY)
				set(Vulkan_FOUND ON)
				MESSAGE("Using bundled Vulkan library version")
			ENDIF()
		ENDIF()
		find_package(Threads REQUIRED)
		IF(USE_D2D_WSI)
			MESSAGE("Using direct to display extension...")
			add_definitions(-D_DIRECT2DISPLAY)
		ELSEIF(USE_DIRECTFB_WSI)
			find_package(DirectFB REQUIRED)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_DIRECTFB_EXT")
			include_directories(${DIRECTFB_INCLUDE_DIR})
		ELSEIF(USE_WAYLAND_WSI)
			find_program(PKG_CONFIG pkg-config)
			if (NOT PKG_CONFIG)
				message(FATAL_ERROR "pkg-config binary not found")
			endif ()
			find_package(Wayland REQUIRED)
			if (NOT WAYLAND_FOUND)
				message(FATAL_ERROR "Wayland development package not found")
			endif ()
			pkg_check_modules(WAYLAND_PROTOCOLS REQUIRED wayland-protocols)
			if (NOT WAYLAND_PROTOCOLS_FOUND)
				message(FATAL_ERROR "Wayland protocols package not found")
			endif ()
			find_program(WAYLAND_SCANNER wayland-scanner)
			if (NOT WAYLAND_SCANNER)
				message(FATAL_ERROR "wayland-scanner binary not found")
			endif ()
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WAYLAND_KHR")
			include_directories(${WAYLAND_INCLUDE_DIR})
			execute_process(COMMAND ${PKG_CONFIG} --variable=pkgdatadir wayland-protocols OUTPUT_VARIABLE protocol_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
			execute_process(COMMAND ${WAYLAND_SCANNER} client-header ${protocol_dir}/stable/xdg-shell/xdg-shell.xml ${CMAKE_BINARY_DIR}/xdg-shell-client-protocol.h
					COMMAND ${WAYLAND_SCANNER} private-code ${protocol_dir}/stable/xdg-shell/xdg-shell.xml ${CMAKE_BINARY_DIR}/xdg-shell-protocol.c)
			include_directories(${CMAKE_BINARY_DIR})
		ELSE(USE_D2D_WSI)
			#find_package(XCB REQUIRED)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR")
		ENDIF(USE_D2D_WSI)
	ELSEIF(APPLE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_MACOS_MVK -DVK_EXAMPLE_XCODE_GENERATED")
		# Todo : android?
	ENDIF(WIN32)

	IF (NOT Vulkan_FOUND)
		message(FATAL_ERROR "Could not find Vulkan library!")
	ELSE()
		message(STATUS ${Vulkan_LIBRARY})
	ENDIF()

	# Set preprocessor defines
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES")

	# Clang specific stuff
	if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch-enum")
	endif()


	if("${PERIDYNO_GPU_BACKEND}" STREQUAL "Vulkan")
		add_subdirectory(${PERIDYNO_ROOT}/external/glslang)
		find_program(PyExe NAMES python python3)
		add_custom_target(CompileShader ALL 
			COMMAND ${PyExe} ${PROJECT_SOURCE_DIR}/data/shaders/glsl/compileshaders.py --glslang "$<TARGET_FILE:glslangValidator>"
			DEPENDS glslangValidator
		)
		set_property(TARGET CompileShader PROPERTY FOLDER glslang)
	endif()
else()

endif()


if(PERIDYNO_LIBRARY_PLUGIN)
	option(PERIDYNO_PLUGIN_ALEMBIC "Enable binding the abc exporter library" OFF)
	option(PERIDYNO_PLUGIN_GMSH "Enable binding the Gmsh library" OFF)

	if(PERIDYNO_PLUGIN_ALEMBIC)
		add_subdirectory(external/Imath)
		add_subdirectory(external/alembic/)
	endif()

	if(PERIDYNO_PLUGIN_GMSH)
		add_subdirectory(external/gmsh-fork)
	endif()
endif()

# Add Plugin folder
if(PERIDYNO_LIBRARY_PLUGIN)
	add_subdirectory(plugins)
endif()

add_subdirectory(src)


option(PERIDYNO_EXAMPLE "Enable building examples" ON)
if(PERIDYNO_EXAMPLE)
	add_subdirectory(examples)
endif()

if(PERIDYNO_PYTHON_BINDING)
	add_subdirectory(python)
endif()

