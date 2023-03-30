cmake_minimum_required(VERSION 3.10)

project(glslangext)
set(TargetLib glslangext)
set(TargetExe glslangplugin)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJ_EXTERNAL_SRC ../../external)
set(GLSLANG_SRC ${PROJ_EXTERNAL_SRC}/glslang)

include_directories(${PROJ_EXTERNAL_SRC})
include_directories(${GLSLANG_SRC})

add_subdirectory(${GLSLANG_SRC} ${CMAKE_BINARY_DIR}/glslang)

file(
        GLOB
        SRC
        *.h
        *.cpp
)

include_directories(${ANDROID_NDK}/sources/android/native_app_glue)

if (WIN32)
    set(PLATFORM win)
    set(GLSLANG_LIB_DEPEND glslang SPIRV glslang-default-resource-limits)
    set(LIB_TYPE STATIC)
else ()
    set(PLATFORM ${ANDROID_ABI})
    set(GLSLANG_LIB_DEPEND glslang SPIRV glslang-default-resource-limits android)
    set(LIB_TYPE SHARED)
endif ()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output/libs/${PLATFORM})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/libs/${PLATFORM})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output/bin/${PLATFORM})

add_library(
        ${TargetLib}
        ${LIB_TYPE}
        ${SRC}
)

target_link_libraries(
        ${TargetLib}
        ${GLSLANG_LIB_DEPEND}
)

set_property(TARGET ${TargetLib} PROPERTY FOLDER glslang)

if (WIN32)
    add_executable(
            ${TargetExe}
            main.cpp
    )

    target_link_libraries(
            ${TargetExe}
            ${TargetLib}
    )
    set_property(TARGET ${TargetExe} PROPERTY FOLDER glslang)
endif ()