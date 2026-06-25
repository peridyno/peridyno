cmake_minimum_required(VERSION 3.16)
project(oit_demo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# repo external/ deps (header-only glm, glad loader)
set(PERIDYNO_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/../external")

# Mesa EGL (software GL via llvmpipe) - Homebrew keg
set(MESA_PREFIX "/opt/homebrew" CACHE PATH "Homebrew mesa prefix")
find_path(EGL_INCLUDE_DIR EGL/egl.h
    HINTS ${MESA_PREFIX}/include /opt/homebrew/opt/mesa/include /usr/local/include)
find_library(EGL_LIBRARY NAMES EGL
    HINTS ${MESA_PREFIX}/lib /opt/homebrew/opt/mesa/lib /usr/local/lib)

if(NOT EGL_INCLUDE_DIR OR NOT EGL_LIBRARY)
    message(FATAL_ERROR "Mesa EGL not found. `brew install mesa`. "
            "Got inc=${EGL_INCLUDE_DIR} lib=${EGL_LIBRARY}")
endif()

add_executable(oit_demo
    main.cpp
    ${PERIDYNO_EXTERNAL}/glad-4.6/glad/glad.c)

target_include_directories(oit_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PERIDYNO_EXTERNAL}/glad-4.6
    ${PERIDYNO_EXTERNAL}/glm-0.9.9.7
    ${EGL_INCLUDE_DIR})

target_link_libraries(oit_demo PRIVATE ${EGL_LIBRARY})
