cmake_minimum_required(VERSION 3.10)

include_directories("${CMAKE_SOURCE_DIR}/src")                                                              #默认为本项目下各编译目标添加Source/ 目录

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "Vulkan")
    add_subdirectory(ImGUI)
endif()

add_subdirectory(ImWidgets)
add_subdirectory(GlfwGUI)

if(PERIDYNO_Qt5_GUI)
    add_subdirectory(QtGUI)
endif()