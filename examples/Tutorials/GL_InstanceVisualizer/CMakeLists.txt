set(PROJECT_NAME GL_InstanceVisualizer)

set(LIB_SRC main.cpp)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${LIB_SRC})

add_executable(${PROJECT_NAME} ${LIB_SRC})

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set(GUI_ONLY_BINARIES ${PROJECT_NAME})

set_target_properties(${GUI_ONLY_BINARIES} PROPERTIES
                      C_STANDARD 99
                      FOLDER "Examples/Tutorials")

target_link_libraries(${PROJECT_NAME} GlfwGUI GLRenderEngine)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples/Tutorials")

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${GUI_ONLY_BINARIES} PROPERTIES
                          LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()