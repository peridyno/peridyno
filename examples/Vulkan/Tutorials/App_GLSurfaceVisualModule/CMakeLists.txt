set(PROJECT_NAME App_GLSurfaceVisualModule)

set(LIB_SRC Cloth.h Cloth.cpp main.cpp)
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
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_FLAGS}")

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
elseif(UNIX)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    else()
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    endif()
endif()

if (MSVC)
    # Tell MSVC to use main instead of WinMain for Windows subsystem executables
    set_target_properties(${GUI_ONLY_BINARIES} PROPERTIES
                          LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()