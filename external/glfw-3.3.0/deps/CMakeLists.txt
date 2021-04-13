set(GLAD_GL_SRC
    "${GLFW_SOURCE_DIR}/deps/glad/gl.h"
    "${GLFW_SOURCE_DIR}/deps/glad_gl.c")

add_library(glad STATIC ${GLAD_GL_SRC})
set_target_properties(glad PROPERTIES
    OUTPUT_NAME "glad-2.0.0")

set_target_properties(glad PROPERTIES FOLDER "ThirdParty")

target_include_directories(glad PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)