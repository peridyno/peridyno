set(GLAD_GL_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/glad/glad.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/glad/glad.c")

add_library(glad STATIC ${GLAD_GL_SRC})
set_target_properties(glad PROPERTIES
    OUTPUT_NAME "glad-2.0.0")

set_target_properties(glad PROPERTIES FOLDER "ThirdParty")

target_include_directories(glad PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)