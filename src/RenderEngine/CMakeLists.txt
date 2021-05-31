set(LIB_NAME RenderEngine)
# precompile shaders
set(SHADER_HEADER_FILE ${CMAKE_BINARY_DIR}/shader_header.h)
set(SHADER_BUILD_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/shader/shader.cmake)
set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shader)

file(GLOB_RECURSE SHADER_FILES ${SHADER_SOURCE_DIR}/*.glsl)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SHADER_FILES})

add_custom_command(
	PRE_BUILD
	OUTPUT ${SHADER_HEADER_FILE}
	COMMAND ${CMAKE_COMMAND} "-DSHADER_FILES=${SHADER_FILES}" "-DSHADER_HEADER_FILE=${SHADER_HEADER_FILE}" -P ${SHADER_BUILD_SCRIPT}
	DEPENDS ${SHADER_FILES} ${SHADER_BUILD_SCRIPT}
	)	
# add_custom_target(BuildShaderHeader
# 	DEPENDS ${SHADER_HEADER_FILE}
# )

file(GLOB_RECURSE SOURCES *.cpp *.h *.c)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
add_library(${LIB_NAME} STATIC ${SOURCES} ${SHADER_FILES} ${SHADER_HEADER_FILE})

target_link_libraries(${LIB_NAME} Core Framework glad)

target_include_directories(${LIB_NAME} 
	PRIVATE ${CMAKE_BINARY_DIR}
)

set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine")

#To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/RenderEngine>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>)

#add_dependencies(${LIB_NAME} BuildShaderHeader)
