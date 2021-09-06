set(LIB_NAME GLRenderEngine)

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

file(GLOB_RECURSE SOURCES *.cpp *.h *.c *.cu)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
add_library(${LIB_NAME} STATIC ${SOURCES} ${SHADER_FILES} ${SHADER_HEADER_FILE})

target_link_libraries(${LIB_NAME} PUBLIC
	Core 
	Framework 
	RenderCore 
	glad 
	imgui)

target_include_directories(${LIB_NAME} 
	PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
)

#To disable the warning "calling a constexpr __host__ function("***") from a __host__ __device__ function("***") is not allowed."
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

#Tell the required compilers to enable device linking on the library target.
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Rendering/Engine")
set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "60")

#To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL>)

#add_dependencies(${LIB_NAME} BuildShaderHeader)
install(TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}Targets
    RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
    )

install(EXPORT ${LIB_NAME}Targets DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
    FILE ${LIB_NAME}Targets.cmake)

get_property(LOCAL_CMAKES_NAMES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")
list(APPEND LOCAL_CMAKES_NAMES "${LIB_NAME}Targets.cmake")    
set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES ${LOCAL_CMAKES_NAMES})

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL)

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/gl/*.h")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/gl)

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/shader/*.*")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/shader)

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/shader/helper/*.*")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/OpenGL/shader/helpler)